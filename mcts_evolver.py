"""MCTSEvolver:基于蒙特卡洛树搜索的 Agent 自迭代演化引擎。

搜索空间建模
============
- 状态 s   :Agent 代码库的一个版本(git commit)
- 动作 a   :一个语义补丁 Patch(针对失败通路的最小修复)
- 转移     :apply(patch, commit) → new commit
- 奖励 R(s):在该版本上,对流程规约 FlowGraph 的全部关键通路做
             LLM 司机模拟 rollout,由 Evaluator 打分后聚合的适应度 F ∈ [0,1]

四阶段
======
1. Selection  :UCT 选择——从根(baseline)沿树下行,
                UCT(v) = Q(v)/N(v) + c·sqrt(ln N(parent) / N(v))
                并加入 progressive widening 限制分支因子
2. Expansion  :对选中版本,取其最差通路的失败信号,
                由 Patcher 生成 k 个候选补丁 → 子节点
3. Simulation :对新版本执行通路 rollout 池(优先评估上一轮失败通路,
                预算内轮换全通路,防止过拟合单通路)
4. Backprop   :将适应度沿路径回传;若子节点适应度低于父节点(退化),
                施加退化惩罚,使搜索偏向单调改进链

回溯 / 收敛
===========
- 任意节点 ←→ git commit,可随时 checkout 回滚
- 收敛条件:F ≥ target_fitness 或达到迭代预算
- 输出:最优版本 commit + 演化报告(每次迭代的补丁、通路得分曲线)
"""
from __future__ import annotations

import json
import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evaluator import (Evaluator, PathScore, aggregate_fitness,
                       aggregate_repeated_scores)
from flow_graph import FlowGraph, Edge
from llm_client import LLMBudgetExceeded
from patcher import Patch, Patcher, RepoManager
from search_model import (GRAVEStatistics, ProgramTransitionModel,
                          SearchState)
from simulator import DriverSimulator


@dataclass
class TreeNode:
    commit: str
    patch: Optional[Patch] = None
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    fitness: float = -1.0                        # 最近一次完整评估的适应度
    path_scores: list[PathScore] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def uct(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        parent_n = self.parent.visits if self.parent else self.visits
        return self.q + c * math.sqrt(math.log(max(parent_n, 1)) / self.visits)


class MCTSEvolver:
    def __init__(self, graph: FlowGraph, repo: RepoManager,
                 simulator: DriverSimulator, evaluator: Evaluator,
                 patcher: Patcher, *,
                 c_uct: float = 1.2,
                 expand_k: int = 3,
                 max_children: int = 4,
                 rollout_paths_per_eval: int = 6,
                 repetitions_per_path: int = 2,
                 target_fitness: float = 0.95,
                 max_iterations: int = 30,
                 restart_cmd: Optional[str] = None,
                 regression_guard: bool = True,
                 regression_tol: float = 0.05,
                 full_sweep_on_finish: bool = True,
                 final_suite_max_paths: int = 128,
                 seed: int = 42,
                 report_dir: str = "evolve_reports"):
        random.seed(seed)
        self.graph = graph
        self.repo = repo
        self.simulator = simulator
        self.evaluator = evaluator
        self.patcher = patcher
        self.c_uct = c_uct
        self.expand_k = expand_k
        self.max_children = max_children
        self.rollout_budget = rollout_paths_per_eval
        self.repetitions_per_path = max(1, repetitions_per_path)
        self.target_fitness = target_fitness
        self.max_iterations = max_iterations
        self.restart_cmd = restart_cmd
        self.regression_guard = regression_guard
        self.regression_tol = regression_tol
        self.full_sweep_on_finish = full_sweep_on_finish
        self.final_suite_max_paths = max(0, final_suite_max_paths)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)

        self.all_paths: list[list[Edge]] = list(graph.enumerate_paths())
        random.shuffle(self.all_paths)
        self._path_cursor = 0
        self.history: list[dict] = []
        self._root_scores: list[PathScore] = []
        self._visited_trees: set[str] = set()
        self._spec_hash = hashlib.sha256(graph.to_json().encode()).hexdigest()[:16]
        self._transition_model = ProgramTransitionModel(repo, patcher)
        self._grave = GRAVEStatistics()

    # ==================================================================
    def evolve(self) -> TreeNode:
        root = TreeNode(commit=self.repo.head())
        self._visited_trees.add(self.repo.tree_hash(root.commit))
        self._event("experiment_start", commit=root.commit,
                    paths=len(self.all_paths), seed_state="configured")
        self._evaluate(root)
        self._root_scores = list(root.path_scores)
        best = root
        for it in range(1, self.max_iterations + 1):
            try:
                node = self._select(root)                   # 1. Selection
                child = self._expand(node)                  # 2. Expansion
                target = child or node
                self._evaluate(target)                      # 3. Simulation
                self._backpropagate(target)                 # 4. Backprop
            except LLMBudgetExceeded as e:
                print(f"[预算终止] {e}")
                break

            if target.fitness > best.fitness:
                best = target
            self._log_iteration(it, target, best)
            if best.fitness >= self.target_fitness:
                break
        # 回溯到最优版本
        self.repo.checkout(best.commit)
        # 全通路终检:抽样评估易有方差,收敛后对所有通路做一次完整验证
        if self.full_sweep_on_finish:
            self._final_sweep(best)
        self._dump_report(best)
        return best

    def _final_sweep(self, best: TreeNode) -> None:
        suite = self.graph.coverage_suite(max_paths=self.final_suite_max_paths)
        coverage = self.graph.coverage_ratio(suite)
        print(f"\n【覆盖完备终检】对最优版本 {best.commit[:10]} 评估 "
              f"{len(suite)} 条代表通路（控制流特征覆盖率 {coverage:.1%}）…")
        self.repo.checkout(best.commit)
        self._restart_agent()
        scores = self._eval_paths(suite)
        best.path_scores = scores
        best.fitness = aggregate_fitness(scores)
        passed = sum(1 for score in scores if score.reward >= 0.8)
        print(f"【覆盖完备终检】通过 {passed}/{len(scores)} 条,"
              f"完整适应度 F={best.fitness:.4f}")
        log_path = self.report_dir / "evolve_progress.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n{'=' * 70}\n【覆盖完备终检】{best.commit[:10]} "
                f"通过 {passed}/{len(scores)} 覆盖率={coverage:.1%} "
                f"完整F={best.fitness:.4f}\n"
            )
            for score in scores:
                if score.reward < 0.8:
                    stream.write(
                        f"  ✗ 通路 {score.path_signature} reward={score.reward:.2f} "
                        f"{score.violations}\n"
                    )

    # ==================================================================
    def _select(self, root: TreeNode) -> TreeNode:
        node = root
        while node.children:
            # progressive widening:访问足够多才允许继续加宽
            if (len(node.children) < self.max_children
                    and node.visits >= len(node.children) ** 2):
                return node
            node = max(node.children, key=lambda child: self._grave_score(node, child))
        return node

    def _grave_score(self, parent: TreeNode, child: TreeNode) -> float:
        if child.visits == 0:
            return float("inf")
        action_key = child.patch.grave_key if child.patch else "root"
        return self._grave.selection_score(
            child.q, child.visits, parent.visits, action_key, self.c_uct
        )

    # ==================================================================
    def _expand(self, node: TreeNode) -> Optional[TreeNode]:
        if node.fitness < 0:
            self._evaluate(node)
        failures = self._failure_report(node)
        if not failures:
            return None
        patches = self.patcher.propose(self.graph.describe(), failures,
                                       n_candidates=self.expand_k)
        existing = {ch.patch.action_id for ch in node.children if ch.patch}
        failure_signature = hashlib.sha256(failures.encode()).hexdigest()[:16]
        state = SearchState(
            commit=node.commit,
            tree_hash=self.repo.tree_hash(node.commit),
            spec_hash=self._spec_hash,
            failure_signature=failure_signature,
        )
        for patch in patches:
            if patch.action_id in existing:
                continue
            transition = self._transition_model.transition(
                state, patch, self._visited_trees
            )
            if not transition.accepted or not transition.next_commit:
                self._event("patch_rejected", action_id=patch.action_id,
                            grave_key=patch.grave_key,
                            reason=transition.reason)
                continue  # gate 未通过,自动回滚
            new_commit = transition.next_commit
            self._visited_trees.add(transition.next_tree_hash)
            # 回归门禁:补丁不得弄坏父节点已通过的通路
            if self.regression_guard and self._has_regression(node, new_commit):
                self.repo.checkout(node.commit)
                print(f"  ↳ 补丁 {patch.action_id} 引发回归,已拒绝")
                self._event("patch_rejected", action_id=patch.action_id,
                            reason="statistical_regression")
                continue
            child = TreeNode(commit=new_commit, patch=patch, parent=node)
            node.children.append(child)
            return child  # 每次迭代扩展一个,其余候选留给后续 widening
        return None

    def _has_regression(self, parent: TreeNode, new_commit: str) -> bool:
        """在新版本上复评父节点已通过(reward≥0.8)的通路。"""
        passed = [s.path_signature for s in parent.path_scores if s.reward >= 0.8]
        if not passed:
            return False
        check = [p for p in self.all_paths
                 if self.graph.path_signature(p) in passed][:max(2, self.rollout_budget // 2)]
        if not check:
            return False
        self.repo.checkout(new_commit)
        self._restart_agent()
        new_scores = {s.path_signature: s.reward_lcb for s in self._eval_paths(check)}
        for s in parent.path_scores:
            if s.reward >= 0.8 and s.path_signature in new_scores:
                baseline = s.reward_lcb if s.samples > 1 else s.reward
                if new_scores[s.path_signature] < baseline - self.regression_tol:
                    return True
        return False

    # ==================================================================
    def _evaluate(self, node: TreeNode) -> None:
        self.repo.checkout(node.commit)
        self._restart_agent()
        batch = self._next_path_batch(node)
        node.path_scores = self._eval_paths(batch)
        node.fitness = aggregate_fitness(node.path_scores)

    def _eval_paths(self, paths: list[list[Edge]]) -> list[PathScore]:
        """在当前已 checkout 的版本上评估一组通路。"""
        scores: list[PathScore] = []
        for path in paths:
            mode = "again" if any("再次" in (e.guard + e.label) for e in path) else "first"
            repeated: list[PathScore] = []
            for _ in range(self.repetitions_per_path):
                traj = self.simulator.rollout(self.graph, path, mode=mode)
                repeated.append(self.evaluator.score(traj, self.graph.counters))
            scores.append(aggregate_repeated_scores(repeated))
        return scores

    def _restart_agent(self) -> None:
        """切换版本后重启被测服务,使补丁生效。"""
        if not self.restart_cmd:
            return
        import subprocess
        subprocess.run(self.restart_cmd, shell=True, cwd=self.repo.repo,
                       capture_output=True, timeout=60)
        time.sleep(2)

    def _next_path_batch(self, node: TreeNode) -> list[list[Edge]]:
        """优先复评父节点失败通路(定向验证补丁),剩余预算轮换全通路池。"""
        batch: list[list[Edge]] = []
        if node.parent and node.parent.path_scores:
            failed_sigs = {s.path_signature for s in node.parent.path_scores
                           if s.reward < 0.8}
            batch += [p for p in self.all_paths
                      if self.graph.path_signature(p) in failed_sigs][:self.rollout_budget // 2]
        while len(batch) < self.rollout_budget and self.all_paths:
            batch.append(self.all_paths[self._path_cursor % len(self.all_paths)])
            self._path_cursor += 1
        return batch

    # ==================================================================
    def _backpropagate(self, node: TreeNode) -> None:
        value = node.fitness
        # 退化惩罚:低于父节点适应度,回传打折,抑制该分支
        if node.parent and node.parent.fitness >= 0 and value < node.parent.fitness:
            value *= 0.5
        cur: Optional[TreeNode] = node
        seen_action_keys: set[str] = set()
        while cur is not None:
            cur.visits += 1
            cur.total_value += value
            if cur.patch and cur.patch.grave_key not in seen_action_keys:
                self._grave.update(cur.patch.grave_key, value)
                seen_action_keys.add(cur.patch.grave_key)
            cur = cur.parent

    # ==================================================================
    def _failure_report(self, node: TreeNode) -> str:
        lines = []
        for s in sorted(node.path_scores,
                        key=lambda x: x.reward_lcb if x.samples > 1 else x.reward)[:3]:
            conservative = s.reward_lcb if s.samples > 1 else s.reward
            if conservative >= 0.95:
                continue
            lines.append(f"通路 {s.path_signature}: mean={s.reward:.2f} "
                         f"LCB={conservative:.2f} std={s.reward_std:.2f} n={s.samples} "
                         f"节点覆盖={s.node_coverage:.2f} 守卫={s.guard_correctness:.2f} "
                         f"终态正确={s.terminal_correct}")
            for v in s.violations:
                lines.append(f"  - 违规: {v}")
        return "\n".join(lines)

    def _log_iteration(self, it: int, target: TreeNode, best: TreeNode) -> None:
        entry = {
            "iteration": it,
            "time": time.strftime("%H:%M:%S"),
            "commit": target.commit[:10],
            "patch": target.patch.strategy if target.patch else "(root/re-eval)",
            "root_cause": target.patch.root_cause if target.patch else "",
            "fitness": round(target.fitness, 4),
            "best_fitness": round(best.fitness, 4),
            "path_scores": [{"sig": s.path_signature, "reward": round(s.reward, 3),
                             "lcb": round(s.reward_lcb, 3),
                             "std": round(s.reward_std, 3),
                             "samples": s.samples,
                             "terminal_ok": s.terminal_correct,
                             "violations": s.violations} for s in target.path_scores],
        }
        self.history.append(entry)
        self._event("iteration_complete", **entry)
        print(f"[iter {it:02d}] F={entry['fitness']:.3f} best={entry['best_fitness']:.3f} "
              f"| {entry['patch'][:50]}")
        # 详细进度日志:修改的 Agent 逻辑(diff)/指标改进/迭代进度
        log_path = self.report_dir / "evolve_progress.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n【迭代 {it}/{self.max_iterations}】 {entry['time']} "
                    f"commit={entry['commit']}\n")
            if target.patch:
                f.write(f"▸ 根因分析: {target.patch.root_cause}\n")
                f.write(f"▸ 修改策略: {target.patch.strategy}\n")
                try:
                    diff = self.repo._git("show", "--stat", "--format=", target.commit)
                    f.write(f"▸ Agent 逻辑变更(diff --stat):\n{diff}\n")
                except Exception:
                    pass
            parent_f = target.parent.fitness if target.parent else None
            delta = (f"{target.fitness - parent_f:+.4f}" if parent_f is not None
                     and parent_f >= 0 else "n/a")
            f.write(f"▸ 适应度: {entry['fitness']} (相对父节点 Δ={delta}) "
                    f"历史最优={entry['best_fitness']}\n")
            f.write("▸ 通路明细:\n")
            for ps in entry["path_scores"]:
                f.write(f"    通路 {ps['sig']}: mean={ps['reward']} "
                    f"LCB={ps['lcb']} std={ps['std']} n={ps['samples']} "
                        f"终态{'✓' if ps['terminal_ok'] else '✗'}\n")
                for v in ps["violations"]:
                    f.write(f"      ✗ {v}\n")

    def _event(self, event: str, **payload) -> None:
        """追加结构化 JSONL 事件，便于复现实验与生成论文曲线。"""
        record = {"event": event, "timestamp": time.time(), **payload}
        path = self.report_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _dump_report(self, best: TreeNode) -> None:
        chain = []
        cur: Optional[TreeNode] = best
        while cur:
            chain.append({
                "commit": cur.commit[:10],
                "fitness": round(cur.fitness, 4),
                "patch": cur.patch.strategy if cur.patch else "baseline",
                "root_cause": cur.patch.root_cause if cur.patch else "",
            })
            cur = cur.parent
        chain = list(reversed(chain))
        root = chain[0] if chain else {}
        # 迭代前后对比总结
        baseline_scores = self._root_scores or []
        best_scores = best.path_scores
        summary_lines = [
            "【迭代前后对比总结】",
            f"适应度: {root.get('fitness', 'n/a')} → {round(best.fitness, 4)}",
            f"终态正确通路: {sum(1 for s in baseline_scores if s.terminal_correct)}"
            f"/{len(baseline_scores)} → "
            f"{sum(1 for s in best_scores if s.terminal_correct)}/{len(best_scores)}",
            f"计数器合规通路: {sum(1 for s in baseline_scores if s.counter_correct)}"
            f"/{len(baseline_scores)} → "
            f"{sum(1 for s in best_scores if s.counter_correct)}/{len(best_scores)}",
            f"剩余违规总数: {sum(len(s.violations) for s in baseline_scores)} → "
            f"{sum(len(s.violations) for s in best_scores)}",
            "提升方面(演化链逐步修复):",
        ] + [f"  {i}. {c['patch']} (F→{c['fitness']})"
             for i, c in enumerate(chain[1:], 1)]
        summary = "\n".join(summary_lines)
        report = {
            "best_commit": best.commit,
            "best_fitness": best.fitness,
            "llm_calls": getattr(self.evaluator.llm, "calls", None),
            "comparison_summary": summary,
            "evolution_chain": chain,
            "iterations": self.history,
        }
        out = self.report_dir / f"report_{int(time.time())}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n" + summary)
        print(f"演化报告: {out}")
