"""Evaluator:轨迹-规约一致性评估器(奖励函数)。

对每条 rollout 轨迹,计算其与 FlowGraph 目标通路的符合度 reward ∈ [0, 1]:

  R = w1·节点覆盖率 + w2·守卫正确率 + w3·计数器正确性 + w4·终态正确性 - 惩罚项

其中守卫/终态判定由 LLM-as-Judge 完成(轨迹语义 vs 规约语义),
计数器由确定性规则校验(state facts 中的计数字段)。

聚合层面:
  Repo 版本 V 的适应度 F(V) = 通路覆盖加权平均 reward
这是 MCTS backpropagation 回传的价值信号。
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from flow_graph import FlowGraph
from fsm_reward import FSMTraceVerifier
from llm_client import LLMClient
from simulator import Trajectory

JUDGE_PROMPT = """你是流程一致性审计员。给定目标流程通路和 Agent 实际对话轨迹,逐项打分。

## 目标通路(规约)
<<<PATH>>>

## 实际对话轨迹
<<<TRAJ>>>

请输出 JSON:
{
  "node_coverage": 0.0~1.0,        // 通路中的关键节点(话术/动作)有多少被 Agent 实际执行
  "guard_correctness": 0.0~1.0,    // Agent 的分支决策是否符合规约中的 guard 条件
  "terminal_correct": true/false,  // 最终终态(退出原因/建单结果)是否与通路一致
  "violations": ["违规点描述", ...],
  "reasoning": "简要说明"
}
"""


@dataclass
class PathScore:
    path_signature: str
    reward: float
    reward_std: float = 0.0
    reward_lcb: float = 0.0
    samples: int = 1
    fsm_reward: float = 0.0
    fsm_coverage: float = 0.0
    transition_correctness: float = 0.0
    verification_mode: str = "none"
    node_coverage: float = 0.0
    guard_correctness: float = 0.0
    counter_correct: bool = True
    terminal_correct: bool = False
    violations: list[str] = field(default_factory=list)


class Evaluator:
    """多目标奖励：形式化项为主，LLM 仅评估语义残差。

    R(τ) = 0.65 * R_FSM(τ) + 0.35 * R_semantic(τ) - penalties
    R_FSM 已包含节点/转移、计数器不变量和终态义务。
    """
    W_FSM, W_SEMANTIC = 0.65, 0.35

    def __init__(self, llm: Optional[LLMClient] = None,
                 graph: Optional[FlowGraph] = None):
        self.llm = llm or LLMClient()
        self.graph = graph
        self.fsm_verifier = FSMTraceVerifier()

    # ------------------------------------------------------------------
    def score(self, traj: Trajectory, counters_spec: dict[str, int]) -> PathScore:
        if traj.error:
            return PathScore(traj.path_signature, reward=0.0,
                             violations=[f"轨迹执行错误: {traj.error}"])

        judged = self._judge(traj)
        counter_ok = self._check_counters(traj, counters_spec)
        if self.graph is not None and traj.target_path:
            fsm = self.fsm_verifier.score(self.graph, traj.target_path, traj)
        else:
            from fsm_reward import FSMReward
            fsm = FSMReward(
                coverage=0.0,
                transition_correctness=0.0,
                counter_correctness=1.0 if counter_ok else 0.0,
                terminal_correctness=1.0 if judged.get("terminal_correct") else 0.0,
                dense_reward=(0.5 if counter_ok else 0.0)
                             + (0.5 if judged.get("terminal_correct") else 0.0),
                mode="legacy_fallback", obligations=2,
                satisfied=int(counter_ok) + int(bool(judged.get("terminal_correct"))),
            )
        semantic = 0.5 * judged.get("node_coverage", 0.0) \
            + 0.5 * judged.get("guard_correctness", 0.0)
        violations = list(dict.fromkeys(
            fsm.violations + judged.get("violations", [])
        ))
        reward = self.W_FSM * fsm.dense_reward + self.W_SEMANTIC * semantic
        reward -= 0.03 * len(violations)
        reward = max(0.0, min(1.0, reward))

        return PathScore(
            path_signature=traj.path_signature,
            reward=reward,
            reward_lcb=reward,
            fsm_reward=fsm.dense_reward,
            fsm_coverage=fsm.coverage,
            transition_correctness=fsm.transition_correctness,
            verification_mode=fsm.mode,
            node_coverage=judged.get("node_coverage", 0.0),
            guard_correctness=judged.get("guard_correctness", 0.0),
            counter_correct=counter_ok,
            terminal_correct=bool(judged.get("terminal_correct")),
            violations=violations,
        )

    # ------------------------------------------------------------------
    def _judge(self, traj: Trajectory) -> dict:
        traj_text = "\n".join(
            f"[轮{i+1}] 司机: {t.driver}\n        Agent: {t.agent}\n        状态: {json.dumps(t.state, ensure_ascii=False)}"
            for i, t in enumerate(traj.turns)
        ) + f"\n终态原因: {traj.terminal_reason or '(未到终态)'}"
        prompt = (JUDGE_PROMPT
                  .replace("<<<PATH>>>", traj.target_path_desc)
                  .replace("<<<TRAJ>>>", traj_text[:14000]))
        try:
            return self.llm.chat_json([{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001
            return {"node_coverage": 0.0, "guard_correctness": 0.0,
                    "terminal_correct": False, "violations": [f"judge 失败: {e}"]}

    # ------------------------------------------------------------------
    @staticmethod
    def _check_counters(traj: Trajectory, counters_spec: dict[str, int]) -> bool:
        """确定性校验:任何计数器不得超过规约阈值。"""
        field_map = {
            "off_topic": "off_topic_count",
            "type_mismatch": "type_mismatch_count",
            "not_recognized": "not_recognized_count",
            "postpone": "postpone_count",
            "invalid_desc": "invalid_desc_count",
        }
        for t in traj.turns:
            for spec_key, threshold in counters_spec.items():
                fact_key = field_map.get(spec_key, spec_key)
                v = t.state.get(fact_key)
                if isinstance(v, int) and v > threshold:
                    return False
        return True


def aggregate_repeated_scores(scores: list[PathScore],
                              confidence_z: float = 1.96) -> PathScore:
    """合并同一路径的重复 rollout，并计算均值的置信下界。

    MCTS 使用 LCB 而非裸均值，可显著降低一次偶然高分造成的错误扩展。
    """
    if not scores:
        raise ValueError("重复评分列表不能为空")
    signatures = {score.path_signature for score in scores}
    if len(signatures) != 1:
        raise ValueError("只能合并同一路径的评分")
    rewards = [score.reward for score in scores]
    mean = statistics.fmean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    stderr = std / math.sqrt(len(rewards))
    lcb = max(0.0, mean - confidence_z * stderr)
    violations = list(dict.fromkeys(
        violation for score in scores for violation in score.violations
    ))
    return PathScore(
        path_signature=scores[0].path_signature,
        reward=mean,
        reward_std=std,
        reward_lcb=lcb,
        samples=len(scores),
        fsm_reward=statistics.fmean(score.fsm_reward for score in scores),
        fsm_coverage=statistics.fmean(score.fsm_coverage for score in scores),
        transition_correctness=statistics.fmean(
            score.transition_correctness for score in scores
        ),
        verification_mode=("exact_trace" if all(
            score.verification_mode == "exact_trace" for score in scores
        ) else "observable_contract"),
        node_coverage=statistics.fmean(score.node_coverage for score in scores),
        guard_correctness=statistics.fmean(score.guard_correctness for score in scores),
        counter_correct=all(score.counter_correct for score in scores),
        terminal_correct=all(score.terminal_correct for score in scores),
        violations=violations,
    )


def aggregate_fitness(scores: list[PathScore]) -> float:
    """版本级保守适应度：置信下界均值 + 最坏路径正则。"""
    if not scores:
        return 0.0
    rewards = [s.reward_lcb if s.samples > 1 else s.reward for s in scores]
    mean = sum(rewards) / len(rewards)
    worst = min(rewards)
    return 0.8 * mean + 0.2 * worst  # 鼓励木桶补齐,避免只优化易走通路
