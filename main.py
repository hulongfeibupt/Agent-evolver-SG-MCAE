"""agent_evolver 主入口。

用法:
  # 1. 仅解析流程图 → 生成/缓存形式化规约
  python main.py parse --flow ../whiteboard.pdf --prd ../绕路产品需求文档.md

  # 2. 全流程演化(需先启动被测 Agent 的 mock UI server)
  python main.py evolve --flow ../whiteboard.pdf --prd ../绕路产品需求文档.md \
      --repo .. --agent-url http://127.0.0.1:18081 --iterations 20

  # 3. 只做一次评估(不打补丁),用于基线打分
  python main.py assess --flow ../whiteboard.pdf --repo ..
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator import Evaluator, aggregate_fitness
from flow_graph import FlowGraph
from flow_parser import FlowParser
from llm_client import LLMClient
from mcts_evolver import MCTSEvolver
from patcher import Patcher, RepoManager
from simulator import AgentUnderTest, DriverSimulator
from spec_overrides import apply_overrides, REQUIREMENTS_TEXT, OVERRIDE_VERSION
from spec_validator import FlowSpecValidator

SPEC_CACHE = Path(__file__).parent / "flow_spec.json"


def load_or_parse_graph(flow: str, prd: str, force: bool = False) -> FlowGraph:
    if SPEC_CACHE.exists() and not force:
        print(f"使用缓存规约: {SPEC_CACHE}")
        graph = FlowGraph.from_json(SPEC_CACHE.read_text(encoding="utf-8"))
    else:
        print(f"解析流程图: {flow}")
        graph = FlowParser().parse(flow, prd_path=prd)
        SPEC_CACHE.write_text(graph.to_json(), encoding="utf-8")
        print(f"规约已缓存 → {SPEC_CACHE}")
    if graph.meta.get("override_version", 0) < OVERRIDE_VERSION:
        graph = apply_overrides(graph)
        SPEC_CACHE.write_text(graph.to_json(), encoding="utf-8")
        print(f"已施加需求增量: {graph.meta['overrides']}")
    report = FlowSpecValidator().validate(graph)
    print(report.format())
    report.raise_for_errors()
    return graph


def cmd_parse(args):
    graph = load_or_parse_graph(args.flow, args.prd, force=True)
    print(graph.describe())
    paths = list(graph.enumerate_paths())
    print(f"\n共枚举出 {len(paths)} 条 start→terminal 通路")


def cmd_validate(args):
    graph = load_or_parse_graph(args.flow, args.prd)
    print(f"规约规模: nodes={len(graph.nodes)} edges={len(graph.edges)} "
          f"terminals={len(graph.terminal_nodes)} "
          f"paths={sum(1 for _ in graph.enumerate_paths())}")


def cmd_assess(args):
    graph = load_or_parse_graph(args.flow, args.prd)
    sim = DriverSimulator(agent=AgentUnderTest(args.agent_url))
    ev = Evaluator(graph=graph)
    scores = []
    for i, path in enumerate(graph.enumerate_paths()):
        if i >= args.max_paths:
            break
        traj = sim.rollout(graph, path)
        s = ev.score(traj, graph.counters)
        scores.append(s)
        print(f"通路 {s.path_signature}: reward={s.reward:.3f} 违规={s.violations}")
    print(f"\n基线适应度 F = {aggregate_fitness(scores):.4f}")


def cmd_evolve(args):
    graph = load_or_parse_graph(args.flow, args.prd)
    # 将需求增量拼入规约描述,确保补丁 LLM 看到 R1~R3 与硬约束
    graph.meta["requirements_text"] = REQUIREMENTS_TEXT
    _orig_describe = graph.describe
    graph.describe = lambda: _orig_describe() + "\n" + REQUIREMENTS_TEXT  # type: ignore
    repo = RepoManager(args.repo)
    llm = LLMClient()
    evolver = MCTSEvolver(
        graph=graph,
        repo=repo,
        simulator=DriverSimulator(llm=llm, agent=AgentUnderTest(args.agent_url)),
        evaluator=Evaluator(llm=llm, graph=graph),
        patcher=Patcher(repo, llm=llm,
                        source_globs=args.sources.split(",") if args.sources else None,
                        protected_globs=args.protected.split(",") if args.protected else None,
                        protected_literals=([l for l in args.protected_literals.split("|") if l]
                                            if args.protected_literals else None),
                        build_cmd=args.build_cmd.split() if args.build_cmd else None),
        max_iterations=args.iterations,
        target_fitness=args.target,
        rollout_paths_per_eval=args.rollouts,
        repetitions_per_path=args.repetitions,
        restart_cmd=args.restart_cmd or None,
        regression_guard=not args.no_regression_guard,
        full_sweep_on_finish=not args.no_full_sweep,
        final_suite_max_paths=args.final_suite_max_paths,
        seed=args.seed,
    )
    best = evolver.evolve()
    print(f"\n✅ 演化完成 最优版本 {best.commit[:10]} 适应度 F={best.fitness:.4f}")


def main():
    p = argparse.ArgumentParser(description="MCTS 驱动的 Agent 自迭代演化框架")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--flow", default="../whiteboard.pdf", help="流程图文件(pdf/png/md)")
        sp.add_argument("--prd", default="../绕路产品需求文档.md", help="产品需求文档")
        sp.add_argument("--agent-url", default="http://127.0.0.1:18081")

    sp = sub.add_parser("parse"); common(sp); sp.set_defaults(fn=cmd_parse)

    sp = sub.add_parser("validate"); common(sp); sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("assess"); common(sp)
    sp.add_argument("--repo", default="..")
    sp.add_argument("--max-paths", type=int, default=8)
    sp.set_defaults(fn=cmd_assess)

    sp = sub.add_parser("evolve"); common(sp)
    sp.add_argument("--repo", default="..")
    sp.add_argument("--iterations", type=int, default=20)
    sp.add_argument("--target", type=float, default=0.95)
    sp.add_argument("--rollouts", type=int, default=6, help="每次评估的通路数")
    sp.add_argument("--repetitions", type=int, default=2, help="每条通路重复 rollout 次数")
    sp.add_argument("--sources", default="", help="逗号分隔的可修改源码 glob")
    sp.add_argument("--protected", default="", help="逗号分隔的禁改文件 glob(固定接口/话术)")
    sp.add_argument("--protected-literals", default="", help="| 分隔的受保护话术字面量")
    sp.add_argument("--no-regression-guard", action="store_true", help="关闭回归门禁")
    sp.add_argument("--no-full-sweep", action="store_true", help="关闭全通路终检")
    sp.add_argument("--final-suite-max-paths", type=int, default=128,
                    help="终检覆盖套件最大路径数，0 表示不限制")
    sp.add_argument("--seed", type=int, default=42, help="随机种子(可复现)")
    sp.add_argument("--restart-cmd", default="", help="切换版本后重启被测服务的 shell 命令")
    sp.add_argument("--build-cmd", default="go build ./...", help="gate 编译命令,空则跳过")
    sp.set_defaults(fn=cmd_evolve)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
