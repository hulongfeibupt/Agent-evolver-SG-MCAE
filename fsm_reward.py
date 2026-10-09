"""可验证 FSM 稠密奖励。

优先使用被测 Agent 暴露的 ``facts.fsm_trace``（节点 ID 序列）做精确对齐；
若 demo 尚未提供该诊断字段，则退化为基于计数器、终态和禁用分支的
可观测义务验证。两种模式会显式记录，避免把 LLM 判断伪装成形式化证明。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flow_graph import Edge, FlowGraph
from simulator import Trajectory


_COUNTER_FIELDS = {
    "题外计数": "off_topic_count",
    "off_topic": "off_topic_count",
    "类型不一致计数": "type_mismatch_count",
    "type_mismatch": "type_mismatch_count",
    "不认可计数": "not_recognized_count",
    "not_recognized": "not_recognized_count",
    "暂缓计数": "postpone_count",
    "postpone": "postpone_count",
    "描述无效计数": "invalid_desc_count",
    "invalid_desc": "invalid_desc_count",
}


@dataclass
class FSMReward:
    coverage: float
    transition_correctness: float
    counter_correctness: float
    terminal_correctness: float
    dense_reward: float
    mode: str
    obligations: int
    satisfied: int
    violations: list[str] = field(default_factory=list)


class FSMTraceVerifier:
    """将目标路径与运行轨迹对齐并产生 [0,1] 稠密奖励。"""

    W_COVERAGE = 0.35
    W_TRANSITION = 0.25
    W_COUNTER = 0.25
    W_TERMINAL = 0.15

    def score(self, graph: FlowGraph, path: list[Edge],
              trajectory: Trajectory) -> FSMReward:
        traces = self._fsm_traces(trajectory)
        if traces:
            coverage, transition, trace_violations = self._exact_alignment(path, traces)
            mode = "exact_trace"
        else:
            coverage, transition, trace_violations = self._observable_alignment(
                graph, path, trajectory
            )
            mode = "observable_contract"
        counter, counter_violations, counter_total, counter_ok = self._counter_obligations(
            path, trajectory
        )
        terminal, terminal_violations = self._terminal_obligation(graph, path, trajectory)
        violations = trace_violations + counter_violations + terminal_violations
        dense = (
            self.W_COVERAGE * coverage
            + self.W_TRANSITION * transition
            + self.W_COUNTER * counter
            + self.W_TERMINAL * terminal
        )
        observable_total = max(1, len(path) + counter_total + 1)
        satisfied = round(coverage * len(path)) + counter_ok + int(terminal == 1.0)
        return FSMReward(
            coverage=coverage,
            transition_correctness=transition,
            counter_correctness=counter,
            terminal_correctness=terminal,
            dense_reward=max(0.0, min(1.0, dense)),
            mode=mode,
            obligations=observable_total,
            satisfied=satisfied,
            violations=violations,
        )

    @staticmethod
    def _fsm_traces(trajectory: Trajectory) -> list[list[str]]:
        traces: list[list[str]] = []
        for turn in trajectory.turns:
            raw = turn.state.get("fsm_trace")
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                traces.append(raw)
        return traces

    @staticmethod
    def _exact_alignment(path: list[Edge], traces: list[list[str]]) -> tuple[float, float, list[str]]:
        expected_nodes = [path[0].src] + [edge.dst for edge in path] if path else []
        actual: list[str] = []
        for trace in traces:
            for node in trace:
                if not actual or actual[-1] != node:
                    actual.append(node)
        cursor = 0
        matched = 0
        for node in actual:
            if cursor < len(expected_nodes) and node == expected_nodes[cursor]:
                matched += 1
                cursor += 1
        coverage = matched / len(expected_nodes) if expected_nodes else 1.0
        expected_edges = set(zip(expected_nodes, expected_nodes[1:]))
        actual_edges = set(zip(actual, actual[1:]))
        transition = (len(expected_edges & actual_edges) / len(expected_edges)
                      if expected_edges else 1.0)
        violations = []
        if coverage < 1.0:
            violations.append(f"FSM 节点序列未完整对齐: {matched}/{len(expected_nodes)}")
        if transition < 1.0:
            violations.append(f"FSM 转移覆盖不足: {transition:.1%}")
        return coverage, transition, violations

    def _observable_alignment(self, graph: FlowGraph, path: list[Edge],
                              trajectory: Trajectory) -> tuple[float, float, list[str]]:
        """无节点 trace 时验证可观测状态契约，不声称精确节点覆盖。"""
        expected_labels = [graph.nodes[edge.dst].label for edge in path
                           if edge.dst in graph.nodes]
        observations: list[str] = []
        for turn in trajectory.turns:
            state = turn.state
            observations.extend(str(state.get(key, "")) for key in (
                "last_tag", "last_verdict", "last_label", "last_agent_status"
            ))
            observations.append(turn.agent)
        text = "\n".join(observations)
        observable_keywords = [self._keyword(label) for label in expected_labels]
        observable_keywords = [key for key in observable_keywords if key]
        matched = sum(1 for key in observable_keywords if key in text)
        semantic_coverage = (matched / len(observable_keywords)
                             if observable_keywords else 1.0)
        # 终态、计数器和明确分类是可验证观察点；其余节点仅给部分信用。
        coverage = min(1.0, 0.35 + 0.65 * semantic_coverage)
        invalid_seen = any(
            "描述无效" in turn.agent
            or "invalid_desc" in str(turn.state)
            for turn in trajectory.turns
        )
        transition = 0.0 if invalid_seen else coverage
        violations = []
        if invalid_seen:
            violations.append("检测到已被规约删除的“描述无效”状态或话术")
        if coverage < 0.7:
            violations.append("可观测 FSM 义务覆盖不足；建议 Agent 输出 facts.fsm_trace")
        return coverage, transition, violations

    @staticmethod
    def _keyword(label: str) -> str:
        candidates = (
            "题外", "类型不一致", "确认提交", "暂缓", "不认可", "放弃",
            "工单", "分类", "申诉", "成功", "失败", "清零",
        )
        return next((candidate for candidate in candidates if candidate in label), "")

    def _counter_obligations(self, path: list[Edge], trajectory: Trajectory
                             ) -> tuple[float, list[str], int, int]:
        obligations = [edge.counter for edge in path if edge.counter]
        if not obligations:
            return 1.0, [], 0, 0
        states = [turn.state for turn in trajectory.turns]
        satisfied = 0
        violations: list[str] = []
        for effect in obligations:
            if ":=0" in effect:
                name = effect.split(":=", 1)[0]
                field = _COUNTER_FIELDS.get(name, f"{name}_count")
                values = [state.get(field) for state in states]
                seen_positive = False
                ok = False
                for value in values:
                    if isinstance(value, int) and value > 0:
                        seen_positive = True
                    if seen_positive and value == 0:
                        ok = True
                        break
            elif "+1" in effect:
                name = effect.split("+", 1)[0]
                field = _COUNTER_FIELDS.get(name, f"{name}_count")
                values = [state.get(field) for state in states]
                ok = any(isinstance(value, int) and value >= 1 for value in values)
            else:
                ok = True
            if ok:
                satisfied += 1
            else:
                violations.append(f"计数器效果未观测到: {effect}")
        return satisfied / len(obligations), violations, len(obligations), satisfied

    @staticmethod
    def _terminal_obligation(graph: FlowGraph, path: list[Edge],
                             trajectory: Trajectory) -> tuple[float, list[str]]:
        if not path:
            return 1.0, []
        target = graph.nodes.get(path[-1].dst)
        if not target or target.type != "terminal":
            return 0.0, ["目标路径未以 terminal 节点结束"]
        if not trajectory.turns:
            return 0.0, ["轨迹为空，无法验证终态"]
        final_state = trajectory.turns[-1].state
        if not final_state.get("terminal"):
            return 0.0, [f"未到达目标终态: {target.label}"]
        observed = f"{trajectory.terminal_reason} {trajectory.turns[-1].agent}"
        keywords = [word for word in ("题外", "类型不一致", "暂缓", "不认可",
                                      "放弃", "成功", "失败", "流程结束")
                    if word in target.label]
        if not keywords or any(word in observed for word in keywords):
            return 1.0, []
        return 0.0, [f"终态不匹配: 目标={target.label}, 实际={trajectory.terminal_reason}"]
