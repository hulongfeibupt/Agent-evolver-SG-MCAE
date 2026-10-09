"""FlowGraph 静态规约验证器。

LLM 解析流程图后不应直接进入搜索。该模块在演化前检查图结构、
控制流完备性、计数器约束和本项目需求增量，避免在错误规约上优化。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from flow_graph import FlowGraph


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    location: str = ""


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def format(self) -> str:
        if not self.issues:
            return "规约验证通过：未发现结构或语义问题"
        lines = []
        for issue in self.issues:
            where = f" [{issue.location}]" if issue.location else ""
            lines.append(f"{issue.severity.upper()} {issue.code}{where}: {issue.message}")
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("流程规约验证失败：\n" + self.format())


class FlowSpecValidator:
    def validate(self, graph: FlowGraph) -> ValidationReport:
        report = ValidationReport()
        self._check_entry_and_terminal(graph, report)
        self._check_edges(graph, report)
        self._check_reachability(graph, report)
        self._check_decisions(graph, report)
        self._check_counters(graph, report)
        self._check_requirement_overrides(graph, report)
        return report

    @staticmethod
    def _add(report: ValidationReport, severity: str, code: str,
             message: str, location: str = "") -> None:
        report.issues.append(ValidationIssue(severity, code, message, location))

    def _check_entry_and_terminal(self, graph: FlowGraph, report: ValidationReport) -> None:
        if len(graph.start_nodes) != 1:
            self._add(report, "error", "START_CARDINALITY",
                      f"必须恰有一个 start 节点，实际为 {len(graph.start_nodes)}")
        if not graph.terminal_nodes:
            self._add(report, "error", "NO_TERMINAL", "至少需要一个 terminal 节点")

    def _check_edges(self, graph: FlowGraph, report: ValidationReport) -> None:
        seen: set[tuple[str, str, str]] = set()
        for index, edge in enumerate(graph.edges):
            location = f"edge:{index}:{edge.src}->{edge.dst}"
            if edge.src not in graph.nodes:
                self._add(report, "error", "DANGLING_SOURCE",
                          f"源节点 {edge.src!r} 不存在", location)
            if edge.dst not in graph.nodes:
                self._add(report, "error", "DANGLING_TARGET",
                          f"目标节点 {edge.dst!r} 不存在", location)
            key = (edge.src, edge.dst, edge.guard.strip())
            if key in seen:
                self._add(report, "warning", "DUPLICATE_EDGE", "存在重复转移边", location)
            seen.add(key)
        for node in graph.nodes.values():
            if node.type != "terminal" and not graph.out_edges(node.id):
                self._add(report, "error", "DEAD_END",
                          "非终态节点没有出边", node.id)
            if node.type == "terminal" and graph.out_edges(node.id):
                self._add(report, "warning", "TERMINAL_HAS_OUTGOING",
                          "终态节点仍有出边", node.id)

    def _check_reachability(self, graph: FlowGraph, report: ValidationReport) -> None:
        starts = [node.id for node in graph.start_nodes]
        reachable: set[str] = set(starts)
        stack = list(starts)
        while stack:
            current = stack.pop()
            for edge in graph.out_edges(current):
                if edge.dst in graph.nodes and edge.dst not in reachable:
                    reachable.add(edge.dst)
                    stack.append(edge.dst)
        for node_id in sorted(set(graph.nodes) - reachable):
            self._add(report, "error", "UNREACHABLE_NODE",
                      "节点无法从入口到达", node_id)
        reachable_terminals = {node.id for node in graph.terminal_nodes} & reachable
        if graph.terminal_nodes and not reachable_terminals:
            self._add(report, "error", "UNREACHABLE_TERMINAL",
                      "所有终态均无法从入口到达")

    def _check_decisions(self, graph: FlowGraph, report: ValidationReport) -> None:
        for node in graph.nodes.values():
            if node.type != "decision":
                continue
            outgoing = graph.out_edges(node.id)
            if len(outgoing) < 2:
                self._add(report, "error", "DECISION_BRANCH_COUNT",
                          "决策节点至少需要两条出边", node.id)
            normalized_guards = []
            for edge in outgoing:
                guard = edge.guard.strip()
                if not guard:
                    self._add(report, "error", "MISSING_GUARD",
                              f"决策边 {edge.src}->{edge.dst} 缺少 guard", node.id)
                normalized_guards.append(guard)
            guards = [guard for guard in normalized_guards if guard]
            if len(guards) != len(set(guards)):
                self._add(report, "warning", "DUPLICATE_GUARD",
                          "同一决策节点存在重复 guard，可能导致分支歧义", node.id)

    def _check_counters(self, graph: FlowGraph, report: ValidationReport) -> None:
        for name, threshold in graph.counters.items():
            if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
                self._add(report, "error", "INVALID_COUNTER_THRESHOLD",
                          f"计数器 {name!r} 阈值必须是正整数，实际为 {threshold!r}", name)

    def _check_requirement_overrides(self, graph: FlowGraph,
                                     report: ValidationReport) -> None:
        overrides: Iterable[str] = graph.meta.get("overrides", [])
        if not overrides:
            return
        labels = "\n".join(node.label for node in graph.nodes.values())
        if any(token in labels for token in ("描述无效", "无效描述", "引导补充有效")):
            self._add(report, "error", "R1_INVALID_BRANCH_REMAINS",
                      "需求 R1 要求删除的描述无效分支仍存在")
        categories = graph.meta.get("categories", [])
        if len(categories) != 10 or "与申诉无关" not in categories:
            self._add(report, "error", "R2_CATEGORY_10_MISSING",
                      "需求 R2 要求恰有 10 个分类且包含“与申诉无关”")
        reset_nodes = [node for node in graph.nodes.values()
                       if "题外计数清零" in node.label]
        if not reset_nodes:
            self._add(report, "error", "R3_RESET_RULE_MISSING",
                      "需求 R3 的题外计数清零节点不存在")
        elif not any(edge.counter == "off_topic:=0" for edge in graph.edges):
            self._add(report, "error", "R3_RESET_EFFECT_MISSING",
                      "需求 R3 缺少 off_topic:=0 的计数器效果")
