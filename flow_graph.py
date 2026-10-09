"""FlowGraph:流程图中间表示(IR)。

将白板流程图(whiteboard.pdf 等)结构化为带守卫条件的有向图,
作为 Agent 行为规约(behavioral specification)的形式化载体。

核心概念:
- Node:流程节点(state / decision / action / terminal)
- Edge:带守卫条件(guard)与计数器语义(counter)的转移
- Path:从入口到终态的一条通路 —— MCTS 的评估与覆盖单位
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

NODE_TYPES = ("start", "state", "decision", "action", "terminal")


@dataclass
class Node:
    id: str
    label: str
    type: str = "state"          # start / state / decision / action / terminal
    step: str = ""               # 所属阶段,如 "Step1 开场识别"

    def __post_init__(self):
        assert self.type in NODE_TYPES, f"非法节点类型: {self.type}"


@dataclass
class Edge:
    src: str
    dst: str
    guard: str = ""              # 转移条件的自然语言/伪代码描述
    counter: str = ""            # 关联计数器,如 "off_topic<3"
    label: str = ""


@dataclass
class FlowGraph:
    name: str = "flow"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)   # 计数器 -> 阈值,如 {"off_topic": 3}
    meta: dict = field(default_factory=dict)

    # ---------------- 构建 ----------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    # ---------------- 查询 ----------------
    @property
    def start_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == "start"]

    @property
    def terminal_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == "terminal"]

    def out_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src == node_id]

    # ---------------- 路径枚举 ----------------
    def enumerate_paths(self, max_depth: int = 40, max_loop_visits: int = 3) -> Iterator[list[Edge]]:
        """DFS 枚举 start→terminal 的所有通路。

        允许环(计数器重试语义)最多经过 max_loop_visits 次,
        以覆盖 "题外3次退出" 这类需要多次绕环才能触发的路径。
        """
        for start in self.start_nodes:
            stack: list[tuple[str, list[Edge], dict[str, int]]] = [(start.id, [], {})]
            while stack:
                cur, path, visits = stack.pop()
                if len(path) > max_depth:
                    continue
                node = self.nodes.get(cur)
                if node and node.type == "terminal":
                    yield path
                    continue
                for edge in self.out_edges(cur):
                    key = f"{edge.src}->{edge.dst}"
                    v = visits.get(key, 0)
                    if v >= max_loop_visits:
                        continue
                    nv = dict(visits)
                    nv[key] = v + 1
                    stack.append((edge.dst, path + [edge], nv))

    def path_signature(self, path: list[Edge]) -> str:
        raw = "|".join(f"{e.src}->{e.dst}[{e.guard}]" for e in path)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _edge_key(edge: Edge) -> str:
        return f"{edge.src}->{edge.dst}[{edge.guard}][{edge.counter}]"

    def path_features(self, path: list[Edge]) -> set[str]:
        """提取用于覆盖约简的控制流特征。

        除边和终态外，还记录同一边第 1/2/3 次出现，确保计数器循环的
        边界次数不会被普通 edge coverage 掩盖。
        """
        features: set[str] = set()
        counts: Counter[str] = Counter()
        for edge in path:
            key = self._edge_key(edge)
            counts[key] += 1
            features.add(f"edge:{key}")
            features.add(f"visit:{key}#{counts[key]}")
            if edge.counter:
                features.add(f"counter:{edge.counter}")
        if path:
            features.add(f"terminal:{path[-1].dst}")
        return features

    def coverage_suite(self, max_depth: int = 40,
                       max_loop_visits: int = 3,
                       max_paths: int = 0) -> list[list[Edge]]:
        """贪心集合覆盖生成代表性路径集。

        返回的套件覆盖所有可达边、终态、计数器效果和循环次数边界。
        `max_paths=0` 表示不限制；非零时保留覆盖收益最高的前 N 条。
        """
        candidates = list(self.enumerate_paths(max_depth, max_loop_visits))
        if not candidates:
            return []
        feature_sets = [self.path_features(path) for path in candidates]
        uncovered = set().union(*feature_sets)
        selected: list[list[Edge]] = []
        remaining = set(range(len(candidates)))
        while uncovered and remaining and (not max_paths or len(selected) < max_paths):
            best_index = max(
                remaining,
                key=lambda index: (len(feature_sets[index] & uncovered),
                                   -len(candidates[index])),
            )
            gain = feature_sets[best_index] & uncovered
            if not gain:
                break
            selected.append(candidates[best_index])
            uncovered -= gain
            remaining.remove(best_index)
        return selected

    def coverage_ratio(self, paths: list[list[Edge]],
                       max_depth: int = 40,
                       max_loop_visits: int = 3) -> float:
        all_paths = list(self.enumerate_paths(max_depth, max_loop_visits))
        universe = set().union(*(self.path_features(path) for path in all_paths))
        covered = set().union(*(self.path_features(path) for path in paths)) if paths else set()
        return len(covered & universe) / len(universe) if universe else 1.0

    # ---------------- 序列化 ----------------
    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "counters": self.counters,
            "meta": self.meta,
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "FlowGraph":
        data = json.loads(text)
        g = cls(name=data.get("name", "flow"),
                counters=data.get("counters", {}),
                meta=data.get("meta", {}))
        for n in data.get("nodes", []):
            g.add_node(Node(**n))
        for e in data.get("edges", []):
            g.add_edge(Edge(**e))
        return g

    # ---------------- 规约对比 ----------------
    def describe(self) -> str:
        """生成人类/LLM 可读的规约文本,用于补丁提示词与一致性检查。"""
        lines = [f"# 流程规约: {self.name}", ""]
        by_step: dict[str, list[Node]] = {}
        for n in self.nodes.values():
            by_step.setdefault(n.step or "未分组", []).append(n)
        for step, nodes in by_step.items():
            lines.append(f"## {step}")
            for n in nodes:
                lines.append(f"- [{n.type}] {n.id}: {n.label}")
                for e in self.out_edges(n.id):
                    guard = f" 当({e.guard})" if e.guard else ""
                    counter = f" [计数:{e.counter}]" if e.counter else ""
                    lines.append(f"    -> {e.dst}{guard}{counter}")
        if self.counters:
            lines.append("\n## 计数器阈值")
            for k, v in self.counters.items():
                lines.append(f"- {k}: 达到 {v} 次触发退出")
        return "\n".join(lines)
