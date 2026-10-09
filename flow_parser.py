"""FlowParser:多模态流程图解析器。

输入:whiteboard.pdf / whiteboard.png / mermaid / 纯文本流程描述
输出:FlowGraph(形式化行为规约 IR)

解析策略(按优先级降级):
1. PDF 文字层提取(PyMuPDF / pdftotext)→ LLM 结构化
2. 图片 → 多模态 LLM 视觉理解 → 结构化
3. 纯文本 / markdown / mermaid → LLM 结构化

可选:结合产品需求文档(PRD)对解析结果做交叉校验与补全。
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from flow_graph import FlowGraph, Node, Edge
from llm_client import LLMClient

STRUCTURE_PROMPT = """你是流程图形式化专家。请将下面的流程图内容转换为严格的 JSON 有向图。

要求:
1. 识别所有节点,type 取值:start(入口)/state(阶段)/decision(判断菱形)/action(动作)/terminal(终态/退出)
2. 识别所有转移边,decision 节点的每个分支必须有 guard(条件描述)
3. 识别所有计数器语义(如"题外计数<3"),写入 edge.counter,并在顶层 counters 汇总阈值
4. 为节点标注所属阶段 step(如 "Step1 开场识别")
5. 输出 JSON,结构:
{"name": "...", "nodes": [{"id","label","type","step"}], "edges": [{"src","dst","guard","counter","label"}], "counters": {"计数器名": 阈值}}

流程图内容:
<<<FLOW>>>
"""

CROSS_CHECK_PROMPT = """你是需求分析专家。下面是从流程图解析出的形式化规约,以及产品需求文档节选。
请检查规约是否有遗漏/错误(节点缺失、guard 不完整、计数器阈值不符等),
输出修正后的完整 JSON(结构同输入),不要输出解释。

## 当前规约 JSON
<<<GRAPH>>>

## 产品需求文档
<<<PRD>>>
"""


class FlowParser:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------
    def parse(self, flow_path: str, prd_path: str = "") -> FlowGraph:
        path = Path(flow_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw = self._extract_pdf_text(path)
            if len(raw.strip()) < 40:  # 文字层过少 → 尝试视觉
                raw = self._extract_via_vision(path)
        elif suffix in (".png", ".jpg", ".jpeg"):
            raw = self._extract_via_vision(path)
        else:
            raw = path.read_text(encoding="utf-8")

        graph = self._structure(raw)
        if prd_path:
            graph = self._cross_check(graph, Path(prd_path).read_text(encoding="utf-8"))
        return graph

    # ------------------------------------------------------------------
    def _extract_pdf_text(self, path: Path) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            out = subprocess.run(["pdftotext", str(path), "-"],
                                 capture_output=True, text=True, check=True)
            return out.stdout

    def _extract_via_vision(self, path: Path) -> str:
        # PDF → 渲染为 PNG 再走视觉模型
        img_path = str(path)
        if path.suffix.lower() == ".pdf":
            import fitz
            doc = fitz.open(str(path))
            pix = doc[0].get_pixmap(dpi=200)
            img_path = str(path.with_suffix(".render.png"))
            pix.save(img_path)
        return self.llm.vision_chat(
            "请完整描述这张流程图:列出所有节点(含类型:开始/判断/动作/终态)、"
            "所有带条件的转移边、所有计数器及阈值。按拓扑顺序输出。",
            img_path,
        )

    # ------------------------------------------------------------------
    def _structure(self, raw: str) -> FlowGraph:
        data = self.llm.chat_json([
            {"role": "user", "content": STRUCTURE_PROMPT.replace("<<<FLOW>>>", raw)},
        ])
        return self._build(data)

    def _cross_check(self, graph: FlowGraph, prd: str) -> FlowGraph:
        prompt = (CROSS_CHECK_PROMPT
                  .replace("<<<GRAPH>>>", graph.to_json())
                  .replace("<<<PRD>>>", prd[:16000]))
        data = self.llm.chat_json([{"role": "user", "content": prompt}])
        return self._build(data)

    @staticmethod
    def _build(data: dict) -> FlowGraph:
        g = FlowGraph(name=data.get("name", "flow"),
                      counters=data.get("counters", {}))
        for n in data.get("nodes", []):
            g.add_node(Node(id=str(n["id"]), label=n.get("label", ""),
                            type=n.get("type", "state"), step=n.get("step", "")))
        for e in data.get("edges", []):
            g.add_edge(Edge(src=str(e["src"]), dst=str(e["dst"]),
                            guard=e.get("guard", ""), counter=e.get("counter", ""),
                            label=e.get("label", "")))
        return g
