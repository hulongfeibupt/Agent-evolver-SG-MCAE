"""spec_overrides:需求增量(Requirement Delta)模块。

对 FlowParser 解析出的基线规约施加本次迭代的需求变更,
使 MCTS 演化目标 = 流程图基线 ∪ 需求增量:

R1. 删除「描述无效」分支:移除 描述无效判定/引导补充/描述无效计数/退出:描述无效3次
    相关节点与边;原指向这些节点的边改接回「引导司机描述绕路原因」。
R2. 新增第 10 个申诉分类「与申诉无关」(在原 9 类之外)。
R3. 题外计数清零规则:当题外计数未超过 3 时,司机聊出一句有效话术
    (命中前 8 个申诉分类之一)则题外计数清零,并回到 Step2 描述收集。
"""
from __future__ import annotations

from flow_graph import FlowGraph, Node, Edge

# 前 8 个有效分类(第 9 类"以上都不是"、第 10 类"与申诉无关"不触发清零)
VALID_CATEGORIES_1_8 = [
    "乘客要求换路线", "多次修改目的地", "交通管制/修路封路", "恶劣天气绕行",
    "堵车绕行", "导航路线偏差", "乘客中途新增行程", "道路施工临时绕行",
]
CATEGORY_10 = "与申诉无关"
OVERRIDE_VERSION = 2

# R1: 需要删除的「描述无效」分支节点关键词
INVALID_DESC_KEYWORDS = ("描述无效", "无效描述", "引导补充有效")


def apply_overrides(graph: FlowGraph) -> FlowGraph:
    _r1_remove_invalid_desc_branch(graph)
    _r2_add_category_10(graph)
    _r3_off_topic_reset_rule(graph)
    graph.meta["overrides"] = ["R1:删除描述无效分支", "R2:新增分类10-与申诉无关",
                               "R3:题外计数有效话术清零"]
    graph.meta["override_version"] = OVERRIDE_VERSION
    return graph


# ----------------------------------------------------------------------
def _r1_remove_invalid_desc_branch(graph: FlowGraph) -> None:
    doomed = {nid for nid, n in graph.nodes.items()
              if any(k in n.label for k in INVALID_DESC_KEYWORDS)}
    # 找到"引导司机描述绕路原因"节点作为回接点
    guide = next((nid for nid, n in graph.nodes.items()
                  if "引导司机描述" in n.label or "描述收集" in n.label), None)
    off_topic_check = next((nid for nid, n in graph.nodes.items()
                            if "题外计数" in n.label and n.type == "decision"), None)
    kept: list[Edge] = []
    for e in graph.edges:
        # R1 是删除分支，不是把“描述无效”作为新 guard 回接。
        if "描述无效" in e.guard or "无效描述" in e.guard:
            continue
        if e.src in doomed and e.dst in doomed:
            continue
        if e.dst in doomed:
            if guide and e.src not in doomed:
                target = off_topic_check if (
                    off_topic_check and "安抚" in graph.nodes[e.src].label
                ) else guide
                kept.append(Edge(src=e.src, dst=target,
                                 guard=e.guard, counter=e.counter,
                                 label="(R1改接)"))
            continue
        if e.src in doomed:
            continue
        # 修复旧版 override 已产生的错误回接。
        if (off_topic_check and e.src in graph.nodes
                and "安抚" in graph.nodes[e.src].label and e.dst == guide):
            kept.append(Edge(src=e.src, dst=off_topic_check,
                             guard=e.guard, counter=e.counter,
                             label="(R1修复:进入题外计数判断)"))
            continue
        kept.append(e)
    graph.edges = kept
    for nid in doomed:
        graph.nodes.pop(nid, None)
    graph.counters.pop("描述无效计数", None)
    graph.counters.pop("invalid_desc", None)


# ----------------------------------------------------------------------
def _r2_add_category_10(graph: FlowGraph) -> None:
    for n in graph.nodes.values():
        if "9个" in n.label or "9类" in n.label or "匹配9" in n.label:
            n.label = n.label.replace("9个", "10个").replace("9类", "10类")
            suffix = f"(新增第10类:{CATEGORY_10})"
            if suffix not in n.label:
                n.label += suffix
    graph.meta["categories"] = VALID_CATEGORIES_1_8 + ["以上都不是", CATEGORY_10]


# ----------------------------------------------------------------------
def _r3_off_topic_reset_rule(graph: FlowGraph) -> None:
    # 清除旧版不可达的 reset 边，再将清零动作接在“收到下一句有效描述”之后。
    graph.edges = [e for e in graph.edges
                   if e.src != "off_topic_reset" and e.dst != "off_topic_reset"]
    desc_type = next((nid for nid, n in graph.nodes.items()
                      if "描述属于哪类" in n.label and n.type == "decision"), None)
    classify = next((nid for nid, n in graph.nodes.items()
                     if "分类匹配" in n.label and n.type == "action"), None)
    if not (desc_type and classify):
        return
    reset_id = "off_topic_reset"
    graph.add_node(Node(id=reset_id, label="有效话术(命中前8分类)→题外计数清零",
                        type="action", step="Step2 申诉描述收集"))
    # 原“与申诉有关”边仅处理无题外历史或第9类；前8类且已有题外计数走 reset。
    for edge in graph.out_edges(desc_type):
        if edge.dst == classify and edge.guard == "与申诉有关":
            edge.guard = "与申诉有关且(题外计数=0或命中第9类以上都不是)"
    graph.add_edge(Edge(src=desc_type, dst=reset_id,
                        guard="题外计数为1或2且命中前8个申诉分类之一",
                        counter="off_topic:=0", label="(R3新增)"))
    graph.add_edge(Edge(src=reset_id, dst=classify,
                        label="清零后继续Step2分类匹配"))


REQUIREMENTS_TEXT = f"""
## 本次迭代需求增量(必须全部满足)
R1. 【删除分支】完全移除「司机描述无效」分支:不再有"描述无效计数"、
    不再有"引导补充有效描述"话术、不再有"描述无效3次退出"终态。
R2. 【新增分类】申诉分类从 9 类扩展为 10 类,第 10 类名称为「{CATEGORY_10}」。
R3. 【题外清零】当题外计数未超过 3 时,司机聊出一句有效话术
    (命中前 8 个申诉分类之一:{', '.join(VALID_CATEGORIES_1_8)})
    则题外计数立即清零,并回到 Step2 申诉描述收集继续流程。
约束:不得修改 demo 中写死的固定接口(/v1/agent/* 的调用契约)
     与已编排的固定回复话术文案。
"""
