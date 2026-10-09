"""Simulator:LLM 司机模拟器 + 轨迹采集。

给定 FlowGraph 中的一条目标通路(target path),
用 LLM 扮演司机与被测 Agent 多轮对话,诱导 Agent 沿该通路行进,
记录完整轨迹供 Evaluator 打分。

这是 MCTS 的 "Simulation(rollout)" 阶段:
- 每条通路一次 rollout = 一次完整多轮对话
- 轨迹 = [(driver_utterance, agent_reply, agent_state), ...]
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from flow_graph import FlowGraph, Edge
from llm_client import LLMClient

DRIVER_PROMPT = """你是一名网约车司机模拟器,正在测试一个"绕路申诉"客服 Agent。
你的任务是通过发言,诱导 Agent 走完下面这条目标流程通路:

## 目标通路
<<<PATH>>>

## 对话历史
<<<HISTORY>>>

请输出司机的下一句话。规则:
- 若通路要求"题外话",就说与申诉无关的话(如聊天气)
- 若要求"描述无效",就说空泛无信息量的话(如"就是那样")
- 若要求某申诉分类,就给出对应的具体绕路理由
- 若要求"暂缓/不认可/放弃/确认提交",就明确表达对应意图
- 只输出司机的一句话,不要任何解释
"""


@dataclass
class Turn:
    driver: str
    agent: str
    state: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    path_signature: str
    target_path_desc: str
    target_path: list[Edge] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    terminal_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "path_signature": self.path_signature,
            "target_path_desc": self.target_path_desc,
            "target_path": [{"src": edge.src, "dst": edge.dst,
                             "guard": edge.guard, "counter": edge.counter,
                             "label": edge.label} for edge in self.target_path],
            "turns": [{"driver": t.driver, "agent": t.agent, "state": t.state} for t in self.turns],
            "terminal_reason": self.terminal_reason,
            "error": self.error,
        }


class AgentUnderTest:
    """通过 UI-test 的 mock server HTTP API 驱动被测 Agent。"""

    def __init__(self, base_url: str = "http://127.0.0.1:18081"):
        self.base_url = base_url.rstrip("/")
        self.session_id = ""

    def reset(self, mode: str = "first") -> dict:
        self.session_id = f"evolve-{int(time.time() * 1000)}"
        r = requests.post(f"{self.base_url}/api/reset",
                          json={"session_id": self.session_id, "mode": mode}, timeout=30)
        r.raise_for_status()
        return r.json()

    def chat(self, query: str) -> dict:
        r = requests.post(f"{self.base_url}/api/chat",
                          json={"session_id": self.session_id, "query": query}, timeout=120)
        r.raise_for_status()
        return r.json()


class DriverSimulator:
    def __init__(self, llm: Optional[LLMClient] = None,
                 agent: Optional[AgentUnderTest] = None,
                 max_turns: int = 14):
        self.llm = llm or LLMClient()
        self.agent = agent or AgentUnderTest()
        self.max_turns = max_turns

    def rollout(self, graph: FlowGraph, path: list[Edge], mode: str = "first") -> Trajectory:
        path_desc = self._describe_path(graph, path)
        traj = Trajectory(path_signature=graph.path_signature(path),
                          target_path_desc=path_desc,
                          target_path=list(path))
        try:
            state = self.agent.reset(mode=mode)
        except Exception as e:  # noqa: BLE001
            traj.error = f"agent reset 失败: {e}"
            return traj

        history: list[str] = [f"Agent: {m.get('text','')}" for m in state.get("messages", [])]

        for _ in range(self.max_turns):
            driver_say = self._next_driver_utterance(path_desc, history)
            try:
                result = self.agent.chat(driver_say)
            except Exception as e:  # noqa: BLE001
                traj.error = f"chat 失败: {e}"
                break
            st = result.get("state", {})
            facts = st.get("facts", {})
            msgs = st.get("messages", [])
            agent_reply = msgs[-1].get("text", "") if msgs else ""
            traj.turns.append(Turn(driver=driver_say, agent=agent_reply, state=facts))
            history.append(f"司机: {driver_say}")
            history.append(f"Agent: {agent_reply}")
            if facts.get("terminal"):
                traj.terminal_reason = facts.get("terminal_reason", "") or (
                    "建单成功" if facts.get("fake_work_id") else "终态")
                break
        return traj

    # ------------------------------------------------------------------
    def _describe_path(self, graph: FlowGraph, path: list[Edge]) -> str:
        lines = []
        for e in path:
            src = graph.nodes.get(e.src)
            dst = graph.nodes.get(e.dst)
            guard = f"(条件: {e.guard})" if e.guard else ""
            lines.append(f"{src.label if src else e.src} → {dst.label if dst else e.dst} {guard}")
        return "\n".join(lines)

    def _next_driver_utterance(self, path_desc: str, history: list[str]) -> str:
        prompt = (DRIVER_PROMPT
                  .replace("<<<PATH>>>", path_desc)
                  .replace("<<<HISTORY>>>", "\n".join(history[-20:]) or "(对话尚未开始)"))
        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7).strip()
