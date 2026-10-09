"""程序自迭代 MDP 与 GRAVE 全局动作价值。

State: Git 内容树 + 规约版本 + 失败通路签名
Action: 语义补丁（具体 action_id，抽象 grave_key）
Transition: 保护检查 -> 应用补丁 -> 编译 gate -> 内容树去重 -> 回归门禁
Reward: 统计保守的多目标规约符合度
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from patcher import Patch, Patcher, RepoManager


@dataclass(frozen=True)
class SearchState:
    commit: str
    tree_hash: str
    spec_hash: str
    failure_signature: str

    @property
    def key(self) -> str:
        return f"{self.tree_hash}:{self.spec_hash}:{self.failure_signature}"


@dataclass
class TransitionResult:
    state: SearchState
    action: Patch
    next_commit: Optional[str]
    next_tree_hash: str = ""
    accepted: bool = False
    reason: str = ""


class ProgramTransitionModel:
    """显式、可审计的确定性程序转移函数 T(s,a)。"""

    def __init__(self, repo: RepoManager, patcher: Patcher):
        self.repo = repo
        self.patcher = patcher

    def transition(self, state: SearchState, action: Patch,
                   visited_trees: set[str]) -> TransitionResult:
        commit = self.patcher.apply(action, state.commit)
        if commit is None:
            return TransitionResult(state, action, None, reason="protection_or_build_gate")
        tree_hash = self.repo.tree_hash(commit)
        if tree_hash in visited_trees:
            self.repo.checkout(state.commit)
            return TransitionResult(state, action, commit, tree_hash,
                                    reason="duplicate_program_state")
        return TransitionResult(state, action, commit, tree_hash,
                                accepted=True, reason="candidate")


@dataclass
class ActionValue:
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


@dataclass
class GRAVEStatistics:
    """Global RAVE/GRAVE-style 共享统计。

    同一 ``grave_key``（如 counter:internal）在不同代码状态出现时共享
    AMAF 价值。局部 Q 与全局 AMAF 按访问次数自适应混合。
    """

    bias: float = 20.0
    values: dict[str, ActionValue] = field(default_factory=dict)

    def update(self, action_key: str, value: float) -> None:
        stat = self.values.setdefault(action_key, ActionValue())
        stat.visits += 1
        stat.total_value += value

    def estimate(self, action_key: str) -> ActionValue:
        return self.values.get(action_key, ActionValue())

    def blended_value(self, local_mean: float, local_visits: int,
                      action_key: str) -> tuple[float, float, float]:
        global_stat = self.estimate(action_key)
        if global_stat.visits == 0:
            return local_mean, 0.0, local_mean
        beta = global_stat.visits / (
            global_stat.visits + local_visits + self.bias
        )
        blended = (1.0 - beta) * local_mean + beta * global_stat.mean
        return blended, beta, global_stat.mean

    def selection_score(self, local_mean: float, local_visits: int,
                        parent_visits: int, action_key: str,
                        exploration: float) -> float:
        if local_visits == 0:
            return float("inf")
        blended, _, _ = self.blended_value(local_mean, local_visits, action_key)
        bonus = exploration * math.sqrt(
            math.log(max(1, parent_visits)) / local_visits
        )
        return blended + bonus
