"""Patcher:基于 LLM 的代码补丁生成 / 应用 / 回滚。

MCTS 的 "Action":一次针对失败通路的最小语义补丁(prompt 修改、
状态机分支修复、计数器逻辑修正等)。

工程保障:
- 每个 MCTS 节点对应一个 git commit(可精确回溯任意树节点)
- 补丁以 whole-file-rewrite 形式应用(比 unified diff 更稳),
  每次仅允许改动 ≤ N 个文件,保持动作粒度可控
- 应用后跑编译/静态检查作为快速失败门(gate)
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm_client import LLMClient

PATCH_PROMPT = """你是资深 Agent 工程师。被测 Agent 在以下流程通路上未能符合规约,请生成最小修复补丁。

## 流程规约(目标行为)
<<<SPEC>>>

## 失败通路与违规点
<<<FAILURES>>>

## 相关源码文件
<<<SOURCES>>>

要求:
1. 分析根因(状态机分支缺失?提示词不精确?计数器逻辑错?)
2. 生成最小修改:优先使用 edits(精确查找替换),find 必须是源文件中逐字符存在的唯一片段
3. 输出 JSON:
{
  "root_cause": "根因分析",
    "action_type": "guard|counter|transition|classification|state|other",
  "strategy": "修复策略一句话",
  "edits": [{"path": "相对路径", "find": "原文片段", "replace": "新片段"}]
}
每个 edit 的 find 不超过 40 行;最多 6 个 edits。禁止输出完整文件。
硬约束(违反则补丁作废):
- 禁止修改对外固定接口的请求/响应结构(/v1/agent/* 的调用契约)
- 禁止修改已编排的固定回复话术文案字符串,只允许修改流程控制/判定/计数逻辑
"""


@dataclass
class Patch:
    root_cause: str
    strategy: str
    action_type: str = "other"
    files: dict[str, str] = field(default_factory=dict)   # path -> new content(全文件)
    edits: list[dict] = field(default_factory=list)       # [{path, find, replace}]
    commit: str = ""

    @property
    def action_id(self) -> str:
        import hashlib
        raw = self.strategy + "".join(sorted(self.files)) + json.dumps(self.edits, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    @property
    def grave_key(self) -> str:
        """跨程序状态共享价值的抽象动作键。"""
        allowed = {"guard", "counter", "transition", "classification", "state", "other"}
        action_type = self.action_type if self.action_type in allowed else "other"
        touched = sorted({
            path.split("/", 1)[0]
            for path in list(self.files) + [edit.get("path", "") for edit in self.edits]
            if path
        })
        return f"{action_type}:{','.join(touched) or 'unknown'}"


class RepoManager:
    """git 快照管理:MCTS 树节点 ←→ commit 双射。"""

    def __init__(self, repo_dir: str):
        self.repo = Path(repo_dir)
        if not (self.repo / ".git").exists():
            self._git("init")
            self._git("add", "-A")
            self._git("commit", "-m", "evolver: baseline", "--allow-empty")

    def _git(self, *args: str) -> str:
        out = subprocess.run(["git", *args], cwd=self.repo,
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} 失败: {out.stderr.strip()}")
        return out.stdout.strip()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def tree_hash(self, commit: str = "HEAD") -> str:
        """返回提交对应的内容树哈希；相同源码状态具有相同值。"""
        return self._git("rev-parse", f"{commit}^{{tree}}")

    def checkout(self, commit: str) -> None:
        self._git("checkout", "--force", commit)

    def commit_all(self, message: str) -> str:
        self._git("add", "-A")
        self._git("commit", "-m", message, "--allow-empty")
        return self.head()

    def read(self, rel_path: str) -> str:
        return (self.repo / rel_path).read_text(encoding="utf-8")

    def write(self, rel_path: str, content: str) -> None:
        p = self.repo / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


class Patcher:
    def __init__(self, repo: RepoManager, llm: Optional[LLMClient] = None,
                 source_globs: Optional[list[str]] = None,
                 protected_globs: Optional[list[str]] = None,
                 protected_literals: Optional[list[str]] = None,
                 build_cmd: Optional[list[str]] = None):
        self.repo = repo
        self.llm = llm or LLMClient()
        self.source_globs = source_globs or ["internal/**/*.go", "conf/**/*",
                                             "UI-test/mock_ui_server.py"]
        self.protected_globs = protected_globs or []
        # 行级保护:这些话术字面量片段不得被补丁新增/删除
        self.protected_literals = protected_literals or []
        self.build_cmd = build_cmd  # 例如 ["go", "build", "./..."]

    def _is_protected(self, rel_path: str) -> bool:
        from fnmatch import fnmatch
        return any(fnmatch(rel_path, g) for g in self.protected_globs)

    def _violates_literal(self, patch: "Patch") -> bool:
        """行级保护:若补丁的 find 片段包含受保护话术且 replace 不再包含它,
        则视为篡改固定话术,拒绝。"""
        if not self.protected_literals:
            return False
        for edit in patch.edits:
            find, replace = edit.get("find", ""), edit.get("replace", "")
            for lit in self.protected_literals:
                if lit in find and lit not in replace:
                    return True
        for content in patch.files.values():
            for lit in self.protected_literals:
                # 全文件重写且丢失了受保护话术
                if lit not in content:
                    return True
        return False

    # ------------------------------------------------------------------
    def propose(self, spec_text: str, failures_text: str,
                n_candidates: int = 3) -> list[Patch]:
        """针对失败信号生成 n 个候选补丁(MCTS Expansion 的动作集)。"""
        sources = self._collect_sources(failures_text)
        prompt = (PATCH_PROMPT
                  .replace("<<<SPEC>>>", spec_text[:8000])
                  .replace("<<<FAILURES>>>", failures_text[:6000])
                  .replace("<<<SOURCES>>>", sources[:24000]))
        patches: list[Patch] = []
        for i in range(n_candidates):
            try:
                data = self.llm.chat_json(
                    [{"role": "user", "content": prompt}],
                    temperature=0.3 + 0.3 * i,  # 多样性递增
                )
                files = {f["path"]: f["content"] for f in data.get("files", [])[:3]}
                edits = data.get("edits", [])[:6]
                if files or edits:
                    patches.append(Patch(root_cause=data.get("root_cause", ""),
                                         strategy=data.get("strategy", ""),
                                         action_type=data.get("action_type", "other"),
                                         files=files, edits=edits))
            except Exception:  # noqa: BLE001
                continue
        # 去重(同一策略哈希)
        seen, unique = set(), []
        for p in patches:
            if p.action_id not in seen:
                seen.add(p.action_id)
                unique.append(p)
        return unique

    # ------------------------------------------------------------------
    def apply(self, patch: Patch, base_commit: str) -> Optional[str]:
        """在 base_commit 上应用补丁,通过 gate 后 commit,返回新 commit。失败返回 None。"""
        # 保护拦截:触碰固定接口/话术文件的补丁直接拒绝
        touched = list(patch.files) + [e.get("path", "") for e in patch.edits]
        if any(self._is_protected(p) for p in touched):
            return None
        # 行级拦截:保护固定话术字面量不被篡改
        if self._violates_literal(patch):
            return None
        self.repo.checkout(base_commit)
        # 防截断护栏:LLM 全文件重写若使已有文件缩水 >25%,判定为截断,拒绝补丁
        for path, content in patch.files.items():
            target = self.repo.repo / path
            if target.exists():
                old_lines = len(target.read_text(encoding="utf-8",
                                                 errors="ignore").splitlines())
                new_lines = len(content.splitlines())
                if old_lines > 50 and new_lines < old_lines * 0.75:
                    return None
        for path, content in patch.files.items():
            self.repo.write(path, content)
        for edit in patch.edits:
            try:
                src = self.repo.read(edit["path"])
            except FileNotFoundError:
                self.repo.checkout(base_commit)
                return None
            find = edit.get("find", "")
            if not find or src.count(find) != 1:
                self.repo.checkout(base_commit)
                return None  # 非唯一/不存在 → 拒绝,保证确定性
            self.repo.write(edit["path"], src.replace(find, edit.get("replace", ""), 1))
        if not self._gate():
            self.repo.checkout(base_commit)  # 回滚
            return None
        patch.commit = self.repo.commit_all(
            f"evolver[{patch.action_id}]: {patch.strategy[:60]}")
        return patch.commit

    def _gate(self) -> bool:
        if not self.build_cmd:
            return True
        out = subprocess.run(self.build_cmd, cwd=self.repo.repo,
                             capture_output=True, text=True)
        return out.returncode == 0

    # ------------------------------------------------------------------
    def _collect_sources(self, failures_text: str) -> str:
        """启发式收集与失败相关的源码(可替换为语义检索)。"""
        chunks = []
        for pattern in self.source_globs:
            for f in sorted(self.repo.repo.glob(pattern)):
                if f.is_file() and f.stat().st_size < 60_000:
                    chunks.append(f"### {f.relative_to(self.repo.repo)}\n"
                                  f"```\n{f.read_text(encoding='utf-8', errors='ignore')}\n```")
        return "\n\n".join(chunks)
