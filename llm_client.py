"""统一 LLM 客户端(OpenAI 兼容,默认 DeepSeek)。

支持:
- chat():文本对话,支持 JSON 模式
- vision_chat():多模态(流程图图片理解),需要支持视觉的模型
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    vision_model: str = os.getenv("LLM_VISION_MODEL", "")  # 例如 qwen-vl-max / gpt-4o
    temperature: float = 0.2
    max_retries: int = 3
    timeout: int = int(os.getenv("LLM_TIMEOUT", "300"))
    max_calls: int = int(os.getenv("LLM_MAX_CALLS", "0"))  # 0=不限
    trace_path: str = os.getenv("LLM_TRACE", "")           # 非空则记录每次调用
    extra_headers: dict = field(default_factory=dict)


class LLMBudgetExceeded(RuntimeError):
    """LLM 调用次数超过预算上限。"""


class LLMClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self.calls = 0
        self.total_chars = 0

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], *, json_mode: bool = False,
             temperature: Optional[float] = None, model: Optional[str] = None) -> str:
        payload: dict[str, Any] = {
            "model": model or self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            try:
                return self._post(payload)
            except RuntimeError:
                # 部分网关不支持 response_format,降级为提示词约束
                payload.pop("response_format", None)
                payload["messages"] = messages + [
                    {"role": "user", "content": "只输出 JSON,不要任何其它内容"}]
                return self._post(payload)
        return self._post(payload)

    def chat_json(self, messages: list[dict], **kw) -> dict:
        text = self.chat(messages, json_mode=True, **kw)
        return _loads_loose(text)

    # ------------------------------------------------------------------
    def vision_chat(self, prompt: str, image_path: str) -> str:
        if not self.cfg.vision_model:
            raise RuntimeError("未配置视觉模型(LLM_VISION_MODEL),请改用文本层解析或配置多模态模型")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        suffix = image_path.rsplit(".", 1)[-1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
        payload = {
            "model": self.cfg.vision_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        }
        return self._post(payload)

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> str:
        if self.cfg.max_calls and self.calls >= self.cfg.max_calls:
            raise LLMBudgetExceeded(
                f"LLM 调用已达预算上限 {self.cfg.max_calls} 次")
        self.calls += 1
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            **self.cfg.extra_headers,
        }
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                self.total_chars += len(content)
                self._trace(payload, content)
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 请求失败: {last_err}")

    def _trace(self, payload: dict, content: str) -> None:
        if not self.cfg.trace_path:
            return
        try:
            with open(self.cfg.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"call": self.calls, "model": payload.get("model"),
                                    "messages": payload.get("messages"),
                                    "response": content},
                                   ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass


def _loads_loose(text: str) -> dict:
    """宽松 JSON 解析:剥离 markdown 代码块围栏。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)
