"""OpenAI 兼容网关的扩展字段（如阿里云百炼 DashScope）。

与参考仓库 ``general-agentic-memory-main 2`` 中
``eval/halumem/llms._chat_completions_extra_body``、
``gam/generator/openai_generator`` 内对 ``extra_body`` 的处理对齐：
``OPENAI_CHAT_EXTRA_BODY`` JSON 合并 + 百炼 Qwen3 非流式须 ``enable_thinking=false``。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def merge_dashscope_qwen3_nonstream_extra_body(
    model_name: str,
    base_url: Optional[str],
    params: Dict[str, Any],
) -> None:
    """
    在 ``params`` 上原地合并 ``extra_body``（保留调用方已有键）。

    - 环境变量 ``OPENAI_CHAT_EXTRA_BODY``：合法 JSON 对象会并入（与参考仓库一致）。
    - 当 ``base_url`` 含 ``dashscope`` 且模型名含 ``qwen3`` 时，``setdefault("enable_thinking", False)``
      （百炼非流式要求，同 ``tests/test_ai_api.ipynb`` 中 ``extra_body``）。
    """
    merged: Dict[str, Any] = {}
    prev = params.get("extra_body")
    if isinstance(prev, dict):
        merged.update(prev)

    raw = (os.getenv("OPENAI_CHAT_EXTRA_BODY") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                merged.update(parsed)
        except json.JSONDecodeError:
            logger.warning("OPENAI_CHAT_EXTRA_BODY 不是合法 JSON，已忽略: %s", raw[:200])

    bu = (base_url or "").lower()
    mn = (model_name or "").lower()
    if "dashscope" in bu and "qwen3" in mn:
        merged.setdefault("enable_thinking", False)

    if merged:
        params["extra_body"] = merged
