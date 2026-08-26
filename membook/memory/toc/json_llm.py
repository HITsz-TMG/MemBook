# -*- coding: utf-8 -*-
"""JSON parsing + LLM text protocol for MemBook ToC (no OpenAI dependency)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol


class LLMClient(Protocol):
    """MemBook ToC 用 LLM 抽象。``response_json_object=True`` 时由适配层启用 OpenAI 兼容 ``json_object`` 输出（topic-shift 等）。"""

    def complete(self, prompt: str, *, response_json_object: bool = False) -> str: ...


def _strip_redacted_thinking(s: str) -> str:
    """
    与 ``DETECT_TOPIC_SHIFTS`` 等 prompt 中的
    ``<think>...</think>`` + 尾部 JSON 协议对齐。

    顺序：若存在闭合标签，先只保留**最后一次** ``</think>`` 之后的文本
    （漏写开标签、或 thinking 内含 ``{`` 导致难以整块匹配时，仍能拿到 JSON）；
    再删掉仍残留的成对 thinking 块。行为与 ``tests/test_membook/test_llm_client.py`` 覆盖一致。
    """
    raw = (s or "").strip()
    matches = list(re.finditer(r"</think>", raw, flags=re.IGNORECASE))
    if matches:
        after = raw[matches[-1].end() :].strip()
        if after:
            raw = after
    return re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()


def _balanced_brace_substring(s: str, start: int) -> Optional[str]:
    """从 s[start]=='{' 起截取与之配平的 {...}（尊重字符串与转义）。"""
    n = len(s)
    if start < 0 or start >= n or s[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, n):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _try_loads_candidates(candidates: list[str]) -> Optional[Dict[str, Any]]:
    for raw in candidates:
        if not raw:
            continue
        try:
            return _loads_json_with_repair(raw)
        except json.JSONDecodeError:
            continue
    return None


def parse_json_object(text: str) -> Dict[str, Any]:
    s = _strip_redacted_thinking((text or "").strip())

    # Markdown ```json ... ``` / ``` ... ```：取围栏内第一个配平对象（避免 .*? 在串内 } 处截断）
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE):
        inner = (m.group(1) or "").strip()
        j = inner.find("{")
        if j < 0:
            continue
        sub = _balanced_brace_substring(inner, j)
        if sub:
            obj = _try_loads_candidates([sub])
            if obj is not None:
                return obj

    # 优先：包含 "start_indices" 的对象（话题切换等固定 schema，避免前文里其它 {{ 干扰）
    for m in re.finditer(r"\{\s*\"start_indices\"\s*:", s):
        sub = _balanced_brace_substring(s, m.start())
        if sub:
            obj = _try_loads_candidates([sub])
            if obj is not None:
                return obj

    # 从首个 { 配平（可去掉尾部多余说明）
    j0 = s.find("{")
    if j0 >= 0:
        sub = _balanced_brace_substring(s, j0)
        if sub:
            obj = _try_loads_candidates([sub])
            if obj is not None:
                return obj

    # 整段即为单个对象（旧路径）
    if s.startswith("{") and s.endswith("}"):
        return _loads_json_with_repair(s)

    # 回退：尝试每一个 '{' 位置（模型在 JSON 前加了短前缀时）
    i = 0
    while True:
        j = s.find("{", i)
        if j < 0:
            break
        sub = _balanced_brace_substring(s, j)
        if sub:
            obj = _try_loads_candidates([sub])
            if obj is not None:
                return obj
        i = j + 1

    raise ValueError(f"LLM output is not JSON object: {s[:200]}")


def _loads_json_with_repair(raw_json: str) -> Dict[str, Any]:
    s = raw_json
    for _ in range(32):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            if "Invalid \\escape" not in str(e):
                raise
            idx = max(0, e.pos - 1)
            if idx < len(s) and s[idx] == "\\":
                s = s[:idx] + "\\\\" + s[idx + 1 :]
            else:
                s = _repair_invalid_json_escapes(s)
    return json.loads(s)


def _repair_invalid_json_escapes(s: str) -> str:
    out: list[str] = []
    in_string = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '"':
            bs = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if in_string and ch == "\\":
            if i + 1 >= n:
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            valid_basic = {'"', "\\", "/", "b", "f", "n", "r", "t"}
            if nxt in valid_basic:
                out.append("\\")
                i += 1
                continue
            if nxt == "u":
                hex_part = s[i + 2 : i + 6]
                if len(hex_part) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    out.append("\\")
                    i += 1
                    continue
            out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)
