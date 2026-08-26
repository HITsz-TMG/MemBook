# -*- coding: utf-8 -*-
"""Thread-safe LLM API call and token usage aggregation."""

from __future__ import annotations

from membook.compat import env

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union


def _env_encoding() -> str:
    return (env("MEMBOOK_USAGE_TIKTOKEN_ENCODING") or "cl100k_base").strip()


def _messages_to_text(messages: Optional[List[Dict[str, str]]]) -> str:
    if not messages:
        return ""
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def estimate_token_count(text: str, *, encoding_name: Optional[str] = None) -> int:
    """Estimate tokens via tiktoken; fall back to len(text)//4."""
    s = text or ""
    if not s:
        return 0
    enc_name = encoding_name or _env_encoding()
    try:
        import tiktoken

        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(s))
    except Exception:
        return max(1, len(s) // 4) if s else 0


def _usage_fields(usage: Any) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    tt = int(getattr(usage, "total_tokens", 0) or 0)
    if tt <= 0 and (pt > 0 or ct > 0):
        tt = pt + ct
    return pt, ct, tt


@dataclass
class CategoryUsage:
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_from_api: int = 0
    completion_tokens_from_api: int = 0
    prompt_tokens_estimated: int = 0
    completion_tokens_estimated: int = 0
    usage_missing_calls: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens_from_api": self.prompt_tokens_from_api,
            "completion_tokens_from_api": self.completion_tokens_from_api,
            "prompt_tokens_estimated": self.prompt_tokens_estimated,
            "completion_tokens_estimated": self.completion_tokens_estimated,
            "usage_missing_calls": self.usage_missing_calls,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "CategoryUsage":
        if not data:
            return cls()
        return cls(
            api_calls=int(data.get("api_calls", 0) or 0),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            prompt_tokens_from_api=int(data.get("prompt_tokens_from_api", 0) or 0),
            completion_tokens_from_api=int(data.get("completion_tokens_from_api", 0) or 0),
            prompt_tokens_estimated=int(data.get("prompt_tokens_estimated", 0) or 0),
            completion_tokens_estimated=int(data.get("completion_tokens_estimated", 0) or 0),
            usage_missing_calls=int(data.get("usage_missing_calls", 0) or 0),
        )


@dataclass
class LLMUsageStats:
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_from_api: int = 0
    completion_tokens_from_api: int = 0
    prompt_tokens_estimated: int = 0
    completion_tokens_estimated: int = 0
    usage_missing_calls: int = 0
    by_category: Dict[str, CategoryUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _category(self, category: str) -> CategoryUsage:
        key = category or "llm"
        if key not in self.by_category:
            self.by_category[key] = CategoryUsage()
        return self.by_category[key]

    def record(
        self,
        usage: Any,
        category: str = "llm",
        *,
        prompt_text: str = "",
        completion_text: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Record one successful chat completion. Failed calls must not invoke this."""
        if messages is not None and not prompt_text:
            prompt_text = _messages_to_text(messages)

        pt_api, ct_api, tt_api = _usage_fields(usage)
        from_api = usage is not None and (pt_api > 0 or ct_api > 0 or tt_api > 0)

        if from_api:
            pt, ct = pt_api, ct_api
            tt = tt_api if tt_api > 0 else pt + ct
            pt_est = ct_est = 0
            missing = 0
        else:
            pt_est = estimate_token_count(prompt_text)
            ct_est = estimate_token_count(completion_text)
            pt, ct = pt_est, ct_est
            tt = pt + ct
            pt_api = ct_api = 0
            missing = 1

        with self._lock:
            self.api_calls += 1
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.total_tokens += tt
            self.prompt_tokens_from_api += pt_api
            self.completion_tokens_from_api += ct_api
            self.prompt_tokens_estimated += pt_est
            self.completion_tokens_estimated += ct_est
            self.usage_missing_calls += missing

            cat = self._category(category)
            cat.api_calls += 1
            cat.prompt_tokens += pt
            cat.completion_tokens += ct
            cat.total_tokens += tt
            cat.prompt_tokens_from_api += pt_api
            cat.completion_tokens_from_api += ct_api
            cat.prompt_tokens_estimated += pt_est
            cat.completion_tokens_estimated += ct_est
            cat.usage_missing_calls += missing

    def merge(self, other: "LLMUsageStats") -> None:
        with self._lock:
            self.api_calls += other.api_calls
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.total_tokens += other.total_tokens
            self.prompt_tokens_from_api += other.prompt_tokens_from_api
            self.completion_tokens_from_api += other.completion_tokens_from_api
            self.prompt_tokens_estimated += other.prompt_tokens_estimated
            self.completion_tokens_estimated += other.completion_tokens_estimated
            self.usage_missing_calls += other.usage_missing_calls
            for key, cat in other.by_category.items():
                dst = self._category(key)
                dst.api_calls += cat.api_calls
                dst.prompt_tokens += cat.prompt_tokens
                dst.completion_tokens += cat.completion_tokens
                dst.total_tokens += cat.total_tokens
                dst.prompt_tokens_from_api += cat.prompt_tokens_from_api
                dst.completion_tokens_from_api += cat.completion_tokens_from_api
                dst.prompt_tokens_estimated += cat.prompt_tokens_estimated
                dst.completion_tokens_estimated += cat.completion_tokens_estimated
                dst.usage_missing_calls += cat.usage_missing_calls

    def token_source(self) -> str:
        if self.usage_missing_calls <= 0:
            return "api"
        if self.usage_missing_calls >= self.api_calls and self.api_calls > 0:
            return "estimated"
        return "mixed"

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "api_calls": self.api_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_tokens_from_api": self.prompt_tokens_from_api,
                "completion_tokens_from_api": self.completion_tokens_from_api,
                "prompt_tokens_estimated": self.prompt_tokens_estimated,
                "completion_tokens_estimated": self.completion_tokens_estimated,
                "usage_missing_calls": self.usage_missing_calls,
                "token_source": self.token_source(),
                "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "LLMUsageStats":
        stats = cls()
        if not data:
            return stats
        stats.api_calls = int(data.get("api_calls", 0) or 0)
        stats.prompt_tokens = int(data.get("prompt_tokens", 0) or 0)
        stats.completion_tokens = int(data.get("completion_tokens", 0) or 0)
        stats.total_tokens = int(data.get("total_tokens", 0) or 0)
        stats.prompt_tokens_from_api = int(data.get("prompt_tokens_from_api", 0) or 0)
        stats.completion_tokens_from_api = int(data.get("completion_tokens_from_api", 0) or 0)
        stats.prompt_tokens_estimated = int(data.get("prompt_tokens_estimated", 0) or 0)
        stats.completion_tokens_estimated = int(data.get("completion_tokens_estimated", 0) or 0)
        stats.usage_missing_calls = int(data.get("usage_missing_calls", 0) or 0)
        raw_cats = data.get("by_category") or {}
        if isinstance(raw_cats, dict):
            stats.by_category = {
                str(k): CategoryUsage.from_dict(v) for k, v in raw_cats.items()
            }
        return stats


def merge_llm_usage_dicts(
    usages: List[Optional[Union[Dict[str, Any], Mapping[str, Any]]]],
) -> Dict[str, Any]:
    total = LLMUsageStats()
    for item in usages:
        if item:
            total.merge(LLMUsageStats.from_dict(item))
    return total.to_dict()
