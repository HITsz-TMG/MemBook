# -*- coding: utf-8 -*-
"""Bridge AbsGenerator to TocBuilder LLMClient.complete()."""

from __future__ import annotations

from typing import Any, Dict, Optional

from membook.generator import AbsGenerator

from .json_llm import LLMClient


class AbsGeneratorLLMClient:
    """Implements LLMClient for toc.TocBuilder using OpenAI/VLLM generator."""

    def __init__(self, generator: AbsGenerator) -> None:
        self._generator = generator

    def complete(self, prompt: str, *, response_json_object: bool = False) -> str:
        extra_params: Optional[Dict[str, Any]] = None
        if response_json_object:
            extra_params = {"response_format": {"type": "json_object"}}
        resp = self._generator.generate_single(prompt=prompt, extra_params=extra_params)
        if not resp:
            return ""
        return str(resp.get("text") or "")
