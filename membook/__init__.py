# -*- coding: utf-8 -*-
"""
MemBook

A dual-agent architecture for building long-term memory with deep research capabilities.

Key Components:
- MemoryAgent: Builds structured ToC memory from dialogue sessions
- ResearchAgent: Performs multi-iteration research with reflection
"""

from __future__ import annotations

from typing import Any

# Core agents
from membook.agents import MemoryAgent, ResearchAgent

# Generators
from membook.generator import AbsGenerator, OpenAIGenerator, VLLMGenerator

# Retrievers（BM25 / Dense 为可选依赖，见 membook.retriever 惰性 __getattr__）
from membook.retriever import AbsRetriever, IndexRetriever

# Configurations
from membook.config import (
    OpenAIGeneratorConfig,
    VLLMGeneratorConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    IndexRetrieverConfig
)

# Schemas
from membook.schemas import (
    MemoryState,
    Page,
    MemoryUpdate,
    SearchPlan,
    Hit,
    Result,
    EnoughDecision,
    ReflectionDecision,
    ResearchOutput,
    InMemoryMemoryStore,
    InMemoryPageStore,
    TTLMemoryStore,
    TTLPageStore,
)

__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    if name in {"BM25Retriever", "DenseRetriever"}:
        from membook import retriever

        return getattr(retriever, name)
    if name == "TocMemoryAgent":
        # Backward-compatible alias during rename.
        return MemoryAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core agents
    "MemoryAgent",
    "ResearchAgent",

    # Generators
    "AbsGenerator",
    "OpenAIGenerator",
    "VLLMGenerator",

    # Retrievers
    "AbsRetriever",
    "IndexRetriever",
    "BM25Retriever",
    "DenseRetriever",

    # Configurations
    "OpenAIGeneratorConfig",
    "VLLMGeneratorConfig",
    "DenseRetrieverConfig",
    "BM25RetrieverConfig",
    "IndexRetrieverConfig",

    # Schemas
    "MemoryState",
    "Page",
    "MemoryUpdate",
    "SearchPlan",
    "Hit",
    "Result",
    "EnoughDecision",
    "ReflectionDecision",
    "ResearchOutput",
    "InMemoryMemoryStore",
    "InMemoryPageStore",
    "TTLMemoryStore",
    "TTLPageStore",
]
