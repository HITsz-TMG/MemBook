# -*- coding: utf-8 -*-
"""
Retriever Module

This module contains all retrieval implementations for MemBook.
Retrievers provide different search strategies for finding relevant information.

Available Retrievers:
- DenseRetriever: Semantic search using dense vector embeddings
- BM25Retriever: Keyword-based search using BM25 algorithm
- IndexRetriever: Direct page access by index

BM25Retriever / DenseRetriever 为可选依赖，采用惰性导入，因此
``import membook`` 不会加载 faiss / pyserini 等重依赖。
"""

from __future__ import annotations

from typing import Any

from .base import AbsRetriever
from .index_retriever import IndexRetriever

__all__ = [
    "AbsRetriever",
    "IndexRetriever",
    "BM25Retriever",
    "DenseRetriever",
]


def __getattr__(name: str) -> Any:
    if name == "BM25Retriever":
        from .bm25 import BM25Retriever as _BM25Retriever

        return _BM25Retriever
    if name == "DenseRetriever":
        from .dense_retriever import DenseRetriever as _DenseRetriever

        return _DenseRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
