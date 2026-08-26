# -*- coding: utf-8 -*-
"""Shared HaluMem evaluation adapter infrastructure (stores / retrievers / research)."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from membook import (
    ResearchAgent,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)
from membook.compat import env

RETRY_TIMES = 10
WAIT_TIME = 60

MEMORY_TEMPERATURE = 0.3
MEMORY_MAX_TOKENS = 2048

RESEARCH_TEMPERATURE = 0.3
RESEARCH_MAX_TOKENS = 2048
RESEARCH_MAX_ITERS = 3
RESEARCH_MAX_DEEP_ITERS = 3

WORKING_TEMPERATURE = 0.3
WORKING_MAX_TOKENS = 256

BM25_THREADS = 1

TEMPLATE_MEMORY = """Memories for user {user_id}:

    {memories}
"""

# Back-compat alias
TEMPLATE_GAM = TEMPLATE_MEMORY


def _env_truthy(raw: Optional[str], default: str = "1") -> bool:
    v = raw if raw is not None else default
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _env_bm25_build_enabled() -> bool:
    """MEMBOOK_USE_BM25：未设为 0/false/off 时构建 BM25。"""
    return _env_truthy(env("MEMBOOK_USE_BM25", "1"))


def _env_dense_retriever_enabled() -> bool:
    """MEMBOOK_USE_DENSE_RETRIEVER：未设为 0/false/off 时构建 DenseRetriever。"""
    return _env_truthy(env("MEMBOOK_USE_DENSE_RETRIEVER", "1"))


def _env_page_index_build_enabled() -> bool:
    """与 ResearchAgent._use_page_index 一致；为 0 时不构建 page_index。"""
    raw = (env("MEMBOOK_USE_PAGE_INDEX", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _resolve_embedding_openai_credentials() -> Tuple[Optional[str], Optional[str]]:
    api_key = (os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip() or None
    base_url = (os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    return api_key, base_url


class HaluMemBaseAdapter:
    """Shared HaluMem plumbing: stores, retrievers, ResearchAgent, working LLM.

    Subclasses must set ``self.memory_agent`` and implement
    ``add_dialogue`` / ``get_dialogue_memory``.
    """

    def __init__(
        self,
        user_id: str,
        user_dir: str,
        memory_api_key: str,
        memory_base_url: str,
        memory_model: str,
        research_api_key: str,
        research_base_url: str,
        research_model: str,
        working_api_key: str,
        working_base_url: str,
        working_model: str,
        memory_api_type: str = "openai",
        research_api_type: str = "openai",
        working_api_type: str = "openai",
        research_max_iters: int = RESEARCH_MAX_ITERS,
        research_max_deep_iters: int = RESEARCH_MAX_DEEP_ITERS,
    ):
        self.user_id = user_id
        self.user_dir = user_dir
        self.session_counter = 0

        os.makedirs(user_dir, exist_ok=True)
        self.memory_store = InMemoryMemoryStore(dir_path=user_dir)
        self.page_store = InMemoryPageStore(dir_path=user_dir)

        if memory_api_type == "openai":
            memory_config = OpenAIGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=MEMORY_TEMPERATURE,
                max_tokens=MEMORY_MAX_TOKENS,
            )
            self._memory_generator = OpenAIGenerator(memory_config.__dict__)
        else:
            raise ValueError(f"Unsupported memory_api_type: {memory_api_type}")

        self.retrievers: Dict[str, Any] = {}
        self._build_retrievers()

        if research_api_type == "openai":
            research_config = OpenAIGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=RESEARCH_TEMPERATURE,
                max_tokens=RESEARCH_MAX_TOKENS,
            )
            research_generator = OpenAIGenerator(research_config.__dict__)
        else:
            raise ValueError(f"Unsupported research_api_type: {research_api_type}")

        self.research_agent = ResearchAgent(
            page_store=self.page_store,
            memory_store=self.memory_store,
            retrievers=self.retrievers,
            generator=research_generator,
            max_iters=research_max_iters,
            max_deep_iters=research_max_deep_iters,
        )

        if working_api_type == "openai":
            working_config = OpenAIGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=WORKING_TEMPERATURE,
                max_tokens=WORKING_MAX_TOKENS,
            )
            self.working_generator = OpenAIGenerator(working_config.__dict__)
        else:
            raise ValueError(f"Unsupported working_api_type: {working_api_type}")

        # Subclasses assign MemoryAgent / MemoryAgent
        self.memory_agent = None

    def _build_retrievers(self) -> None:
        label = self.__class__.__name__
        if _env_page_index_build_enabled():
            try:
                index_dir = os.path.join(self.user_dir, "page_index")
                if os.path.exists(index_dir):
                    shutil.rmtree(index_dir)
                index_config = IndexRetrieverConfig(index_dir=index_dir)
                index_retriever = IndexRetriever(index_config.__dict__)
                index_retriever.build(self.page_store)
                self.retrievers["page_index"] = index_retriever
            except Exception as e:
                print(f"[WARN] IndexRetriever 创建失败: {e}")
        else:
            print(f"[{label}] MEMBOOK_USE_PAGE_INDEX=0，跳过 IndexRetriever 构建")

        if _env_bm25_build_enabled():
            try:
                bm25_dir = os.path.join(self.user_dir, "bm25_index")
                if os.path.exists(bm25_dir):
                    shutil.rmtree(bm25_dir)
                bm25_config = BM25RetrieverConfig(
                    index_dir=bm25_dir,
                    threads=BM25_THREADS,
                )
                bm25_retriever = BM25Retriever(bm25_config.__dict__)
                bm25_retriever.build(self.page_store)
                self.retrievers["keyword"] = bm25_retriever
            except Exception as e:
                print(f"[WARN] BM25Retriever 创建失败: {e}")
        else:
            print(f"[{label}] MEMBOOK_USE_BM25=0，跳过 BM25Retriever 构建")

        if _env_dense_retriever_enabled():
            try:
                dense_dir = os.path.join(self.user_dir, "dense_index")
                if os.path.exists(dense_dir):
                    shutil.rmtree(dense_dir)
                emb_key, emb_base = _resolve_embedding_openai_credentials()
                dense_config = DenseRetrieverConfig(
                    index_dir=dense_dir,
                    use_openai_api=True,
                    openai_api_key=emb_key,
                    openai_base_url=emb_base,
                    openai_embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
                    openai_embedding_max_chars=8192,
                )
                dense_retriever = DenseRetriever(dense_config.__dict__)
                dense_retriever.build(self.page_store)
                self.retrievers["vector"] = dense_retriever
            except Exception as e:
                print(f"[WARN] DenseRetriever 创建失败: {e}")
                print(
                    "  提示：需设置 EMBEDDING_API_KEY（或 OPENAI_API_KEY）以使用 text-embedding-v4。"
                )
        else:
            print(f"[{label}] MEMBOOK_USE_DENSE_RETRIEVER=0，跳过 DenseRetriever 构建")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(WAIT_TIME),
        stop=stop_after_attempt(RETRY_TIMES),
        reraise=True,
    )
    def retrieve_memory(
        self,
        query: str,
        top_k: int = 5,
        use_research_agent: bool = True,
    ) -> Tuple[str, List[str], float]:
        import time

        start_time = time.time()

        if use_research_agent:
            research_result = self.research_agent.research(query)
            research_summary = research_result.integrated_memory
            memories = [research_summary] if research_summary else []
            context = TEMPLATE_MEMORY.format(
                user_id=self.user_id,
                memories=json.dumps(memories, indent=4, ensure_ascii=False),
            )
        else:
            all_hits = []
            for name, retriever in self.retrievers.items():
                try:
                    hits_list = retriever.search([query], top_k=top_k)
                    if hits_list:
                        all_hits.extend(hits_list[0])
                except Exception as e:
                    print(f"[WARN] {name}检索失败: {e}")

            unique_hits = {}
            for hit in all_hits:
                page_id = hit.page_id
                if page_id not in unique_hits:
                    unique_hits[page_id] = hit
                else:
                    existing_score = (
                        unique_hits[page_id].meta.get("score", 0)
                        if unique_hits[page_id].meta
                        else 0
                    )
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[page_id] = hit

            sorted_hits = sorted(
                unique_hits.values(),
                key=lambda h: h.meta.get("score", 0) if h.meta else 0,
                reverse=True,
            )[:top_k]

            memories = [
                f"Page {i}: {hit.snippet[:200]}"
                for i, hit in enumerate(sorted_hits, 1)
            ]
            context = TEMPLATE_MEMORY.format(
                user_id=self.user_id,
                memories=json.dumps(memories, indent=4, ensure_ascii=False),
            )

        duration_ms = (time.time() - start_time) * 1000
        return context, memories, duration_ms
