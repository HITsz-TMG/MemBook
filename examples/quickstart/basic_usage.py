#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemBook Basic Usage Example

Build ToC memory with MemoryAgent, then answer with ResearchAgent.
"""

import os

if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"

from membook import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    IndexRetrieverConfig,
    BM25Retriever,
    BM25RetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
)


def basic_memory_example():
    print("=== MemBook MemoryAgent example ===\n")

    gen_config = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.3,
        max_tokens=1024,
    )
    generator = OpenAIGenerator.from_config(gen_config)

    memory_store = InMemoryMemoryStore()
    page_store = InMemoryPageStore()

    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store,
    )

    dialogue = [
        {
            "role": "user",
            "content": "I prefer morning workouts and drink black coffee every day.",
            "timestamp": "Jan 1, 2026, 08:00:00",
        },
        {
            "role": "assistant",
            "content": "Noted: morning workouts and black coffee.",
            "timestamp": "Jan 1, 2026, 08:00:05",
        },
        {
            "role": "user",
            "content": "I am allergic to peanuts.",
            "timestamp": "Jan 1, 2026, 08:01:00",
        },
        {
            "role": "assistant",
            "content": "Got it — peanut allergy recorded.",
            "timestamp": "Jan 1, 2026, 08:01:05",
        },
    ]

    print("Building ToC memory from one session...")
    memory_agent.memorize_dialogue(dialogue, session_id="session_0", user_id="demo")
    pages = page_store.load()
    print(f"✅ Pages stored: {len(pages)}")
    return memory_agent, memory_store, page_store, generator


def research_example(memory_store, page_store, generator):
    print("\n=== ResearchAgent Q&A ===\n")

    retrievers = {}
    index_dir = "./tmp_membook_quickstart"
    os.makedirs(index_dir, exist_ok=True)

    try:
        page_index_dir = os.path.join(index_dir, "page_index")
        if os.path.exists(page_index_dir):
            import shutil

            shutil.rmtree(page_index_dir)
        index_config = IndexRetrieverConfig(index_dir=page_index_dir)
        index_retriever = IndexRetriever(index_config.__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
        print("✅ page_index ready")
    except Exception as e:
        print(f"[WARN] page_index: {e}")

    try:
        bm25_index_dir = os.path.join(index_dir, "bm25_index")
        if os.path.exists(bm25_index_dir):
            import shutil

            shutil.rmtree(bm25_index_dir)
        bm25_config = BM25RetrieverConfig(index_dir=bm25_index_dir, threads=1)
        bm25_retriever = BM25Retriever(bm25_config.__dict__)
        bm25_retriever.build(page_store)
        retrievers["keyword"] = bm25_retriever
        print("✅ BM25 ready")
    except Exception as e:
        print(f"[WARN] BM25: {e}")

    try:
        dense_index_dir = os.path.join(index_dir, "dense_index")
        if os.path.exists(dense_index_dir):
            import shutil

            shutil.rmtree(dense_index_dir)
        dense_config = DenseRetrieverConfig(
            index_dir=dense_index_dir,
            use_openai_api=True,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        )
        dense_retriever = DenseRetriever(dense_config.__dict__)
        dense_retriever.build(page_store)
        retrievers["vector"] = dense_retriever
        print("✅ Dense ready")
    except Exception as e:
        print(f"[WARN] Dense: {e}")

    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=generator,
        max_iters=3,
    )

    question = "What food am I allergic to?"
    print(f"Question: {question}")
    research_result = research_agent.research(question)
    print(f"Answer/context: {research_result.integrated_memory}")
    return research_result


def main():
    print("MemBook quickstart\n" + "=" * 40)
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY (and optionally OPENAI_BASE_URL).")
        return
    _, memory_store, page_store, generator = basic_memory_example()
    research_example(memory_store, page_store, generator)
    print("\nDone.")


if __name__ == "__main__":
    main()
