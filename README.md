# MemBook: A Hierarchical Reasoning-based Memory Framework for Hallucination-Resistant LLM Agents

Official implementation of:

**MemBook: A Hierarchical Reasoning-based Memory Framework for Hallucination-Resistant LLM Agents**

[中文文档](README_CN.md) | English

Accepted to **Findings of EMNLP 2026**.

[GitHub](https://github.com/springCozyRock/MemBook) · [Issues](https://github.com/springCozyRock/MemBook/issues)

## Overview

We introduce **MemBook**, a hierarchical time-aware memory framework designed to improve memory reliability for long-term LLM agents.

MemBook contains two key components:

- **Stateful Memory Stabilization**
  - Maintains reliable memory through a pending–confirmed verification mechanism.
  - Separates durable user information from transient QA and interaction content using multi-lane memory isolation.
  - Organizes memories into a hierarchical structure: **Session → Chapter → Memory Point**.
- **Hierarchical Memory Retrieval**
  - Performs reasoning-based navigation over the memory hierarchy.
  - Combines lexical retrieval, semantic retrieval, and reasoning-based retrieval.
  - Improves temporal consistency and logical relevance during memory access.

Our experiments on **HaluMem-Medium**, **HaluMem-Long**, and **LongMemEval** demonstrate that MemBook improves long-term memory extraction, retrieval, and downstream QA performance while reducing hallucination propagation.

## Framework

<p align="center">
  <img src="./assets/membook-architecture.png" width="90%">
</p>

## Highlights

Results remain stable from Medium to Long (gpt-4o unless noted), outperforming the representative system **GAM by absolute 19.57%** on HaluMem-Long:

| Benchmark | Extraction F1 | QA Acc. |
|-----------|---------------|---------|
| HaluMem-Medium | **79.21%** | **67.82%** |
| HaluMem-Long | **79.69%** | **68.37%** |
| LongMemEval (gpt-4o-mini) | — | **69.4%** (overall) |

## Project Structure

```
MemBook/
├── membook/                              # Core package
│   ├── agents/
│   │   ├── memory_agent.py               # Construction (MemoryAgent)
│   │   └── research_agent.py             # Retrieval (ResearchAgent)
│   ├── memory/toc/                       # ToC builder, lanes, stabilization
│   ├── generator/                        # OpenAI-compatible (default) / optional vLLM
│   ├── retriever/                        # Index / BM25 / dense
│   ├── prompts/                          # Construction + retrieval prompts
│   ├── schemas/
│   ├── usage/                            # Optional LLM usage stats
│   └── config/
├── eval/halumem/                         # HaluMem: eval_membook.py + run_membook.bash
├── eval/baseline/longmemeval_membook.py  # LongMemEval runner (used by script below)
├── scripts/eval_longmemeval_membook.sh
├── examples/quickstart/
├── assets/membook-architecture.png
├── pyproject.toml
├── requirements.txt
└── README.md
```



## Installation

**Requirements**

- Python >= 3.10
- OpenAI API or compatible LLM endpoint

```bash
git clone https://github.com/springCozyRock/MemBook.git
cd MemBook

pip install -e .

export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```



## Quick Start

```python
import os
from membook import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
)

generator = OpenAIGenerator.from_config(
    OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
)
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()

memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store,
)
memory_agent.memorize_dialogue(
    [
        {"role": "user", "content": "I prefer morning workouts and drink black coffee."},
        {"role": "assistant", "content": "Got it — mornings and black coffee."},
        {"role": "user", "content": "Also allergic to peanuts."},
        {"role": "assistant", "content": "Noted, no peanuts."},
    ],
    session_id="session_0",
    user_id="demo",
)

result = ResearchAgent(
    page_store=page_store,
    memory_store=memory_store,
    retrievers={},
    generator=generator,
).research("What are my food allergies?")
print(result.integrated_memory)
```

More examples: [`examples/quickstart/`](./examples/quickstart/).  
Reproduce paper results: [`eval/halumem/`](./eval/halumem/) (HaluMem) and `scripts/eval_longmemeval_membook.sh` (LongMemEval).

## Citation

```bibtex
@inproceedings{membook2026,
  title     = {MemBook: A Hierarchical Reasoning-based Memory Framework for Hallucination-Resistant LLM Agents},
  author    = {Bian, Tongshu and Li, Yunxin and Xu, Zhenran and Shi, Tianqi and Wang, Longyue and Hu, Baotian and Zhang, Min},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  note      = {To appear}
}
```



## License

MIT — see [LICENSE](LICENSE).