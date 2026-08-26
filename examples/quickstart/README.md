# MemBook 快速开始示例

本目录提供 MemBook（Findings of EMNLP 2026：分层 ToC 记忆 + Research 检索）的入门示例。

## 文件说明

### 1. `basic_usage.py`
- 用 `MemoryAgent` 从对话构建 Session–Chapter–Memory Point 记忆
- 用 `ResearchAgent` 做规划式检索与整合问答

### 2. `model_usage.py`
- OpenAI / 兼容端点 / vLLM 本地模型配置

### 3. `ttl_usage.py`
- `TTLMemoryStore` / `TTLPageStore` 生产清理

## 快速开始

```bash
pip install -e .

export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"

cd examples/quickstart
python basic_usage.py
```

## 核心用法（节选）

```python
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
        temperature=0.3,
    )
)
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()

memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store,
)
memory_agent.memorize_dialogue(dialogue, session_id="session_0", user_id="demo")

research_agent = ResearchAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store,
    retrievers={},  # 可挂 BM25 / dense / index，见 basic_usage.py
    max_iters=3,
)
result = research_agent.research("你的问题")
print(result.integrated_memory)
```

检索器示例（dense / BM25 / page index）见 `basic_usage.py` 与主文档 [`README_CN.md`](../../README_CN.md)。

## 论文复现

主结果在 **HaluMem** / **LongMemEval**，见：

- [`eval/halumem/README.md`](../../eval/halumem/README.md)
- [`README_CN.md`](../../README_CN.md)「复现论文结果」

## 相关文档

- [README](../../README.md) / [中文 README](../../README_CN.md)
