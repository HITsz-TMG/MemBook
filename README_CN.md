# MemBook: 面向抗幻觉智能体的分层推理记忆框架

官方实现：

**MemBook: A Hierarchical Reasoning-based Memory Framework for Hallucination-Resistant LLM Agents**

[English](README.md) | 中文文档

论文已被 **Findings of EMNLP 2026** 录用。

[GitHub](https://github.com/springCozyRock/MemBook) · [Issues](https://github.com/springCozyRock/MemBook/issues)

## Overview

我们提出 **MemBook**，一种分层、时间感知的记忆框架，用于提升长期 LLM Agent 的记忆可靠性。

MemBook 包含两个核心模块：

- **Stateful Memory Stabilization（有状态记忆稳定化）**
  - 通过 pending–confirmed 校验机制维护可靠记忆。
  - 用多车道隔离区分用户持久信息与短暂 QA / 交互内容。
  - 将记忆组织为层级结构：**Session → Chapter → Memory Point**。
- **Hierarchical Memory Retrieval（分层记忆检索）**
  - 在记忆层级上做基于推理的导航。
  - 联合词法检索、语义检索与推理检索。
  - 提升记忆访问的时间一致性与逻辑相关性。

在 **HaluMem-Medium**、**HaluMem-Long** 与 **LongMemEval** 上的实验表明，MemBook 提升了长期记忆抽取、检索与下游 QA 表现，并抑制幻觉跨阶段传播。

## Framework


<p align="center">
  <img src="./assets/membook-architecture.png" width="90%">
</p>


## Highlights

从 Medium 到 Long 表现保持稳定（默认 gpt-4o），在 HaluMem-Long 上相对代表性系统 **GAM 绝对提升 19.57%**：


| Benchmark                | 抽取 F1      | QA Acc.            |
| ------------------------ | ---------- | ------------------ |
| HaluMem-Medium           | **79.21%** | **67.82%**         |
| HaluMem-Long             | **79.69%** | **68.37%**         |
| LongMemEval（gpt-4o-mini） | —          | **69.4%**（overall） |




## 项目结构

```
MemBook/
├── membook/                              # 核心包
│   ├── agents/
│   │   ├── memory_agent.py               # 构建（MemoryAgent）
│   │   └── research_agent.py             # 检索（ResearchAgent）
│   ├── memory/toc/                       # ToC 构建、车道与稳定化
│   ├── generator/                        # 默认 OpenAI 兼容；可选 vLLM
│   ├── retriever/                        # Index / BM25 / dense
│   ├── prompts/                          # 构建 + 检索侧 prompt
│   ├── schemas/
│   ├── usage/                            # 可选 token 统计
│   └── config/
├── eval/halumem/                         # HaluMem：eval_membook.py + run_membook.bash
├── eval/baseline/longmemeval_membook.py  # LongMemEval 入口（供下方脚本调用）
├── scripts/eval_longmemeval_membook.sh
├── examples/quickstart/
├── assets/membook-architecture.png
├── pyproject.toml
├── requirements.txt
└── README_CN.md
```



## 安装

**环境要求**

- Python >= 3.10
- OpenAI API 或兼容 LLM 接口

```bash
git clone https://github.com/springCozyRock/MemBook.git
cd MemBook

pip install -e .

export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```



## 快速开始

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
        {"role": "user", "content": "我喜欢早上健身，常喝美式咖啡。"},
        {"role": "assistant", "content": "好的，记下早上健身和美式。"},
        {"role": "user", "content": "另外我对花生过敏。"},
        {"role": "assistant", "content": "已记录花生过敏。"},
    ],
    session_id="session_0",
    user_id="demo",
)

result = ResearchAgent(
    page_store=page_store,
    memory_store=memory_store,
    retrievers={},
    generator=generator,
).research("我对什么食物过敏？")
print(result.integrated_memory)
```

更多示例：`[examples/quickstart/](./examples/quickstart/)`。  
论文复现：`[eval/halumem/](./eval/halumem/)`（HaluMem）与 `scripts/eval_longmemeval_membook.sh`（LongMemEval）。

## 引用

```bibtex
@inproceedings{membook2026,
  title     = {MemBook: A Hierarchical Reasoning-based Memory Framework for Hallucination-Resistant LLM Agents},
  author    = {Bian, Tongshu and Li, Yunxin and Xu, Zhenran and Shi, Tianqi and Wang, Longyue and Hu, Baotian and Zhang, Min},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  note      = {To appear}
}
```



## 许可证

MIT — 见 [LICENSE](LICENSE)。