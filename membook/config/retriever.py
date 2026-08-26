from dataclasses import dataclass, field
from typing import Any, Union, List


@dataclass
class DenseRetrieverConfig:
    """密集向量检索器配置"""
    model_name: str = ""  # 仅 use_openai_api=False 时用；用 API 时用 openai_embedding_model
    model_class: str = "encoder-only-base"
    normalize_embeddings: bool = True
    pooling_method: str = "cls"
    trust_remote_code: bool = True
    query_instruction_for_retrieval: str | None = None
    use_fp16: bool = False
    devices: List[str] = field(default_factory=lambda: ["cuda:0"])
    batch_size: int = 32
    max_length: int = 512
    index_dir: str = "./index/dense"
    api_url: str | None = None
    # OpenAI 兼容 embedding API（如 text-embedding-v4）
    use_openai_api: bool = False
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-v4"
    openai_embedding_max_chars: int = 8192  # 单条输入截断长度，避免超 API 限制（如 8192 tokens）


@dataclass
class IndexRetrieverConfig:
    """索引检索器配置"""
    index_dir: str = "./index/index"


@dataclass
class BM25RetrieverConfig:
    """BM25关键词检索器配置"""
    index_dir: str = "./index/bm25"
    threads: int = 4