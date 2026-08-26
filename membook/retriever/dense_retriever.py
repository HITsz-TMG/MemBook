from __future__ import annotations

import os

# 与 Java(Lucene/BM25) 同进程时，限制 OpenMP 线程数，避免 libomp pthread_mutex_init 崩溃（macOS 上常见）
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import json
from typing import Dict, Any, List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

from membook.retriever.base import AbsRetriever
from membook.schemas import InMemoryPageStore, Hit, Page

try:
    from FlagEmbedding import FlagAutoModel
except ImportError:
    FlagAutoModel = None  # type: ignore


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    构建 FAISS 索引
    embeddings: (n, dim) 的 numpy 数组
    """
    dimension = embeddings.shape[1]
    # 使用内积索引（cosine similarity）
    index = faiss.IndexFlatIP(dimension)
    # L2 归一化以支持 cosine similarity（复制数组以避免修改原始数据）
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    index.add(embeddings_normalized)
    return index


def _search_faiss_index(index: faiss.Index, query_embeddings: np.ndarray, top_k: int):
    """
    在 FAISS 索引中搜索
    index: FAISS 索引
    query_embeddings: (n_queries, dim) 的查询向量
    top_k: 返回的 top-k 结果数
    返回: (scores_list, indices_list) 其中每个元素都是 (top_k,) 的数组
    """
    # L2 归一化查询向量（复制以避免修改原始数据）
    query_embeddings_normalized = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_normalized)
    
    # 搜索
    scores, indices = index.search(query_embeddings_normalized, top_k)
    
    scores_list = [scores[i] for i in range(len(query_embeddings))]
    indices_list = [indices[i] for i in range(len(query_embeddings))]
    
    return scores_list, indices_list


class DenseRetriever(AbsRetriever):
    _META_SCHEMA_VERSION = 1

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if faiss is None or np is None:
            raise ImportError(
                "DenseRetriever requires optional dependencies; "
                'install them with `pip install "membook[dense]"`.'
            )
        self.pages = None
        self.index = None
        self.doc_emb = None
        self._last_indexed_page_count = 0

        # 优先级: use_openai_api > api_url > 本地模型
        self.use_openai_api = config.get("use_openai_api", False)
        self.api_url = config.get("api_url")
        self.use_api = self.use_openai_api or (self.api_url is not None)

        # 单条 embedding 输入截断长度（OpenAI/兼容 API 普遍 8192 tokens，按字符保守截断）
        self.openai_embedding_max_chars = config.get("openai_embedding_max_chars", self.OPENAI_EMBED_MAX_CHARS)
        if self.use_openai_api:
            # OpenAI 兼容 API 模式（如 text-embedding-v4）
            from openai import OpenAI
            api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            base_url = config.get("openai_base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if not api_key:
                raise ValueError("[DenseRetriever] use_openai_api=True 需要 openai_api_key 或环境变量 OPENAI_API_KEY")
            self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
            self.openai_embedding_model = config.get("openai_embedding_model", "text-embedding-v4")
            self.openai_batch_size = config.get("openai_embedding_batch_size", 10)  # text-embedding-v4 建议 10
            self.model = None
            print(f"[DenseRetriever] 使用 OpenAI 兼容 API: {self.openai_embedding_model}, base_url={base_url}, max_chars={self.openai_embedding_max_chars}")
        elif self.api_url:
            if requests is None:
                raise ImportError(
                    "Custom-API dense retrieval requires requests; "
                    'install it with `pip install "membook[dense]"`.'
                )
            # 自定义 API 模式
            print(f"[DenseRetriever] 使用自定义 API 模式: {self.api_url}")
            self.model = None
            self.openai_client = None
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[DenseRetriever] API 服务连接成功: {response.json()}")
                else:
                    print(f"[DenseRetriever] 警告: API 服务响应异常: {response.status_code}")
            except Exception as e:
                print(f"[DenseRetriever] 警告: 无法连接到 API 服务: {e}")
        else:
            # 本地模式：加载 BGE 等模型
            self.openai_client = None
            if FlagAutoModel is None:
                raise ImportError("[DenseRetriever] 本地模式需要安装 FlagEmbedding: pip install FlagEmbedding")
            model_name = config.get("model_name") or ""
            if not model_name:
                raise ValueError(
                    "[DenseRetriever] 本地模式需指定 model_name，或改用 use_openai_api=True 使用 text-embedding-v4"
                )
            print(f"[DenseRetriever] 使用本地模式，加载模型: {model_name}")
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                default_device = "cuda:0" if has_cuda else "cpu"
                devices = config.get("devices", default_device)
                self.model = FlagAutoModel.from_finetuned(
                    model_name,
                    model_class=config.get("model_class", None),
                    normalize_embeddings=config.get("normalize_embeddings", True),
                    pooling_method=config.get("pooling_method", "cls"),
                    trust_remote_code=config.get("trust_remote_code", True),
                    query_instruction_for_retrieval=config.get("query_instruction_for_retrieval"),
                    use_fp16=config.get("use_fp16", False),
                    devices=devices
                )
                if self.model is None:
                    raise RuntimeError(f"模型加载失败：FlagAutoModel.from_finetuned() 返回了 None")
                print(f"[DenseRetriever] 模型加载成功，使用设备: {devices}")
            except Exception as e:
                error_msg = (
                    f"[DenseRetriever] 模型加载失败: {e}\n"
                    f"  模型名称: {model_name}\n"
                    f"  请检查: 1) 模型名称是否正确 2) 网络连接是否正常 3) 是否有足够的磁盘空间"
                )
                print(error_msg)
                raise RuntimeError(error_msg) from e


    # ---------- 内部小工具 ----------
    def _index_dir(self) -> str:
        return self.config["index_dir"]

    def _pages_dir(self) -> str:
        return os.path.join(self._index_dir(), "pages")

    def _emb_path(self) -> str:
        return os.path.join(self._index_dir(), "doc_emb.npy")

    def _meta_path(self) -> str:
        return os.path.join(self._index_dir(), "index_meta.json")

    def _save_meta(self) -> None:
        os.makedirs(self._index_dir(), exist_ok=True)
        payload = {
            "schema_version": self._META_SCHEMA_VERSION,
            "last_indexed_page_count": int(self._last_indexed_page_count),
        }
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_meta(self) -> None:
        path = self._meta_path()
        if not os.path.exists(path):
            self._last_indexed_page_count = 0
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._last_indexed_page_count = int(data.get("last_indexed_page_count", 0) or 0)
        except Exception:
            # 元数据损坏时保守退化：后续 update 将走重建兜底。
            self._last_indexed_page_count = 0

    def _encode_via_api(self, texts: List[str], encode_type: str = "corpus") -> np.ndarray:
        """
        通过 API 编码文本
        
        Args:
            texts: 文本列表
            encode_type: "corpus" 或 "query"
        
        Returns:
            embeddings as numpy array
        """
        # 验证输入
        if not texts:
            raise ValueError(f"[DenseRetriever] 文本列表为空，无法编码")
        # 与 _encode_via_openai 一致：单条截断，避免超长（如 text-embedding-v4 的 8192 token 限制）
        max_chars = getattr(self, "openai_embedding_max_chars", self.OPENAI_EMBED_MAX_CHARS)
        def _trunc(s: str) -> str:
            s = (s or "").strip() or " "
            return s[:max_chars] if len(s) > max_chars else s
        texts = [_trunc(t) for t in texts]

        # 过滤掉空字符串（占位保留顺序）
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError(f"[DenseRetriever] 所有文本都为空，无法编码")
        
        if len(non_empty_texts) != len(texts):
            print(f"[DenseRetriever] 警告: 过滤掉了 {len(texts) - len(non_empty_texts)} 个空文本")
        
        try:
            request_data = {
                "texts": non_empty_texts,
                "type": encode_type,
                "batch_size": self.config.get("batch_size", 32),
                "max_length": self.config.get("max_length", 512),
            }
            
            response = requests.post(
                f"{self.api_url}/encode",
                json=request_data,
                timeout=300  # 5分钟超时，大批量编码可能需要较长时间
            )
            
            # 如果请求失败，打印详细的错误信息
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_response = response.json()
                    error_detail = f" 服务器错误信息: {error_response}"
                except:
                    error_detail = f" 响应内容: {response.text[:500]}"
                
                error_msg = (
                    f"[DenseRetriever] API 编码失败: {response.status_code} {response.reason}\n"
                    f"  请求URL: {self.api_url}/encode\n"
                    f"  请求参数: texts数量={len(non_empty_texts)}, type={encode_type}, "
                    f"batch_size={request_data['batch_size']}, max_length={request_data['max_length']}\n"
                    f"{error_detail}"
                )
                print(error_msg)
                response.raise_for_status()
            
            result = response.json()
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            
            # 如果过滤了空文本，需要补充空向量以保持索引对应
            if len(non_empty_texts) != len(texts):
                # 找到空文本的位置
                empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
                # 构建完整的 embeddings 数组
                full_embeddings = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
                non_empty_idx = 0
                for i in range(len(texts)):
                    if i not in empty_indices:
                        full_embeddings[i] = embeddings[non_empty_idx]
                        non_empty_idx += 1
                embeddings = full_embeddings
            
            return embeddings
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"[DenseRetriever] API 编码失败（网络错误）: {e}\n"
                f"  请求URL: {self.api_url}/encode\n"
                f"  请检查: 1) API服务是否运行 2) URL是否正确 3) 网络连接是否正常"
            )
            print(error_msg)
            raise
        except Exception as e:
            print(f"[DenseRetriever] API 编码失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    # 多数 embedding API（如 text-embedding-v4）要求单条输入长度在 [1, 8192]（字符或 token），超出会报 InvalidParameter
    OPENAI_EMBED_MAX_CHARS = 8192

    def _encode_via_openai(self, texts: List[str]) -> np.ndarray:
        """
        通过 OpenAI 兼容 API 编码文本（如 text-embedding-v4）
        texts: 文本列表，corpus 和 query 使用相同接口
        API 要求单条长度 [1, 8192]（token），空串会报 InvalidParameter，过长会报错，故做过滤与截断。
        """
        if not texts:
            raise ValueError("[DenseRetriever] 文本列表为空，无法编码")
        # 过滤空串，并将空位用占位符代替以便与原文顺序对应
        max_chars = getattr(self, "openai_embedding_max_chars", self.OPENAI_EMBED_MAX_CHARS)
        processed: List[str] = []
        for t in texts:
            if not t or not str(t).strip():
                processed.append(" ")  # API 要求长度至少 1，用空格避免 InvalidParameter
            else:
                s = str(t).strip()
                if len(s) > max_chars:
                    s = s[:max_chars]
                processed.append(s)
        batch_size = getattr(self, "openai_batch_size", 10)
        all_embeddings: List[List[float]] = []
        for i in range(0, len(processed), batch_size):
            batch = processed[i : i + batch_size]
            resp = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=batch,
            )
            for item in resp.data:
                all_embeddings.append(item.embedding)
        return np.array(all_embeddings, dtype=np.float32)

    def _encode_pages(self, pages: List[Page]) -> np.ndarray:
        # 和 build() / update() 保持一致的编码方式
        # 处理可能的 None 值
        texts = []
        for p in pages:
            header = p.header if p.header is not None else ""
            content = p.content if p.content is not None else ""
            text = (header + " " + content).strip()
            text = p.content
            texts.append(text)
        
        if self.use_openai_api:
            return self._encode_via_openai(texts)
        if self.api_url:
            return self._encode_via_api(texts, encode_type="corpus")
        else:
            # 本地模式
            if self.model is None:
                raise RuntimeError("DenseRetriever 模型未初始化，无法编码。请检查模型加载是否成功。")
            return self.model.encode_corpus(
                texts,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

    # ---------- 对外接口 ----------
    def load(self) -> None:
        """
        从磁盘恢复：
        - pages 快照
        - doc_emb.npy
        - faiss 索引

        若 ``doc_emb.npy`` 不存在（从未 build、或 point-first 下尚无 memory point 页因而未落盘），
        视为空索引，**不打印**；仅当文件存在但损坏或 pages 快照异常时再告警。
        """
        emb_path = self._emb_path()
        self._load_meta()
        if not os.path.isfile(emb_path):
            self.doc_emb = None
            self.index = None
            self.pages = []
            return

        try:
            self.doc_emb = np.load(emb_path)
            self.index = _build_faiss_index(self.doc_emb)
            self.pages = InMemoryPageStore(dir_path=self._pages_dir()).load()
            if self._last_indexed_page_count <= 0 and self.pages:
                max_global = max(
                    int((p.meta or {}).get("page_global_index", -1))
                    for p in self.pages
                    if str((p.meta or {}).get("memory_point_id") or "").strip()
                )
                self._last_indexed_page_count = max_global + 1 if max_global >= 0 else 0
        except Exception as e:
            self.doc_emb = None
            self.index = None
            self.pages = []
            print(f"[DenseRetriever] load failed (index may be corrupt or pages snapshot missing): {e}")

    def build(self, page_store: InMemoryPageStore) -> None:
        """
        point-first：只给带有 meta.memory_point_id 的 point pages 建索引。
        """
        os.makedirs(self._pages_dir(), exist_ok=True)

        pages_all = page_store.load()
        if not pages_all:
            self.doc_emb = None
            self.index = None
            self.pages = []
            self._last_indexed_page_count = 0
            temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
            temp_page_store.save([])
            emb_path = self._emb_path()
            if os.path.exists(emb_path):
                os.remove(emb_path)
            self._save_meta()
            return

        # 1) Select point pages (strict: 仅 confirmed memory point 页参与索引)
        pages_index: List[Page] = []
        for global_idx, p in enumerate(pages_all):
            pmeta = p.meta or {}
            mp_id = str(pmeta.get("memory_point_id") or "").strip()
            if mp_id:
                # Persist global page index for later mapping to main page_store indices.
                pages_index.append(
                    p.model_copy(update={"meta": {**pmeta, "page_global_index": global_idx}})
                )

        self.pages = pages_index

        # 无 point 页 → 空索引、空持久化；检索返回空列表（无冷启动全量兜底）。
        if not self.pages:
            self.doc_emb = None
            self.index = None
            self._last_indexed_page_count = len(pages_all)
            temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
            temp_page_store.save([])
            emb_path = self._emb_path()
            if os.path.exists(emb_path):
                os.remove(emb_path)
            self._save_meta()
            return

        # 2) Encode
        self.doc_emb = self._encode_pages(self.pages)

        # 3) Build faiss index
        self.index = _build_faiss_index(self.doc_emb)

        # 4) Persist
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(self.pages)
        np.save(self._emb_path(), self.doc_emb)
        self._last_indexed_page_count = len(pages_all)
        self._save_meta()

    def update(self, page_store: InMemoryPageStore) -> None:
        try:
            pages_all = page_store.load()
            current_count = len(pages_all)

            if self.pages is None:
                self.load()

            if self._last_indexed_page_count < 0:
                self._last_indexed_page_count = 0

            # 页数回退：不满足 append-only，走全量重建兜底。
            if current_count < self._last_indexed_page_count:
                self.build(page_store)
                return

            if current_count == self._last_indexed_page_count:
                return

            # 仅处理新增切片，保证在线更新为增量。
            start_idx = self._last_indexed_page_count
            tail_pages = pages_all[start_idx:]
            existing_global_ids = {
                int((p.meta or {}).get("page_global_index", -1))
                for p in (self.pages or [])
                if int((p.meta or {}).get("page_global_index", -1)) >= 0
            }
            new_point_pages: List[Page] = []
            for offset, page in enumerate(tail_pages):
                global_idx = start_idx + offset
                pmeta = page.meta or {}
                mp_id = str(pmeta.get("memory_point_id") or "").strip()
                if not mp_id:
                    continue
                if global_idx in existing_global_ids:
                    continue
                new_point_pages.append(
                    page.model_copy(update={"meta": {**pmeta, "page_global_index": global_idx}})
                )

            self._last_indexed_page_count = current_count

            if not new_point_pages:
                self._save_meta()
                return

            if not self.pages:
                self.pages = []
            if self.index is None or self.doc_emb is None or len(self.pages) == 0:
                # 冷启动或旧索引不可用：直接用新增点页建立初始索引。
                self.pages = list(new_point_pages)
                self.doc_emb = self._encode_pages(self.pages)
                self.index = _build_faiss_index(self.doc_emb)
            else:
                new_emb = self._encode_pages(new_point_pages)
                new_emb_normalized = new_emb.copy()
                faiss.normalize_L2(new_emb_normalized)
                self.index.add(new_emb_normalized)
                self.doc_emb = np.concatenate([self.doc_emb, new_emb], axis=0)
                self.pages.extend(new_point_pages)

            temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
            temp_page_store.save(self.pages)
            np.save(self._emb_path(), self.doc_emb)
            self._save_meta()
        except Exception as e:
            print(f"[DenseRetriever] incremental update failed, fallback full rebuild: {e}")
            self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        """
        输入: 多个query
        输出: 对应多个query的检索结果 (Hit 列表)
        对于多个query，会对每个query进行独立搜索，然后按page_id聚合得分（累加），最后返回top_k结果
        """
        if self.index is None:
            # 如果还没 index（比如没调用 build/load），尝试load一下
            self.load()
            # 如果load也没成功，那 index 还是 None，就直接空返回
            if self.index is None:
                return [[] for _ in query_list]

        # 把所有 query 一起编码
        if self.use_openai_api:
            queries_emb = self._encode_via_openai(query_list)
        elif self.api_url:
            queries_emb = self._encode_via_api(query_list, encode_type="query")
        else:
            # 本地模式
            queries_emb = self.model.encode_queries(
                query_list,
                batch_size=self.config.get("batch_size", 32),
                max_length=self.config.get("max_length", 512),
            )

        # 使用自定义的 search 函数
        scores_list, indices_list = _search_faiss_index(self.index, queries_emb, top_k)

        # 按 page_id 聚合得分：如果同一个 page 被多个 query 搜索到，累加得分
        page_scores: Dict[str, float] = {}  # page_id -> 累计得分
        page_hits: Dict[str, Hit] = {}      # page_id -> Hit对象（保存第一个遇到的Hit作为代表）

        for scores, indices in zip(scores_list, indices_list):
            for rank, (idx, sc) in enumerate(zip(indices, scores)):
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(self.pages):
                    continue
                page = self.pages[idx_int]
                snippet = page.content
                pmeta = page.meta or {}
                page_id = str(pmeta.get("page_global_index", idx_int))
                score = float(sc)

                if page_id in page_scores:
                    # 累加得分
                    page_scores[page_id] += score
                else:
                    # 第一次遇到这个page，保存得分和Hit对象
                    page_scores[page_id] = score
                    page_hits[page_id] = Hit(
                        page_id=page_id,
                        snippet=snippet,
                        source="vector",
                        meta={
                            "rank": rank,
                            "score": score,
                            "memory_point_id": str(pmeta.get("memory_point_id") or "").strip(),
                            "chapter_id": str(pmeta.get("chapter_id") or "").strip(),
                            "session_id": str(pmeta.get("session_id") or "").strip(),
                            "timestamp_start": str(pmeta.get("timestamp_start") or "").strip(),
                            "timestamp_end": str(pmeta.get("timestamp_end") or "").strip(),
                            "source_turn_start": int(pmeta.get("source_turn_start") or 0),
                            "source_turn_end": int(pmeta.get("source_turn_end") or 0),
                        },
                    )

        # 按总分排序，取 top k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_pages = sorted_pages[:top_k]

        # 构建最终的hits列表（使用累加后的得分）
        final_hits: List[Hit] = []
        for rank, (page_id, total_score) in enumerate(top_k_pages):
            hit = page_hits[page_id]
            # 更新meta中的得分为累加后的总分
            updated_meta = hit.meta.copy() if hit.meta else {}
            updated_meta["rank"] = rank
            updated_meta["score"] = total_score
            final_hits.append(
                Hit(
                    page_id=hit.page_id,
                    snippet=hit.snippet,
                    source=hit.source,
                    meta=updated_meta
                )
            )

        # 返回 List[List[Hit]] 格式（只有一个列表，即聚合后的结果）
        return [final_hits]