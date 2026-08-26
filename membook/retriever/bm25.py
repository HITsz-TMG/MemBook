import os, json, subprocess, shutil, sys, time
from typing import Dict, Any, List

try:
    from pyserini.search.lucene import LuceneSearcher
except Exception:
    LuceneSearcher = None  # type: ignore

from membook.retriever.base import AbsRetriever
from membook.schemas import InMemoryPageStore, Hit, Page


def _safe_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    """
    安全地删除目录树，带重试机制
    
    Args:
        path: 要删除的目录路径
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    if not os.path.exists(path):
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            # 确保目录真的被删除了
            if not os.path.exists(path):
                return
            time.sleep(delay)
        except OSError as e:
            if attempt == max_retries - 1:
                # 最后一次尝试仍然失败，强制删除
                try:
                    # 尝试更激进的删除方式
                    import subprocess
                    subprocess.run(['rm', '-rf', path], check=False, capture_output=True)
                    if not os.path.exists(path):
                        return
                except Exception:
                    pass
                raise OSError(f"无法删除目录 {path}: {e}")
            time.sleep(delay)


class BM25Retriever(AbsRetriever):
    """
        关键词检索器 (BM25 / Lucene)
        config 需要:
        {
            "index_dir": "xxx",   # 用来放 index/ 和 pages/
            "threads": 4
        }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if LuceneSearcher is None:
            raise ImportError("BM25Retriever requires pyserini to be installed")
        self.index_dir = self.config["index_dir"]
        self.searcher: LuceneSearcher | None = None
        self.pages: List[Page] = []
        self.last_indexed_page_count: int = 0
        self.append_count: int = 0
        self._pending_docs: List[Dict[str, Any]] = []
        self._append_rebuild_threshold = int(self.config.get("append_rebuild_threshold", 200) or 200)

    def _pages_dir(self):
        return os.path.join(self.index_dir, "pages")

    def _lucene_dir(self):
        return os.path.join(self.index_dir, "index")

    def _docs_dir(self):
        return os.path.join(self.index_dir, "documents")

    def _meta_dir(self):
        return os.path.join(self.index_dir, "meta")

    def _manifest_path(self):
        return os.path.join(self._meta_dir(), "manifest.json")

    def _pending_path(self):
        return os.path.join(self._meta_dir(), "pending_docs.jsonl")

    def _load_meta(self) -> None:
        self.last_indexed_page_count = 0
        self.append_count = 0
        self._pending_docs = []

        manifest_path = self._manifest_path()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.last_indexed_page_count = int(payload.get("last_indexed_page_count", 0) or 0)
                self.append_count = int(payload.get("append_count", 0) or 0)
            except Exception:
                self.last_indexed_page_count = 0
                self.append_count = 0

        pending_path = self._pending_path()
        if os.path.exists(pending_path):
            try:
                with open(pending_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        self._pending_docs.append(json.loads(line))
            except Exception:
                self._pending_docs = []

    def _save_meta(self) -> None:
        os.makedirs(self._meta_dir(), exist_ok=True)
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "last_indexed_page_count": int(self.last_indexed_page_count),
                    "append_count": int(self.append_count),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(self._pending_path(), "w", encoding="utf-8") as f:
            for item in self._pending_docs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self) -> None:
        # 尝试从磁盘恢复
        if not os.path.exists(self._lucene_dir()):
            raise RuntimeError("BM25 index not found, need build() first.")
        self.pages = InMemoryPageStore(dir_path=self._pages_dir()).load()
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore
        self._load_meta()

    def build(self, page_store: InMemoryPageStore) -> None:
        # 0. 首先清理所有旧的目录和文件，确保干净的状态
        # 使用安全删除函数，带重试机制
        _safe_rmtree(self._lucene_dir())
        _safe_rmtree(self._docs_dir())
        
        # 1. 创建必要的目录
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self._docs_dir(), exist_ok=True)

        # 2. dump pages -> documents.jsonl (pyserini需要 id + contents)
        pages = page_store.load()
        docs_path = os.path.join(self._docs_dir(), "documents.jsonl")
        # 严格仅索引带 memory_point_id 的 confirmed memory point 页。无点页即空索引、检索返回空。
        docs_pages = [
            (i, p)
            for i, p in enumerate(pages)
            if isinstance(p.meta, dict) and str(p.meta.get("memory_point_id") or "").strip()
        ]
        with open(docs_path, "w", encoding="utf-8") as f:
            for i, p in docs_pages:
                text = (p.content or "").strip()
                json.dump({"id": str(i), "contents": text}, f, ensure_ascii=False)
                f.write("\n")

        # 3. 确保 lucene index 目录是干净的
        os.makedirs(self._lucene_dir(), exist_ok=True)

        # 4. 调 pyserini 构建 Lucene 索引（必须用当前解释器，避免 venv 下误用系统 python 导致缺包）
        cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", self._docs_dir(),
            "--index", self._lucene_dir(),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.config.get("threads", 1)),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        
        # 添加重试机制，防止偶发的构建失败
        max_build_retries = 2
        for attempt in range(max_build_retries):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_build_retries - 1:
                    print(f"[ERROR] Pyserini 索引构建失败:")
                    print(f"  stdout: {e.stdout}")
                    print(f"  stderr: {e.stderr}")
                    raise
                print(f"[WARN] Pyserini 索引构建失败，重试 {attempt + 1}/{max_build_retries}...")
                # 清理失败的索引
                _safe_rmtree(self._lucene_dir())
                os.makedirs(self._lucene_dir(), exist_ok=True)
                time.sleep(1)

        # 5. 把 pages 也固化到磁盘，供 load() / search() 反查
        # 创建临时 PageStore 实例来保存
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(pages)
        
        # 6. 更新内存镜像
        self.pages = pages
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore
        self.last_indexed_page_count = len(pages)
        self.append_count = 0
        self._pending_docs = []
        self._save_meta()

    def update(self, page_store: InMemoryPageStore) -> None:
        try:
            if self.searcher is None:
                self.load()
        except Exception:
            # 冷启动无索引时直接全量构建。
            self.build(page_store)
            return

        try:
            pages = page_store.load()
            current_count = len(pages)
            if current_count < self.last_indexed_page_count:
                print("[BM25Retriever] detected page count rollback, rebuilding full index.")
                self.build(page_store)
                return
            if current_count == self.last_indexed_page_count:
                return

            start_idx = self.last_indexed_page_count
            new_point_docs: List[Dict[str, Any]] = []
            for idx in range(start_idx, current_count):
                page = pages[idx]
                pmeta = page.meta or {}
                mp_id = str(pmeta.get("memory_point_id") or "").strip()
                if not mp_id:
                    continue

                new_point_docs.append(
                    {
                        "page_id": idx,
                        "contents": (page.content or "").strip(),
                        "meta": {
                            "memory_point_id": mp_id,
                            "chapter_id": str(pmeta.get("chapter_id") or "").strip(),
                            "session_id": str(pmeta.get("session_id") or "").strip(),
                            "timestamp_start": str(pmeta.get("timestamp_start") or "").strip(),
                            "timestamp_end": str(pmeta.get("timestamp_end") or "").strip(),
                            "source_turn_start": int(pmeta.get("source_turn_start") or 0),
                            "source_turn_end": int(pmeta.get("source_turn_end") or 0),
                        },
                    }
                )

            self.last_indexed_page_count = current_count

            if new_point_docs:
                self._pending_docs.extend(new_point_docs)
                self.append_count += len(new_point_docs)

            # 同步主 pages 快照，保证 page_id 反查一致。
            temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
            temp_page_store.save(pages)
            self.pages = pages
            self._save_meta()

            if self.append_count >= self._append_rebuild_threshold:
                print(
                    f"[BM25Retriever] append_count={self.append_count} reached threshold "
                    f"{self._append_rebuild_threshold}, rebuilding full index."
                )
                self.build(page_store)
        except Exception as e:
            print(f"[BM25Retriever] incremental update failed, fallback full rebuild: {e}")
            self.build(page_store)

    def _search_pending_docs(self, query: str, top_k: int) -> List[Hit]:
        query_terms = [t for t in query.lower().split() if t]
        if not query_terms:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        for item in self._pending_docs:
            page_id = int(item.get("page_id", -1))
            if page_id < 0:
                continue
            text = str(item.get("contents") or "")
            text_l = text.lower()
            score = float(sum(text_l.count(term) for term in query_terms))
            if score <= 0:
                continue
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits: List[Hit] = []
        for rank, (score, item) in enumerate(scored[:top_k]):
            meta = {"rank": rank, "score": score}
            meta.update(item.get("meta") or {})
            hits.append(
                Hit(
                    page_id=str(item["page_id"]),
                    snippet=str(item.get("contents") or ""),
                    source="keyword",
                    meta=meta,
                )
            )
        return hits

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self.searcher is None:
            # 容错：如果忘了 load/build
            self.load()

        results_all: List[List[Hit]] = []
        for q in query_list:
            q = q.strip()
            if not q:
                results_all.append([])
                continue

            hits_for_q: List[Hit] = []
            py_hits = self.searcher.search(q, k=top_k)
            for rank, h in enumerate(py_hits):
                # h.docid 是字符串 id
                idx = int(h.docid)
                if idx < 0 or idx >= len(self.pages):
                    continue
                page = self.pages[idx]
                snippet = page.content
                meta = {"rank": rank, "score": float(h.score)}
                pmeta = page.meta or {}
                mp_id = str(pmeta.get("memory_point_id") or "").strip()
                if mp_id:
                    meta.update(
                        {
                            "memory_point_id": mp_id,
                            "chapter_id": str(pmeta.get("chapter_id") or "").strip(),
                            "session_id": str(pmeta.get("session_id") or "").strip(),
                            "timestamp_start": str(pmeta.get("timestamp_start") or "").strip(),
                            "timestamp_end": str(pmeta.get("timestamp_end") or "").strip(),
                            "source_turn_start": int(pmeta.get("source_turn_start") or 0),
                            "source_turn_end": int(pmeta.get("source_turn_end") or 0),
                        }
                    )
                hits_for_q.append(
                    Hit(
                        page_id=str(idx),
                        snippet=snippet,
                        source="keyword",
                        meta=meta,
                    )
                )
            pending_hits = self._search_pending_docs(q, top_k=top_k)

            merged: Dict[str, Hit] = {}
            for hit in hits_for_q + pending_hits:
                key = str(hit.page_id)
                if key not in merged:
                    merged[key] = hit
                    continue
                prev = merged[key]
                prev_score = float((prev.meta or {}).get("score", 0.0))
                curr_score = float((hit.meta or {}).get("score", 0.0))
                if curr_score > prev_score:
                    merged[key] = hit
            merged_hits = sorted(
                merged.values(),
                key=lambda h: float((h.meta or {}).get("score", 0.0)),
                reverse=True,
            )[:top_k]
            for rank, hit in enumerate(merged_hits):
                meta = hit.meta or {}
                meta["rank"] = rank
                hit.meta = meta
            results_all.append(merged_hits)
        return results_all
