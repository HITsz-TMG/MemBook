from __future__ import annotations

from membook.compat import env
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from membook.generator import AbsGenerator

if TYPE_CHECKING:
    from membook.agents.research_agent import ResearchAgent
from membook.memory.toc import (
    GeneralQaMode,
    MEMORY_LANE_GENERAL_QA,
    MEMORY_LANE_USER_DURABLE,
    TocBuilder,
    format_prior_sessions_toc_block,
    normalize_general_qa_mode,
)
from membook.memory.toc.chapter_splitter import normalize_memory_lane
from membook.memory.toc.dialogue_parse import dialogue_from_gam_text
from membook.memory.toc.generator_adapter import AbsGeneratorLLMClient
from membook.memory.toc.reconcile_context import (
    format_grouped_memory_points_block,
    run_reconcile_context_gate,
)
from membook.memory.toc.schemas import ConfirmedMemoryItem, SessionNode, TurnTimeRange, TurnsNode
from membook.memory.toc.toc_builder import (
    _format_turn_for_fragment_excerpt,
    _parse_time_value,
    format_prior_sessions_global_reconcile_context,
    normalized_session_pending_text,
    pending_retrieval_evidence_enabled,
    pending_retrieval_text_append,
    stable_pending_memory_point_id,
)
from membook.schemas import (
    InMemoryMemoryStore,
    InMemoryPageStore,
    MemoryState,
    MemoryStore,
    MemoryUpdate,
    Page,
    PageStore,
)


def _chapter_page_content(dialogue: List[dict], child: Any) -> str:
    parts: List[str] = []
    for sp in child.spans or []:
        for k in range(sp.start_turn, sp.end_turn):
            if 0 <= k < len(dialogue):
                parts.append(_format_turn_for_fragment_excerpt(dialogue[k]))
    return "\n".join(parts)


def _is_empty_like_summary(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return True
    patterns = (
        "does not contain any specific information",
        "does not provide new information",
        "no confirmed memory points available",
        "no specific details or context",
        "no user-confirmed durable facts",
    )
    return any(p in s for p in patterns)


def _validate_chapter_for_persist(session_id: str, child: Any) -> None:
    """Hard gate before persistence: mapping must be stable and aligned."""
    chapter_id = str(getattr(child, "chapter_id", "") or "").strip()
    if not chapter_id:
        raise ValueError(f"[MemoryAgent] missing chapter_id before persist: session={session_id}")
    confirmed = list((getattr(child, "meta", {}) or {}).get("confirmed_memories") or [])
    if confirmed and _is_empty_like_summary(str(getattr(child, "summary", "") or "")):
        raise ValueError(
            f"[MemoryAgent] chapter summary is empty-like with confirmed memories: "
            f"session={session_id}, chapter={chapter_id}, confirmed={len(confirmed)}"
        )
    for it in confirmed:
        if not isinstance(it, dict):
            continue
        item_chapter = str(it.get("chapter_id") or "").strip()
        if item_chapter and item_chapter != chapter_id:
            raise ValueError(
                f"[MemoryAgent] chapter_id mismatch before persist: "
                f"session={session_id}, chapter={chapter_id}, item_chapter={item_chapter}"
            )


def _max_persisted_chapter_index(pages: List[Page]) -> int:
    """Max ``chapter_index`` on all persisted [CHAPTER] pages (user + general); -1 if none."""
    m = -1
    for p in pages:
        meta = p.meta or {}
        if not meta.get("toc_mode"):
            continue
        if not str(p.header or "").strip().startswith("[CHAPTER]"):
            continue
        lane = normalize_memory_lane(str(meta.get("memory_lane") or ""))
        if lane not in (MEMORY_LANE_USER_DURABLE, MEMORY_LANE_GENERAL_QA):
            continue
        raw = meta.get("chapter_index")
        try:
            v = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if v > m:
            m = v
    return m


def _turns_node_with_global_chapter(child: TurnsNode, g: int) -> TurnsNode:
    """Assign global ``chap_{g}`` and sync ``meta.confirmed_memories`` dict chapter_id fields."""
    new_cid = f"chap_{g}"
    cm = list((child.meta or {}).get("confirmed_memories") or [])
    new_cm: List[Any] = []
    for it in cm:
        if isinstance(it, dict):
            d = dict(it)
            d["chapter_id"] = new_cid
            new_cm.append(d)
        else:
            new_cm.append(it)
    new_meta = {**(child.meta or {}), "confirmed_memories": new_cm}
    return child.model_copy(update={"chapter_id": new_cid, "meta": new_meta})


class MemoryAgent:
    """
    Builds structured MemBook ToC memory (Session → Chapter → Memory Point).
    ``nominal`` mode persists only ``user_durable`` chapters; ``full`` mode also persists
    ``general_qa`` chapters and memory points for retrieval.
    """

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        page_store: PageStore | None = None,
        generator: AbsGenerator | None = None,
        dir_path: Optional[str] = None,
        research_agent: Optional["ResearchAgent"] = None,
        general_qa_mode: GeneralQaMode = "nominal",
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for MemoryAgent")
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.page_store = page_store or InMemoryPageStore(dir_path=dir_path)
        self._llm = AbsGeneratorLLMClient(generator)
        self._research_agent = research_agent
        self.general_qa_mode: GeneralQaMode = normalize_general_qa_mode(general_qa_mode)
        ctx = self._make_reconcile_context_provider()
        self.toc_builder = TocBuilder(
            llm=self._llm,
            min_chapter_turns=2,
            context_provider=ctx,
            general_qa_mode=self.general_qa_mode,
        )
        self.session_counter = 0
        self._session_nodes: List[SessionNode] = []
        self._next_session_id: Optional[str] = None

    def _make_reconcile_context_provider(self):
        """
        跨 session 上下文检索（Reconcile）：一轮 Planning + Searching。

        top-k 按 score 与 QA 共用 ``ResearchAgent._sort_hits_by_score``；
        写入 Reconcile 的文本由 ``format_grouped_memory_points_block`` 按时间分层组装。

        ``MEMBOOK_RECONCILE_CROSS_SESSION_MODE=global_dump``：不做 gate/search，直接注入截断后的
        全部 prior session title+summary（需 ``ResearchAgent`` 仅用于 gated 路径；global 可在无 RA 时仍生效）。

        ``MEMBOOK_RECONCILE_CROSS_SESSION_MODE=none``：不向 reconcile 注入跨 session 证据
        （``RELATED_CONFIRMED_CONTEXT`` 恒为 ``(none)``；本 session 真值仍见 ``PRIOR_CONFIRMED_LIST_SNAPSHOT``）。
        """
        mode = (env("MEMBOOK_RECONCILE_CROSS_SESSION_MODE") or "gated").strip().lower()
        if mode in ("none", "off", "disabled", "no_cross_session"):
            def no_cross_session_provider(
                _indexed_dialogue_zone: str,
                _time_hint: str,
                _session_id: str,
                _prior_confirmed: List[ConfirmedMemoryItem],
            ) -> str:
                return "(none)"

            return no_cross_session_provider

        if mode in ("global_dump", "global", "naive", "dump_all"):
            def global_provider(
                _indexed_dialogue_zone: str,
                _time_hint: str,
                _session_id: str,
                _prior_confirmed: List[ConfirmedMemoryItem],
            ) -> str:
                block = format_prior_sessions_global_reconcile_context(self._session_nodes)
                return block if block.strip() else "(none)"

            return global_provider

        ra = self._research_agent
        if ra is None:
            return None
        llm = self._llm

        def provider(
            indexed_dialogue_zone: str,
            time_hint: str,
            session_id: str,
            prior_confirmed: List[ConfirmedMemoryItem],
        ) -> str:
            reqs = run_reconcile_context_gate(
                llm,
                indexed_dialogue_zone=indexed_dialogue_zone,
                time_hint=time_hint,
                prior_confirmed=prior_confirmed,
            )
            if not reqs:
                return "(none)"

            from membook.schemas import Result

            trimmed_reqs = [r.strip() for r in reqs if (r or "").strip()][:5]
            if not trimmed_reqs:
                return "(none)"
            joined_requests = " ".join(trimmed_reqs)

            try:
                ra._update_retrievers()  # type: ignore[attr-defined]
            except Exception:
                pass

            try:
                memory_state = ra.memory_store.load()
            except Exception:
                memory_state = None

            try:
                plan = ra._planning(joined_requests, memory_state)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[MemoryAgent] reconcile planning failed ({type(e).__name__}): {e}")
                return "(none)"

            try:
                _, hits = ra._search_collect_hits(  # type: ignore[attr-defined]
                    plan,
                    Result(content="", sources=[]),
                    joined_requests,
                    deep_mode=False,
                    top_k=10,
                )
            except Exception as e:
                print(f"[MemoryAgent] reconcile search failed ({type(e).__name__}): {e}")
                return "(none)"

            filtered = []
            for h in hits or []:
                meta = h.meta or {}
                mp_id = str(meta.get("memory_point_id") or "").strip()
                if not mp_id:
                    continue
                sid = str(meta.get("session_id") or "").strip()
                if session_id and sid == session_id:
                    continue
                filtered.append(h)

            top10 = ra._sort_hits_by_score(filtered)[:10]  # type: ignore[attr-defined]
            return format_grouped_memory_points_block(top10)

        return provider

    def set_next_session_id(self, session_id: str) -> None:
        self._next_session_id = str(session_id)

    def memorize(self, message: str) -> MemoryUpdate:
        text = (message or "").strip()
        dialogue = dialogue_from_gam_text(text)
        session_id = self._next_session_id or f"session_{self.session_counter}"
        self._next_session_id = None
        out = self.memorize_dialogue(dialogue, session_id=session_id, user_id="")
        self.session_counter += 1
        return out

    def memorize_dialogue(
        self,
        dialogue: List[Dict[str, Any]],
        session_id: str,
        user_id: str = "",
        time_span: Optional[TurnTimeRange] = None,
    ) -> MemoryUpdate:
        prior = format_prior_sessions_toc_block(self._session_nodes)
        session_node = self.toc_builder.build(
            dialogue=dialogue,
            session_id=session_id,
            user_id=user_id,
            time_span=time_span,
            prior_sessions_toc=prior,
        )
        self._session_nodes.append(session_node)

        new_pages: List[Page] = []
        chapter_pages_added = 0
        last_chapter_page: Optional[Page] = None
        session_body = str(session_node.summary or "").strip()
        session_body = (session_body + pending_retrieval_text_append(session_node.session_pending)).strip()
        session_page = Page(
            header=f"[SESSION] {session_node.title}".strip(),
            content=session_body,
            meta={
                "toc_mode": False,
                "node_level": "session",
                "session_id": session_node.session_id,
                "session_title": session_node.title,
                "session_summary": session_node.summary,
                "session_pending": session_node.session_pending,
                "session_confirmed_memory_count": int(getattr(session_node, "confirmed_memory_count", 0) or 0),
            },
        )
        self.page_store.add(session_page)
        new_pages.append(session_page)
        self.memory_store.add(
            f"{session_node.session_id}: {session_node.title} | {session_node.summary}".strip()
        )
        next_global = _max_persisted_chapter_index(self.page_store.load()) + 1
        for i, child in enumerate(session_node.children):
            lane = normalize_memory_lane((child.meta or {}).get("memory_lane"))
            if lane == MEMORY_LANE_GENERAL_QA and self.general_qa_mode != "full":
                continue
            if lane not in (MEMORY_LANE_USER_DURABLE, MEMORY_LANE_GENERAL_QA):
                continue
            g = next_global
            next_global += 1
            patched = _turns_node_with_global_chapter(child, g)
            session_node.children[i] = patched
            child = patched
            _validate_chapter_for_persist(session_node.session_id, child)
            raw = _chapter_page_content(dialogue, child)
            raw = (raw + pending_retrieval_text_append(session_node.session_pending)).strip()
            source_fragment_ids = list((child.meta or {}).get("source_fragment_ids") or [])
            source_fragment_titles = dict((child.meta or {}).get("source_fragment_titles") or {})
            source_fragment_summaries = dict((child.meta or {}).get("source_fragment_summaries") or {})
            header = f"[CHAPTER] {child.title}".strip()
            meta: Dict[str, Any] = {
                "toc_mode": True,
                "node_level": "chapter",
                "session_id": session_node.session_id,
                "session_title": session_node.title,
                "chapter_id": child.chapter_id,
                "chapter_index": g,
                "memory_lane": lane,
                "chapter_title": child.title,
                "chapter_summary": child.summary,
                "session_summary": session_node.summary,
                "session_pending": session_node.session_pending,
                "confirmed_memories": list((child.meta or {}).get("confirmed_memories") or []),
                "source_fragment_ids": source_fragment_ids,
                "source_fragment_titles": source_fragment_titles,
                "source_fragment_summaries": source_fragment_summaries,
                "session_confirmed_memory_count": int(getattr(session_node, "confirmed_memory_count", 0) or 0),
            }
            page = Page(header=header, content=raw, meta=meta)
            self.page_store.add(page)
            new_pages.append(page)
            last_chapter_page = page
            chapter_pages_added += 1
            abstract = (
                f"{session_node.session_id}/cid={g}: {child.title} | {child.summary}"
            ).strip()
            self.memory_store.add(abstract)

            # Add point-level pages for confirmed memories.
            # These point pages are used by BM25/Vector (point-first) retrieval.
            confirmed_points = list((child.meta or {}).get("confirmed_memories") or [])
            for it in confirmed_points:
                if not isinstance(it, dict):
                    continue
                mp_id = str(it.get("memory_point_id") or "").strip()
                point_text = str(it.get("text") or "").strip()
                if not mp_id or not point_text:
                    continue

                ts_start_raw = str(it.get("timestamp_start") or "").strip()
                iso_day = ""
                if ts_start_raw:
                    tv = _parse_time_value(ts_start_raw)
                    if tv.iso:
                        iso_day = str(tv.iso).split("T", 1)[0].strip()
                # Fallback: keep original timestamp_start if iso parsing fails.
                if not iso_day and ts_start_raw:
                    iso_day = ts_start_raw.split(",", 2)[0].strip() if "," in ts_start_raw else ts_start_raw[:10]

                point_content = f"{iso_day}: {point_text}".strip() if iso_day else point_text
                point_header = f"[MEMORY_POINT] {mp_id}".strip()

                point_meta: Dict[str, Any] = {
                    "toc_mode": False,  # exclude from MEMORY_CATALOG
                    "node_level": "memory_point",
                    "memory_lane": lane,
                    "memory_point_id": mp_id,
                    "chapter_id": child.chapter_id,
                    "chapter_title": child.title,
                    "chapter_summary": child.summary,
                    "source_fragment_id": str(it.get("fragment_id") or ""),
                    "source_fragment_ids": source_fragment_ids,
                    "session_id": session_node.session_id,
                    "session_title": session_node.title,
                    "session_pending": session_node.session_pending,
                    "timestamp_start": ts_start_raw,
                    "timestamp_end": str(it.get("timestamp_end") or "").strip(),
                    "source_turn_start": int(it.get("source_turn_start") or 0),
                    "source_turn_end": int(it.get("source_turn_end") or 0),
                }
                point_page = Page(header=point_header, content=point_content, meta=point_meta)
                self.page_store.add(point_page)
                new_pages.append(point_page)

        pend_plain = normalized_session_pending_text(session_node.session_pending)
        if pending_retrieval_evidence_enabled() and pend_plain.strip():
            ref_meta: Dict[str, Any] = (last_chapter_page.meta or {}) if last_chapter_page else {}
            mp_id = stable_pending_memory_point_id(session_node.session_id, pend_plain)
            chapter_id_ref = str(ref_meta.get("chapter_id") or "").strip() or f"session_{session_node.session_id}"
            chapter_idx_ref = ref_meta.get("chapter_index")
            try:
                chapter_idx_int = int(chapter_idx_ref)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                chapter_idx_int = 0
            point_content = f"[PENDING] {pend_plain}".strip()
            point_header = f"[MEMORY_POINT] {mp_id}".strip()
            point_meta = {
                "toc_mode": False,
                "node_level": "memory_point",
                "memory_lane": MEMORY_LANE_USER_DURABLE,
                "memory_point_id": mp_id,
                "chapter_id": chapter_id_ref,
                "chapter_title": str(ref_meta.get("chapter_title") or "session_pending"),
                "chapter_summary": str(ref_meta.get("chapter_summary") or ""),
                "session_id": session_node.session_id,
                "session_title": session_node.title,
                "session_pending": session_node.session_pending,
                "chapter_index": chapter_idx_int,
            }
            pend_point = Page(header=point_header, content=point_content, meta=point_meta)
            self.page_store.add(pend_point)
            new_pages.append(pend_point)

        updated_state: MemoryState = self.memory_store.load()
        new_page = last_chapter_page or (new_pages[-1] if new_pages else Page(header="[CHAPTER]", content="", meta={}))
        return MemoryUpdate(
            new_state=updated_state,
            new_page=new_page,
            debug={
                "toc_mode": True,
                "session_id": session_node.session_id,
                "session_title": session_node.title,
                "session_pending": session_node.session_pending,
                "chapter_count": chapter_pages_added,
                "point_count": len([p for p in new_pages if str(p.header).startswith("[MEMORY_POINT]")]),
            },
        )
