# -*- coding: utf-8 -*-
"""ToC 树构建器"""

from __future__ import annotations

from membook.compat import env
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import re
import time
import os
import traceback

from .schemas import (
    ConfirmedMemoryItem,
    SessionNode,
    TurnsNode,
    TurnTimeRange,
    UserNode,
    TocTree,
    TimeValue,
)
from .schemas import TurnSpan
from .chapter_splitter import (
    MEMORY_LANE_GENERAL_QA,
    MEMORY_LANE_INTERACTION_META,
    MEMORY_LANE_USER_DURABLE,
    _format_turn,
    normalize_memory_lane,
)
from .json_llm import LLMClient, parse_json_object
from .policy import GeneralQaMode, should_skip_non_durable_title_summary
from .progress_log import log_llm_phase, log_progress, log_progress_always
from membook.prompts.memory_prompts import (
    CLASSIFY_FRAGMENT_LANE,
    MERGE_FRAGMENTS_TO_CHAPTERS,
    DETECT_TOPIC_SHIFTS,
    GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA,
    GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META,
    GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE,
    GENERATE_FRAGMENT_TITLE_AND_SUMMARY_DURABLE,
    GENERATE_FRAGMENT_TITLE_AND_SUMMARY_INTERACTION_META,
    GENERATE_FRAGMENT_TITLE_AND_SUMMARY_QA,
    GENERATE_SESSION_SUMMARY_AND_TITLE_FINAL,
    EXTRACT_GENERAL_QA_INCREMENTAL,
    RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL,
)

# (indexed_user_durable_dialogue_zone, time_hint, session_id, prior_confirmed) -> cross-session context block or "(none)"
ReconcileContextProvider = Callable[[str, str, str, List[ConfirmedMemoryItem]], str]


class TocBuilder:
    """从 HaluMem session / dialogue 构建 ToC 树（LLM 话题切换 + LLM 抽取）"""

    def __init__(
        self,
        llm: LLMClient,
        min_chapter_turns: int = 2,
        title_fn: Optional[Callable[[str], str]] = None,
        *,
        context_provider: Optional[ReconcileContextProvider] = None,
        general_qa_mode: GeneralQaMode = "nominal",
    ):
        self.llm = llm
        _ = min_chapter_turns  # 新切分阶段暂不使用；后续若需要可加回约束
        self.title_fn = title_fn
        self.context_provider = context_provider
        self.general_qa_mode: GeneralQaMode = general_qa_mode

    def build(
        self,
        dialogue: List[dict],
        session_id: str,
        user_id: str = "",
        time_span: Optional[TurnTimeRange] = None,
        prior_sessions_toc: Optional[str] = None,
    ) -> SessionNode:
        build_start = time.perf_counter()
        log_progress(f"Build start: session_id={session_id!r}, dialogue_turns={len(dialogue)}")
        prior_block = prior_sessions_toc if prior_sessions_toc is not None else "(none)"
        children, pre_pend, _, pre_confirmed, pre_general_confirmed = _build_fragment_children_topic_shifts(
            self.llm,
            dialogue,
            session_id=session_id,
            title_fn=self.title_fn,
            prior_sessions_toc=prior_block,
            context_provider=self.context_provider,
            general_qa_mode=self.general_qa_mode,
        )
        # Session 级收尾与 MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS（如 15）无关：每个 session 构建结束必做。
        log_progress(
            f"Session finalize: audit + chronicle + optional general_qa merge "
            f"(fragments={len(children)}, turns={len(dialogue)})"
        )
        # pre_pend is None ⇒ 未在滑窗内跑 session reconcile（非 Live/短会话/回退），在单窗 [(0,n)] 补跑完整审计 + 末窗 summary+title。
        if pre_pend is None:
            (
                session_pending_str,
                session_summary,
                session_title,
                session_confirmed_list,
                session_general_confirmed_list,
            ) = run_session_audit_and_chronicle(
                self.llm,
                dialogue,
                children,
                [(0, len(dialogue))],
                prior_block,
                time_span,
                session_id=session_id,
                context_provider=self.context_provider,
                general_qa_mode=self.general_qa_mode,
            )
        else:
            session_pending_str = (pre_pend or "").strip()
            session_confirmed_list = list(pre_confirmed or [])
            session_general_confirmed_list = list(pre_general_confirmed or [])
            hint = format_session_time_hint(time_span, dialogue)
            non_g = _non_general_children(children)
            fb = (non_g[0].title if non_g else None) or session_id
            session_summary, session_title = _llm_session_summary_and_title_final(
                self.llm,
                prior_block,
                session_pending_str,
                session_confirmed_list,
                children,
                hint,
                fallback_title=str(fb),
            )
        children, session_confirmed_list, durable_merge_groups = _consolidate_user_durable_session(
            self.llm,
            dialogue,
            children,
            session_confirmed_list,
            session_id=session_id,
            previous_toc_block=prior_block,
        )
        general_merge_groups: List[List[int]] = []
        interaction_meta_nodes = [
            c
            for c in children
            if normalize_memory_lane((c.meta or {}).get("memory_lane")) == MEMORY_LANE_INTERACTION_META
        ]
        if len(interaction_meta_nodes) > 1:
            merged_meta = _build_merged_interaction_meta_turns_node(
                self.llm,
                interaction_meta_nodes,
                dialogue,
                session_id,
                use_llm=False,
            )
            children = _replace_lane_fragments_with_merged(
                children, MEMORY_LANE_INTERACTION_META, merged_meta
            )
        general_nodes = [
            c
            for c in children
            if normalize_memory_lane((c.meta or {}).get("memory_lane")) == MEMORY_LANE_GENERAL_QA
        ]
        if self.general_qa_mode == "full":
            if len(general_nodes) > 1:
                children, session_general_confirmed_list, general_merge_groups = _consolidate_general_qa_session(
                    self.llm,
                    dialogue,
                    children,
                    session_general_confirmed_list,
                    session_id=session_id,
                    previous_toc_block=prior_block,
                )
            _refresh_general_qa_children_summary_from_confirmed(
                self.llm,
                dialogue,
                children,
                previous_toc_block=prior_block,
            )
        elif len(general_nodes) > 1:
            merged_general = _build_merged_general_turns_node(
                self.llm,
                general_nodes,
                dialogue,
                session_id,
                use_llm=False,
            )
            children = _replace_lane_fragments_with_merged(
                children, MEMORY_LANE_GENERAL_QA, merged_general
            )
        children = _reorder_children_by_lane(children)
        session_confirmed_union = _merge_confirmed_lists(
            list(session_confirmed_list),
            list(session_general_confirmed_list),
        )
        children, session_confirmed_union, id_mappings = _assign_short_ids_and_mappings(
            children,
            session_confirmed_union,
            session_id,
        )
        _repartition_confirmed_to_children(children, session_confirmed_union)
        session_confirmed_list = [
            m for m in session_confirmed_union
            if _memory_item_lane(m, children) == MEMORY_LANE_USER_DURABLE
        ]
        session_general_confirmed_list = [
            m for m in session_confirmed_union
            if _memory_item_lane(m, children) == MEMORY_LANE_GENERAL_QA
        ]
        # Final chapter text must be generated after final id mapping/repartition
        # so summary/title stays consistent with persisted confirmed memories.
        _refresh_durable_children_summary_from_confirmed(
            self.llm,
            dialogue,
            children,
            previous_toc_block=prior_block,
        )
        total_processing_ms = int((time.perf_counter() - build_start) * 1000)

        log_progress(
            f"Build done: session_id={session_id!r}, fragments={len(children)} "
            f"(lane + title + summary + user-track session metadata + optional merged general_qa), "
            f"total_processing_ms={total_processing_ms}"
        )
        session_meta: Dict[str, Any] = {}
        if durable_merge_groups:
            session_meta["user_durable_merge_groups"] = durable_merge_groups
        if general_merge_groups:
            session_meta["general_qa_merge_groups"] = general_merge_groups
        session_meta.update(id_mappings)
        return SessionNode(
            session_id=session_id,
            user_id=user_id,
            title=session_title,
            total_turns=len(dialogue),
            total_processing_ms=total_processing_ms,
            time_span=time_span,
            summary=session_summary,
            session_pending=session_pending_str,
            confirmed_list=session_confirmed_list,
            general_confirmed_list=session_general_confirmed_list,
            confirmed_memory_count=len(session_confirmed_list),
            children=children,
            meta=session_meta,
        )

    def build_from_halumem_session(
        self,
        session: Dict[str, Any],
        session_id: str,
        user_id: str = "",
        prior_sessions_toc: Optional[str] = None,
    ) -> SessionNode:
        dialogue = session.get("dialogue") or []
        start_time = str(session.get("start_time") or "")
        end_time = str(session.get("end_time") or "")
        time_span = (
            TurnTimeRange(start=_parse_time_value(start_time), end=_parse_time_value(end_time))
            if (start_time or end_time)
            else None
        )
        return self.build(
            dialogue=dialogue,
            session_id=session_id,
            user_id=user_id,
            time_span=time_span,
            prior_sessions_toc=prior_sessions_toc,
        )

    def build_user_tree_from_halumem_user(
        self,
        user_obj: Dict[str, Any],
        max_sessions: Optional[int] = None,
    ) -> TocTree:
        uuid = str(user_obj.get("uuid") or "")
        user_id = uuid
        sessions = user_obj.get("sessions") or []
        if max_sessions is not None:
            sessions = sessions[:max_sessions]

        session_nodes: List[SessionNode] = []
        n_sess = len(sessions)
        for i, session in enumerate(sessions):
            log_progress(f"User tree: session {i + 1}/{n_sess} (session_id={uuid}_session_{i})")
            prior_block = format_prior_sessions_toc_block(session_nodes)
            session_nodes.append(
                self.build_from_halumem_session(
                    session=session,
                    session_id=f"{uuid}_session_{i}",
                    user_id=user_id,
                    prior_sessions_toc=prior_block,
                )
            )

        user_node = UserNode(
            user_id=user_id,
            uuid=uuid,
            persona_info=str(user_obj.get("persona_info") or ""),
            sessions=session_nodes,
        )
        return TocTree(users=[user_node])


def _indexed_dialogue(dialogue: List[dict]) -> str:
    lines: List[str] = []
    for i, turn in enumerate(dialogue):
        lines.append(f"{i}. {_format_turn(turn)}")
    return "\n".join(lines)


_ASSISTANT_SHIFT_ELLIPSIS = " ...[truncated]... "

# Assistant 进 LLM 提示：超过 max 则保留 head + ellipsis + tail（DETECT / CLASSIFY / fragment excerpt 共用）
ASSISTANT_DIALOGUE_PROMPT_MAX_CHARS = 400
ASSISTANT_DIALOGUE_PROMPT_HEAD_CHARS = 200
ASSISTANT_DIALOGUE_PROMPT_TAIL_CHARS = 200


def _ellipsize_assistant_for_shift(content: str, *, max_chars: int, head: int, tail: int) -> str:
    """超过 max_chars 时保留头尾；max_chars<=0 表示不截断。"""
    if max_chars <= 0:
        return content
    n = len(content)
    if n <= max_chars:
        return content
    h, t = max(0, head), max(0, tail)
    if h + t >= n:
        return content
    return content[:h] + _ASSISTANT_SHIFT_ELLIPSIS + content[-t:]


def _ellipsize_assistant_for_dialogue_prompt(content: str) -> str:
    """DETECT_TOPIC_SHIFTS / CLASSIFY_FRAGMENT_LANE 等共用：与 INDEXED_DIALOGUE 一致。"""
    return _ellipsize_assistant_for_shift(
        content,
        max_chars=ASSISTANT_DIALOGUE_PROMPT_MAX_CHARS,
        head=ASSISTANT_DIALOGUE_PROMPT_HEAD_CHARS,
        tail=ASSISTANT_DIALOGUE_PROMPT_TAIL_CHARS,
    )


def _format_turn_for_fragment_excerpt(turn: dict) -> str:
    """与 `_format_turn` 类似；assistant 截断规则同 `_indexed_dialogue_for_shift_detection`。"""
    role = str(turn.get("role", "unknown"))
    content = str(turn.get("content", "") or "")
    ts = str(turn.get("timestamp", "") or "")
    if role.lower() == "assistant":
        content = _ellipsize_assistant_for_dialogue_prompt(content)
    return f"[{ts}]{role}: {content}" if ts else f"{role}: {content}"


def _format_turn_for_lane_prompt(turn: dict) -> str:
    """单条 turn 的展示串，与话题检测用 `_indexed_dialogue_for_shift_detection` 中 assistant 截断规则一致。"""
    role = str(turn.get("role", "unknown"))
    content = str(turn.get("content", "") or "")
    ts = str(turn.get("timestamp", "") or "")
    if role.lower() == "assistant":
        content = _ellipsize_assistant_for_dialogue_prompt(content)
    return f"[{ts}]{role}: {content}" if ts else f"{role}: {content}"


def _strict_topic_shifts_enabled() -> bool:
    """若 MEMBOOK_STRICT_TOPIC_SHIFTS=1，探测失败或切分非法时不走每 5 轮硬切兜底，直接报错。"""
    return env("MEMBOOK_STRICT_TOPIC_SHIFTS", "0").strip() == "1"


def _truthy_env(name: str) -> bool:
    return (env(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _membook_skip_lane_classify() -> bool:
    """MEMBOOK_SKIP_LANE_CLASSIFY=1：不调 LLM lane，fragments 一律 user_durable（Flat Indexing 消融）。"""
    return _truthy_env("MEMBOOK_SKIP_LANE_CLASSIFY")


def _membook_fragment_mode() -> str:
    """
    MEMBOOK_FRAGMENT_MODE (default topic_shift):

    - topic_shift — LLM topic-shift 切分（与动态滑窗/短会话路径既有逻辑一致）。
    - fixed_turns — 固定 ``MEMBOOK_FIXED_TURNS_CHUNK`` turns 切段，不做话题探测。
    """
    raw = (env("MEMBOOK_FRAGMENT_MODE") or "topic_shift").strip().lower().replace("-", "_")
    if raw in ("fixed_turns", "fixed", "fixed_window"):
        return "fixed_turns"
    return "topic_shift"


def _membook_fixed_turn_chunk_size() -> int:
    raw = env("MEMBOOK_FIXED_TURNS_CHUNK", "20").strip() or "20"
    try:
        chunk = int(raw)
    except ValueError:
        chunk = 20
    return max(1, chunk)


def _build_fixed_turn_spans(n_turns: int, chunk: int) -> List[TurnSpan]:
    spans: List[TurnSpan] = []
    i = 0
    while i < n_turns:
        end = min(n_turns, i + chunk)
        if end > i:
            spans.append(TurnSpan(start_turn=i, end_turn=end))
        i = end
    return spans


def pending_retrieval_evidence_enabled() -> bool:
    """MEMBOOK_PENDING_IN_RETRIEVAL=1：pending 写入可检索页与 QA page_index snippet（评测与索引一致）。"""
    return _truthy_env("MEMBOOK_PENDING_IN_RETRIEVAL")


def normalized_session_pending_text(raw: Any) -> str:
    """Normalize ``session_pending`` (str or legacy list of str) → single paragraph."""
    if raw is None:
        return ""
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        return " ".join(parts) if parts else ""
    return " ".join(str(raw).strip().split())


def pending_retrieval_text_append(session_pending: Any) -> str:
    """Extra text fused into indexed Page.content when ``pending_retrieval_evidence_enabled()``."""
    if not pending_retrieval_evidence_enabled():
        return ""
    pend = normalized_session_pending_text(session_pending)
    if not pend:
        return ""
    return "\n\n[SESSION_PENDING_FOR_RETRIEVAL]\n" + pend


def scoped_memory_point_id(session_id: str, mp_idx: int) -> str:
    """Globally unique memory point id: session-scoped short index (mp_0, mp_1, ...)."""
    sid = str(session_id or "").strip()
    return f"{sid}/mp_{mp_idx}"


def stable_pending_memory_point_id(session_id: str, pending_text: str) -> str:
    """Deterministic pseudo id so BM25/Chap-ID/evidence formatting stay stable."""
    h = hashlib.sha256(f"{session_id}\0{pending_text}".encode("utf-8")).hexdigest()
    return f"pending_sess_{h[:24]}"


def _topic_shifts_json_object_enabled() -> bool:
    """为 topic-shift 启用 ``response_format: json_object``（经 ``AbsGeneratorLLMClient``）。MEMBOOK_TOPIC_SHIFTS_JSON_OBJECT=0/false/no/off 关闭。"""
    v = env("MEMBOOK_TOPIC_SHIFTS_JSON_OBJECT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _toc_json_object_enabled() -> bool:
    """ToC 内其它依赖 ``parse_json_object`` 的 LLM 步（lane、fragment T+S、pending、session summary/title、merge）。MEMBOOK_TOC_JSON_OBJECT=0/false/no/off 关闭。"""
    v = env("MEMBOOK_TOC_JSON_OBJECT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _detect_topic_shifts_in_segment(
    llm: LLMClient,
    dialogue_seg: List[dict],
    previous_toc_block: str = "(none)",
) -> Tuple[List[int], bool]:
    """
    对一段对话做 topic-shift 探测；返回 (相对 dialogue_seg 的 start_indices, parse_ok)。
    previous_toc_block 注入 DETECT_TOPIC_SHIFTS，供动态滑窗语义继承。
    """
    n_seg = len(dialogue_seg)
    if n_seg <= 0:
        return [0], True

    prompt = DETECT_TOPIC_SHIFTS.format(
        indexed_dialogue=_indexed_dialogue_for_shift_detection(dialogue_seg),
        previous_toc=previous_toc_block,
    )
    log_llm_phase("Detecting topic shifts (DETECT_TOPIC_SHIFTS)", prompt)

    text = ""
    raw: Any = []
    parse_ok = False
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            text = (
                llm.complete(
                    prompt,
                    response_json_object=_topic_shifts_json_object_enabled(),
                )
                or ""
            )
            obj = parse_json_object(text)
            raw = obj.get("start_indices")
            parse_ok = isinstance(raw, list)
            break
        except Exception as e:
            raw = []
            if attempt >= max_retries:
                log_progress(
                    f"Topic-shift detection failed after {max_retries + 1} attempt(s): {type(e).__name__}; "
                    f"will fall back to fixed 5-turn windows if needed."
                )
                break
            sleep_sec = 0.8 * (2 ** attempt)
            log_progress(
                f"Topic-shift detection retry {attempt + 1}/{max_retries} ({type(e).__name__}), sleeping {sleep_sec:.1f}s..."
            )
            time.sleep(sleep_sec)

    idxs: List[int] = []
    if isinstance(raw, list):
        for x in raw:
            try:
                idxs.append(int(x))
            except Exception:
                continue
    else:
        for m in re.findall(r"\d+", text or ""):
            try:
                idxs.append(int(m))
            except Exception:
                pass

    idxs = sorted({i for i in idxs if 0 <= i < n_seg})
    if 0 not in idxs:
        idxs = [0] + idxs
    if not idxs:
        idxs = [0]
    return idxs, parse_ok


def _indexed_dialogue_for_shift_detection(
    dialogue: List[dict],
    *,
    assistant_max_chars: int = ASSISTANT_DIALOGUE_PROMPT_MAX_CHARS,
    assistant_head_chars: int = ASSISTANT_DIALOGUE_PROMPT_HEAD_CHARS,
    assistant_tail_chars: int = ASSISTANT_DIALOGUE_PROMPT_TAIL_CHARS,
) -> str:
    """话题切换检测：超长 assistant 用头尾省略，降低噪声与 token 压力。"""
    lines: List[str] = []
    for i, turn in enumerate(dialogue):
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", "") or "")
        ts = str(turn.get("timestamp", "") or "")
        if role.lower() == "assistant":
            content = _ellipsize_assistant_for_shift(
                content,
                max_chars=assistant_max_chars,
                head=assistant_head_chars,
                tail=assistant_tail_chars,
            )
        formatted = f"[{ts}]{role}: {content}" if ts else f"{role}: {content}"
        lines.append(f"{i}. {formatted}")
    return "\n".join(lines)


def _detect_topic_shift_start_indices(
    llm: LLMClient,
    dialogue: List[dict],
    *,
    previous_toc_block: str = "(none)",
) -> List[int]:
    """
    非动态路径（回退滑窗 / 短会话单次检测）的切分入口。

    `previous_toc_block` 由调用方显式提供当前累计 ToC；会话起点（尚未生成任何 fragment）
    时为 "(none)" 合法。参数需显式传入，保证"构建任意长度下都有状态"的契约。
    """
    n = len(dialogue)
    if n <= 0:
        return [0]
    # 默认模式下，长 session 使用滑动窗口探测 + 合并，避免一次性探测 prompt 过长导致切点不稳。
    # 默认值与 tests/test_membook/toc_builder.py 保持一致。
    sliding_enable_min_turns = int(env("MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS", "15").strip() or "15")
    sliding_window_size = int(env("MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE", "20").strip() or "20")
    sliding_step = int(env("MEMBOOK_SHIFTS_SLIDING_STEP", "15").strip() or "15")

    if n >= sliding_enable_min_turns and sliding_window_size > 0 and sliding_step > 0:
        log_progress(
            f"Topic-shift detection: using sliding windows (window={sliding_window_size}, step={sliding_step})."
        )
        idxs_set: set[int] = set()
        parse_ok_any = False

        # ws 拥有其窗口内的“左半区”(ws .. ws+step)，重叠部分留给下一个窗口决定。
        for ws in range(0, n, sliding_step):
            window_end = min(n, ws + sliding_window_size)
            window_dialogue = dialogue[ws:window_end]
            if not window_dialogue:
                break

            local_idxs, parse_ok_window = _detect_topic_shifts_in_segment(
                llm, window_dialogue, previous_toc_block
            )
            if parse_ok_window:
                parse_ok_any = True

            keep_upper_exclusive = min(n, ws + sliding_step)
            for i in local_idxs:
                gi = ws + i
                if gi < keep_upper_exclusive:
                    idxs_set.add(gi)

        idxs = sorted(idxs_set)
        if 0 not in idxs:
            idxs = [0] + idxs
        if not idxs:
            idxs = [0]

        parse_ok = parse_ok_any
    else:
        idxs, parse_ok = _detect_topic_shifts_in_segment(llm, dialogue, previous_toc_block)

    # 过滤越界 + 排序去重（防御性）
    idxs = sorted({i for i in idxs if 0 <= i < n})
    if not idxs:
        idxs = [0]
    if 0 not in idxs:
        idxs = [0] + idxs

    # A valid single start index means the model intentionally kept one chapter.
    # Use fixed-size fallback only when the response could not be parsed.
    if n > 5 and not parse_ok:
        if _strict_topic_shifts_enabled():
            raise RuntimeError(
                "Topic-shift detection failed while MEMBOOK_STRICT_TOPIC_SHIFTS=1 "
                f"(parse_ok={parse_ok}, n={n}, len(start_indices)={len(idxs)}). "
                "Unset MEMBOOK_STRICT_TOPIC_SHIFTS or fix LLM JSON for DETECT_TOPIC_SHIFTS."
            )
        log_progress(
            "Topic-shift detection: parse failed; using fixed 5-turn segmentation."
        )
        idxs = list(range(0, n, 5))
    else:
        log_progress(f"Topic-shift detection succeeded: {len(idxs)} start index(es).")

    return idxs


def _classify_fragment_lane(
    llm: LLMClient,
    raw_text: str,
    *,
    previous_toc_block: str = "(none)",
) -> str:
    """
    片段 lane 分类；失败时最多重试 5 次（首次 + 4 次重试），尽量避免静默回退。

    重试 LLM 调用 / JSON 解析失败；重试仍失败后，**默认 fallback 为 `user_durable`**
    （与 CLASSIFY_FRAGMENT_LANE 的优先级一致），不回落到 `general_qa`。
    """
    if _membook_skip_lane_classify():
        return MEMORY_LANE_USER_DURABLE
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return MEMORY_LANE_GENERAL_QA
    prompt = CLASSIFY_FRAGMENT_LANE.format(
        text=raw_text,
        previous_toc=previous_toc_block if (previous_toc_block or "").strip() else "(none)",
    )
    log_llm_phase("Classifying fragment memory lane (CLASSIFY_FRAGMENT_LANE)", prompt)

    max_attempts = 5
    lane: Any = None
    last_err_kind = ""
    for attempt in range(max_attempts):
        try:
            obj = parse_json_object(
                llm.complete(prompt, response_json_object=_toc_json_object_enabled())
            )
            lane = obj.get("memory_lane")
            break
        except Exception as e:
            lane = None
            last_err_kind = type(e).__name__
            if attempt >= max_attempts - 1:
                log_progress(
                    f"Lane classification failed after {max_attempts} attempt(s): {last_err_kind}; "
                    f"defaulting to '{MEMORY_LANE_USER_DURABLE}'."
                )
                break
            sleep_sec = 0.8 * (2 ** attempt)
            log_progress(
                f"Lane classification retry {attempt + 1}/{max_attempts - 1} ({last_err_kind}), sleeping {sleep_sec:.1f}s..."
            )
            time.sleep(sleep_sec)

    lane_norm = str(lane or "").strip().lower().replace("-", "_")
    if lane_norm in ("user_durable", "general_qa", "interaction_meta"):
        return lane_norm
    # 解析结果缺失 / 非法枚举 → 优先兜底到 user_durable，保留可召回语义。
    return MEMORY_LANE_USER_DURABLE


def _build_fragments_live_dynamic_sliding(
    llm: LLMClient,
    dialogue: List[dict],
    *,
    session_id: str,
    title_fn: Optional[Callable[[str], str]] = None,
    prior_sessions_toc: str = "(none)",
    context_provider: Optional[ReconcileContextProvider] = None,
    general_qa_mode: GeneralQaMode = "nominal",
) -> Tuple[List[TurnsNode], Optional[str], Optional[str], Optional[List[ConfirmedMemoryItem]], Optional[List[ConfirmedMemoryItem]]]:
    """
    长会话：动态起点滑窗 + 交织 lane/title/summary + 每窗 Session reconcile（pending/confirmed）。
    每轮在 [ws, ws+window_size) 上 DETECT（注入 previous_toc），在推进区 (ws, ws+step] 内取最后一切点作为 ws_next；
    无新切点则 ws_next = min(ws+step, n)；再对 [ws, ws_next) 内子切分做 T+S。
    返回 (children, session_pending, None, confirmed_list)；第三元恒为 None（session summary+title 仅在 build 末窗一次 LLM）。
    """
    n = len(dialogue)
    sliding_window_size = int(env("MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE", "20").strip() or "20")
    sliding_step = int(env("MEMBOOK_SHIFTS_SLIDING_STEP", "15").strip() or "15")
    if sliding_window_size <= 0 or sliding_step <= 0:
        # 动态滑窗配置关闭的退化路径；此处尚未生成任何 fragment，previous_toc 显式为 "(none)"。
        start_indices = _detect_topic_shift_start_indices(llm, dialogue, previous_toc_block="(none)")
        cuts = list(start_indices) + [n]
        spans: List[TurnSpan] = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if 0 <= a < b <= n:
                spans.append(TurnSpan(start_turn=a, end_turn=b))
        ch = _turn_spans_to_fragment_nodes(
            llm,
            dialogue,
            spans,
            session_id=session_id,
            title_fn=title_fn,
            general_qa_mode=general_qa_mode,
        )
        return (ch, None, None, None, None)

    log_progress_always(
        f"Topic-shift detection: dynamic sliding windows (window={sliding_window_size}, "
        f"step={sliding_step}, overlap={sliding_window_size - sliding_step}, interleaved title+summary + session audit)."
    )

    previous_windows_toc_lines: List[str] = []
    children: List[TurnsNode] = []
    frag_idx = 0
    ws = 0
    prev_pending: str = ""
    prev_confirmed: List[ConfirmedMemoryItem] = []
    prev_general_confirmed: List[ConfirmedMemoryItem] = []

    while ws < n:
        window_end = min(n, ws + sliding_window_size)
        window_dialogue = dialogue[ws:window_end]
        if not window_dialogue:
            break

        previous_window_toc_block = (
            "\n".join(previous_windows_toc_lines) if previous_windows_toc_lines else "(none)"
        )
        local_idxs, _parse_ok = _detect_topic_shifts_in_segment(
            llm, window_dialogue, previous_window_toc_block
        )

        n_seg = len(window_dialogue)
        global_idxs = sorted({ws + li for li in local_idxs if 0 <= li < n_seg})

        step_end = min(ws + sliding_step, n)
        cuts_in_zone = [c for c in global_idxs if ws < c <= step_end]
        if cuts_in_zone:
            ws_next = max(cuts_in_zone)
        else:
            ws_next = min(ws + sliding_step, n)

        if ws_next <= ws:
            ws_next = min(ws + sliding_step, n)
        if ws_next <= ws:
            break
        ws_next = min(ws_next, n)

        mids = sorted([c for c in global_idxs if ws < c < ws_next])
        span_starts = [ws] + mids
        span_ends = mids + [ws_next]

        window_specs: List[Tuple[int, int, int]] = []
        for st, en in zip(span_starts, span_ends):
            if not (0 <= st < en <= n):
                continue
            window_specs.append((st, en, frag_idx))
            frag_idx += 1

        def _classify_and_build_meta(spec: Tuple[int, int, int]) -> Dict[str, Any]:
            st, en, frag_no = spec
            log_progress(f"Fragment {frag_no + 1}: turns [{st}, {en}) (dynamic sliding)")
            raw_text_local = "\n".join(_format_turn_for_fragment_excerpt(dialogue[k]) for k in range(st, en))
            lane_local = _classify_fragment_lane(
                llm, raw_text_local, previous_toc_block=previous_window_toc_block
            )
            lane_norm_local = normalize_memory_lane(lane_local)
            title_local, summary_local = "", ""
            if not should_skip_non_durable_title_summary(lane_norm_local, general_qa_mode):
                title_local, summary_local = _llm_fragment_title_and_summary(
                    llm,
                    raw_text_local,
                    lane_norm_local,
                    previous_toc_block=previous_window_toc_block,
                )
            if title_fn is not None:
                title_local = title_fn(raw_text_local)
            if not (title_local or "").strip():
                title_local = f"Fragment_{frag_no}"
            return {
                "st": st,
                "en": en,
                "frag_no": frag_no,
                "raw_text": raw_text_local,
                "lane_norm": lane_norm_local,
                "title": title_local,
                "summary": summary_local,
            }

        if _toc_parallel_enabled() and len(window_specs) > 1:
            workers = min(_toc_parallel_workers(), len(window_specs))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                built_entries = list(ex.map(_classify_and_build_meta, window_specs))
        else:
            built_entries = [_classify_and_build_meta(spec) for spec in window_specs]

        window_child_indices: List[int] = []
        window_raw_text_by_chapter: Dict[str, str] = {}
        for e in built_entries:
            st = int(e["st"])
            en = int(e["en"])
            frag_no = int(e["frag_no"])
            lane_norm = str(e["lane_norm"])
            title = str(e["title"] or "")
            summary = str(e["summary"] or "")
            fragment_id = f"frag_{frag_no}"
            start_ts = (dialogue[st].get("timestamp") or "").strip() if st < n else ""
            end_ts = (dialogue[en - 1].get("timestamp") or "").strip() if (en - 1) < n else ""
            children.append(
                TurnsNode(
                    chapter_id=fragment_id,
                    fragment_id=fragment_id,
                    title=title,
                    confirmed_only=[],
                    pending="",
                    turn_time_range=TurnTimeRange(start=_parse_time_value(start_ts), end=_parse_time_value(end_ts))
                    if start_ts and end_ts
                    else None,
                    spans=[TurnSpan(start_turn=st, end_turn=en)],
                    summary=summary,
                    meta={"start_turn": st, "end_turn": en, "memory_lane": lane_norm},
                )
            )
            window_child_indices.append(len(children) - 1)
            window_raw_text_by_chapter[fragment_id] = str(e["raw_text"] or "")

        prev_pending, prev_confirmed, prev_general_confirmed = _run_one_audit_window(
            llm,
            dialogue,
            children,
            ws,
            ws_next,
            prev_pending,
            prev_confirmed,
            prev_general_confirmed,
            session_id=session_id,
            context_provider=context_provider,
            general_qa_mode=general_qa_mode,
        )

        durable_specs: List[Tuple[int, str, List[Dict[str, Any]], str, str]] = []
        for idx in window_child_indices:
            child = children[idx]
            lane = normalize_memory_lane((child.meta or {}).get("memory_lane"))
            raw_text = window_raw_text_by_chapter.get(child.fragment_id, "")
            confirmed_memories = list((child.meta or {}).get("confirmed_memories") or [])
            if lane == MEMORY_LANE_USER_DURABLE:
                durable_specs.append((idx, raw_text, confirmed_memories, child.title, child.fragment_id))

        def _build_durable_title_summary(
            spec: Tuple[int, str, List[Dict[str, Any]], str, str]
        ) -> Tuple[int, str, str]:
            idx_local, raw_text_local, confirmed_local, old_title, chapter_id_local = spec
            title_local, summary_local = _llm_fragment_title_and_summary_from_confirmed(
                llm,
                confirmed_memories=confirmed_local,
                previous_toc_block=previous_window_toc_block,
            )
            if title_fn is not None:
                title_local = title_fn(raw_text_local)
            if not (title_local or "").strip():
                title_local = old_title or chapter_id_local
            return idx_local, title_local, summary_local

        if _toc_parallel_enabled() and len(durable_specs) > 1:
            workers = min(_toc_parallel_workers(), len(durable_specs))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                durable_results = list(ex.map(_build_durable_title_summary, durable_specs))
        else:
            durable_results = [_build_durable_title_summary(spec) for spec in durable_specs]

        for idx, title, summary in durable_results:
            children[idx] = children[idx].model_copy(update={"title": title, "summary": summary})

        for idx in window_child_indices:
            c = children[idx]
            if normalize_memory_lane((c.meta or {}).get("memory_lane")) != MEMORY_LANE_USER_DURABLE:
                continue
            title = (c.title or "").strip() or f"Fragment_{idx}"
            summary = (c.summary or "").strip() or "(none)"
            previous_windows_toc_lines.append(f"[{idx}] {title}  - {summary}")

        ws = ws_next

    return (children, prev_pending, None, prev_confirmed, prev_general_confirmed)


def _turn_spans_to_fragment_nodes(
    llm: LLMClient,
    dialogue: List[dict],
    spans: List[TurnSpan],
    *,
    session_id: str,
    title_fn: Optional[Callable[[str], str]] = None,
    previous_window_toc_block: str = "(none)",
    general_qa_mode: GeneralQaMode = "nominal",
) -> List[TurnsNode]:
    """将连续 TurnSpan 列表转为 TurnsNode（使用固定 previous_toc 快照，不做 fragment 串行继承）。"""
    n = len(dialogue)
    def _build_one(spec: Tuple[int, TurnSpan]) -> TurnsNode:
        i, sp = spec
        st, en = sp.start_turn, sp.end_turn
        log_progress(f"Fragment {i + 1}/{len(spans)}: turns [{st}, {en})")
        raw_text = "\n".join(_format_turn_for_fragment_excerpt(dialogue[k]) for k in range(st, en))
        lane = _classify_fragment_lane(llm, raw_text, previous_toc_block=previous_window_toc_block)
        lane_norm = normalize_memory_lane(lane)
        start_ts = (dialogue[st].get("timestamp") or "").strip() if st < n else ""
        end_ts = (dialogue[en - 1].get("timestamp") or "").strip() if (en - 1) < n else ""
        title, summary = "", ""
        if not should_skip_non_durable_title_summary(lane_norm, general_qa_mode):
            title, summary = _llm_fragment_title_and_summary(
                llm,
                raw_text,
                lane_norm,
                previous_toc_block=previous_window_toc_block,
            )
        if title_fn is not None:
            title = title_fn(raw_text)
        if not (title or "").strip():
            title = f"Fragment_{i}"
        fragment_id = f"frag_{i}"
        return TurnsNode(
            chapter_id=fragment_id,
            fragment_id=fragment_id,
            title=title,
            confirmed_only=[],
            pending="",
            turn_time_range=TurnTimeRange(start=_parse_time_value(start_ts), end=_parse_time_value(end_ts))
            if start_ts and end_ts
            else None,
            spans=[sp],
            summary=summary,
            meta={"start_turn": st, "end_turn": en, "memory_lane": lane_norm},
        )

    indexed_spans = list(enumerate(spans))
    if _toc_parallel_enabled() and len(indexed_spans) > 1:
        workers = min(_toc_parallel_workers(), len(indexed_spans))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_build_one, indexed_spans))
    return [_build_one(spec) for spec in indexed_spans]


def _build_fragment_children_topic_shifts(
    llm: LLMClient,
    dialogue: List[dict],
    *,
    session_id: str,
    title_fn: Optional[Callable[[str], str]] = None,
    prior_sessions_toc: str = "(none)",
    context_provider: Optional[ReconcileContextProvider] = None,
    general_qa_mode: GeneralQaMode = "nominal",
) -> Tuple[List[TurnsNode], Optional[str], Optional[str], Optional[List[ConfirmedMemoryItem]], Optional[List[ConfirmedMemoryItem]]]:
    n = len(dialogue)
    if n == 0:
        return ([], None, None, None, None)

    if _membook_fragment_mode() == "fixed_turns":
        chunk = _membook_fixed_turn_chunk_size()
        spans = _build_fixed_turn_spans(n, chunk)
        if not spans:
            return ([], None, None, None)
        log_progress(
            f"Fragmentation: MEMBOOK_FRAGMENT_MODE=fixed_turns chunk={chunk} → {len(spans)} fragment(s)."
        )
        ch = _turn_spans_to_fragment_nodes(
            llm,
            dialogue,
            spans,
            session_id=session_id,
            title_fn=title_fn,
            previous_window_toc_block="(none)",
            general_qa_mode=general_qa_mode,
        )
        return (ch, None, None, None, None)

    sliding_enable_min_turns = int(env("MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS", "15").strip() or "15")
    sliding_window_size = int(env("MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE", "20").strip() or "20")

    if n >= sliding_enable_min_turns and sliding_window_size > 0:
        ch, pp, ps, pc, pg = _build_fragments_live_dynamic_sliding(
            llm,
            dialogue,
            session_id=session_id,
            title_fn=title_fn,
            prior_sessions_toc=prior_sessions_toc,
            context_provider=context_provider,
            general_qa_mode=general_qa_mode,
        )
        return (ch, pp, ps, pc, pg)

    # 短会话单次检测：会话起点尚未产生 fragment，previous_toc 显式为 "(none)"。
    start_indices = _detect_topic_shift_start_indices(llm, dialogue, previous_toc_block="(none)")
    cuts = list(start_indices) + [n]

    # 生成连续 spans；若出现异常 cuts，则回退为整段
    spans: List[TurnSpan] = []
    ok = True
    for a, b in zip(cuts[:-1], cuts[1:]):
        if not (0 <= a < b <= n):
            ok = False
            break
        spans.append(TurnSpan(start_turn=a, end_turn=b))
    if not ok or not spans or spans[0].start_turn != 0 or spans[-1].end_turn != n:
        if _strict_topic_shifts_enabled():
            raise RuntimeError(
                "Topic-shift cuts are invalid while MEMBOOK_STRICT_TOPIC_SHIFTS=1 "
                f"(ok={ok}, n={n}, cuts={cuts!r}). "
                "Unset MEMBOOK_STRICT_TOPIC_SHIFTS or fix detection output."
            )
        # 强制硬切分（每 5 turns），避免回退成整段一个 fragment
        spans = []
        for st in range(0, n, 5):
            en = min(n, st + 5)
            if en > st:
                spans.append(TurnSpan(start_turn=st, end_turn=en))

    n_frag = len(spans)
    log_progress(f"Topic shifts → {n_frag} fragment(s). Each fragment: classify lane → title+summary.")
    ch = _turn_spans_to_fragment_nodes(
        llm,
        dialogue,
        spans,
        session_id=session_id,
        title_fn=title_fn,
        general_qa_mode=general_qa_mode,
    )
    return (ch, None, None, None, None)


def _fragment_title_summary_template(lane: str) -> str:
    lane = normalize_memory_lane(lane)
    if lane == MEMORY_LANE_GENERAL_QA:
        return GENERATE_FRAGMENT_TITLE_AND_SUMMARY_QA
    if lane == MEMORY_LANE_INTERACTION_META:
        return GENERATE_FRAGMENT_TITLE_AND_SUMMARY_INTERACTION_META
    return GENERATE_FRAGMENT_TITLE_AND_SUMMARY_DURABLE


def _snippet_for_retry_log(text: str, limit: int = 220) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= limit:
        return s if s else "(empty)"
    return s[: limit - 1] + "…"


def _log_llm_retry_failure(
    op: str,
    attempt_idx: int,
    max_attempts: int,
    kind: str,
    *,
    detail: str = "",
) -> None:
    msg = f"{op} attempt {attempt_idx + 1}/{max_attempts} failed: [{kind}]"
    if detail:
        msg += f" — {detail}"
    log_progress(msg)


def _log_retry_traceback(op: str) -> None:
    """Call only from inside an ``except`` block. Enable with MEMBOOK_RETRY_TRACEBACK=1."""
    if env("MEMBOOK_RETRY_TRACEBACK", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    tb = traceback.format_exc()
    if not (tb or "").strip():
        return
    tail = "\n".join(tb.strip().splitlines()[-15:])
    log_progress(f"{op} traceback (tail): {tail}")


def _llm_fragment_title_and_summary(
    llm: LLMClient,
    raw_text: str,
    lane: str,
    *,
    previous_toc_block: str,
    confirmed_memory_points: str = "(none)",
) -> Tuple[str, str]:
    """一次调用生成 title + summary；JSON 解析失败时最多重试三次。"""
    raw_text = (raw_text or "").strip()
    lane_norm = normalize_memory_lane(lane)
    template = _fragment_title_summary_template(lane)
    if lane_norm != MEMORY_LANE_USER_DURABLE and not raw_text:
        return "", ""
    fmt_args: Dict[str, str] = {}
    if "{text}" in template:
        fmt_args["text"] = raw_text
    if "{previous_toc}" in template:
        fmt_args["previous_toc"] = previous_toc_block
    if "{confirmed_memory_points}" in template:
        fmt_args["confirmed_memory_points"] = confirmed_memory_points
    prompt = template.format(**fmt_args)
    last_out = ""
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Generating fragment title+summary (GENERATE_FRAGMENT_TITLE_AND_SUMMARY_*)"
                if attempt == 0
                else (
                    "Retry generating fragment title+summary "
                    f"(attempt {attempt + 1}/3, same prompt)"
                )
            )
            log_llm_phase(phase, prompt)
            out = (
                llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or ""
            ).strip()
            last_out = out
            if not out:
                _log_llm_retry_failure(
                    "GENERATE_FRAGMENT_TITLE_AND_SUMMARY",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            obj = parse_json_object(out)
            title = str(obj.get("title") or "").strip()
            summary = str(obj.get("summary") or "").strip()
            if title or summary:
                return title, summary
            keys = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            _log_llm_retry_failure(
                "GENERATE_FRAGMENT_TITLE_AND_SUMMARY",
                attempt,
                3,
                "parsed_json_but_empty_title_and_summary",
                detail=f"top_level_keys={keys} raw_head={_snippet_for_retry_log(out)}",
            )
        except Exception as e:
            _log_llm_retry_failure(
                "GENERATE_FRAGMENT_TITLE_AND_SUMMARY",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("GENERATE_FRAGMENT_TITLE_AND_SUMMARY")
            continue
    log_progress(
        "GENERATE_FRAGMENT_TITLE_AND_SUMMARY: exhausted 3 attempts; "
        "using title='' and first-line fallback from last raw output if any"
    )
    fallback = last_out.split("\n")[0].strip() if last_out else ""
    return "", fallback


def _format_fragment_confirmed_memories_for_prompt(confirmed_memories: List[Dict[str, Any]]) -> str:
    if not confirmed_memories:
        return "(none)"
    lines: List[str] = []
    for i, item in enumerate(confirmed_memories):
        txt = str(item.get("text") or "").strip()
        if not txt:
            continue
        t0 = int(item.get("source_turn_start") or 0)
        t1 = int(item.get("source_turn_end") or t0 + 1)
        lines.append(f"- [{i}] turns=[{t0},{t1}) text={txt}")
    return "\n".join(lines) if lines else "(none)"


def _llm_fragment_title_and_summary_from_confirmed(
    llm: LLMClient,
    *,
    confirmed_memories: List[Dict[str, Any]],
    previous_toc_block: str,
) -> Tuple[str, str]:
    """user_durable 片段：仅基于 confirmed memory points 生成 title+summary。"""
    points_block = _format_fragment_confirmed_memories_for_prompt(confirmed_memories)
    return _llm_fragment_title_and_summary(
        llm,
        "",
        MEMORY_LANE_USER_DURABLE,
        previous_toc_block=previous_toc_block,
        confirmed_memory_points=points_block,
    )


def _llm_general_qa_fragment_title_and_summary_from_confirmed(
    llm: LLMClient,
    *,
    raw_text: str,
    confirmed_memories: List[Dict[str, Any]],
    previous_toc_block: str,
) -> Tuple[str, str]:
    points_block = _format_fragment_confirmed_memories_for_prompt(confirmed_memories)
    return _llm_fragment_title_and_summary(
        llm,
        raw_text,
        MEMORY_LANE_GENERAL_QA,
        previous_toc_block=previous_toc_block,
        confirmed_memory_points=points_block,
    )


def _parse_time_value(raw: str) -> TimeValue:
    raw = (raw or "").strip()
    if not raw:
        return TimeValue(raw="")
    try:
        dt = datetime.strptime(raw, "%b %d, %Y, %H:%M:%S")
        return TimeValue(raw=raw, iso=dt.isoformat(), unix_sec=int(dt.timestamp()))
    except Exception:
        # 保底保留 raw，避免因个别格式异常导致构建失败
        return TimeValue(raw=raw, iso="", unix_sec=None)


_CHRONICLE_LME_TS_RE = re.compile(
    r"^\s*(?P<y>\d{4})[/-](?P<m>\d{1,2})[/-](?P<d>\d{1,2})\s*\((?P<wd>[A-Za-z]{3,})\)\s*(?P<h>\d{1,2}):(?P<min>\d{2})",
    re.IGNORECASE,
)

_ISO_CHRONICLE_FMTS: Tuple[Tuple[str, int], ...] = (
    ("%Y-%m-%dT%H:%M:%S", 19),
    ("%Y-%m-%dT%H:%M", 16),
    ("%Y-%m-%d %H:%M:%S", 19),
    ("%Y-%m-%d %H:%M", 16),
)


def chronicle_title_prefix_from_time_hint(session_time_hint: str) -> str:
    """
    Build LongMemEval/HaluMem-compatible title prefix: ``[YYYY-MM-DD (Weekday) HH:MM]``.
    Uses the START of TIME_HINT (before `` - `` when a range).
    """
    raw = (session_time_hint or "").strip()
    if not raw or raw == "(unknown)":
        return ""
    anchor = raw.split(" - ", 1)[0].strip()
    m = _CHRONICLE_LME_TS_RE.match(anchor)
    if m:
        wd = m.group("wd")
        if len(wd) == 3:
            wd = wd.title()
        return (
            f"[{int(m.group('y')):04d}-{int(m.group('m')):02d}-{int(m.group('d')):02d} "
            f"({wd}) {int(m.group('h')):02d}:{m.group('min')}]"
        )
    for fmt, n in _ISO_CHRONICLE_FMTS:
        chunk = anchor[:n] if len(anchor) >= n else anchor
        try:
            dt = datetime.strptime(chunk, fmt)
        except ValueError:
            continue
        wd = dt.strftime("%a")
        return f"[{dt.strftime('%Y-%m-%d')} ({wd}) {dt.strftime('%H:%M')}]"
    return ""


def apply_chronicle_title_time_prefix(title: str, session_time_hint: str) -> str:
    """Merge LLM theme with deterministic time prefix from TIME_HINT."""
    prefix = chronicle_title_prefix_from_time_hint(session_time_hint)
    if not prefix:
        return (title or "").strip()
    stripped = (title or "").strip()
    theme = ""
    if stripped.startswith("["):
        close = stripped.find("]")
        if close > 0:
            theme = stripped[close + 1 :].strip()
    elif stripped:
        theme = stripped
    if theme:
        return f"{prefix} {theme}"
    return prefix


def format_session_time_hint(
    time_span: Optional[TurnTimeRange],
    dialogue: List[dict],
) -> str:
    """编年史 Title：优先 Session time_span；否则 dialogue 首尾 turn 时间戳。"""
    if time_span is not None:
        s = (time_span.start.iso or time_span.start.raw or "").strip()
        e = (time_span.end.iso or time_span.end.raw or "").strip()
        if s and e:
            return f"{s} - {e}"
        if s:
            return s
        if e:
            return e
    if dialogue:
        ts0 = str((dialogue[0].get("timestamp") or "")).strip()
        ts1 = str((dialogue[-1].get("timestamp") or "")).strip()
        if ts0 and ts1:
            return f"{ts0} - {ts1}"
        if ts0:
            return ts0
        if ts1:
            return ts1
    return "(unknown)"


def format_prior_sessions_toc_block(session_nodes: List[SessionNode]) -> str:
    """过往 Session 的 Title + Summary 列表（不包含 fragment 明细）。"""
    if not session_nodes:
        return "(none)"
    blocks: List[str] = []
    for sn in session_nodes:
        title = (sn.title or sn.session_id or "").strip() or "(untitled session)"
        summary = (sn.summary or "").strip() or "(none)"
        blocks.append(f"{title} - {summary}")
    return "\n".join(blocks)


def _truncate_middle(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[truncated]...\n\n" + text[-half:]


def _env_int(name: str, default: int) -> int:
    raw = (env(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (env(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _toc_parallel_workers() -> int:
    """并行 worker 数；<=1 视为关闭并行。"""
    raw = env("MEMBOOK_TOC_PARALLEL_WORKERS", "").strip()
    if not raw:
        return 8
    try:
        return int(raw)
    except ValueError:
        return 8


def _toc_parallel_enabled() -> bool:
    return _toc_parallel_workers() > 1


def _shared_truncation_char_cap() -> int:
    """
    Default UTF-8-ish char budget for one middle-truncated long block, derived from
    ``MEMBOOK_CONTEXT_WINDOW_TOKENS`` minus ``MEMBOOK_LLM_RESERVE_TOKENS`` (room for completion,
    system/template, other prompt fields, and sibling blocks).

    Defaults target gpt-4o / gpt-4o-mini class 128k context; override ``MEMBOOK_CONTEXT_WINDOW_TOKENS``
    for other deployments (e.g. 64k). ``MEMBOOK_CHARS_PER_TOKEN_ESTIMATE`` (default 4) converts
    token budget to a conservative char cap (mixed CJK/EN).

    ``MEMBOOK_SESSION_SUMMARY_BUDGET_TOKENS`` caps the nominal block size in tokens (default 7000,
    i.e. ~28k chars at 4 chars/token — historical default).
    """
    cpt = _env_float("MEMBOOK_CHARS_PER_TOKEN_ESTIMATE", 4.0)
    ctx = _env_int("MEMBOOK_CONTEXT_WINDOW_TOKENS", 128000)
    reserve = _env_int("MEMBOOK_LLM_RESERVE_TOKENS", 36000)
    remainder = max(4096, ctx - reserve)
    baseline_tok = _env_int("MEMBOOK_SESSION_SUMMARY_BUDGET_TOKENS", 7000)
    block_tok = min(baseline_tok, max(2048, remainder // 6))
    return max(256, int(block_tok * cpt))


def _char_cap_from_env_or_shared(explicit_chars_env: str) -> int:
    """Use ``explicit_chars_env`` if set; else ``_shared_truncation_char_cap()``."""
    raw = (env(explicit_chars_env, "") or "").strip()
    if raw:
        try:
            return max(256, int(raw))
        except ValueError:
            pass
    return _shared_truncation_char_cap()


def format_prior_sessions_global_reconcile_context(session_nodes: List[SessionNode]) -> str:
    """
    Naive Global Summary（Reconcile 消融）：整块 prior Session title+summaries，截断后注入 RELATED context。

    复用 ``MEMBOOK_PRIOR_SESSIONS_TITLE_SUMMARIES_MAX_CHARS``；未单独设置时与同 env 推导的默认 cap 一致。
    """
    inner = format_prior_sessions_toc_block(session_nodes)
    if not inner or inner.strip() == "(none)":
        return "(none)"
    max_chars = _char_cap_from_env_or_shared("MEMBOOK_PRIOR_SESSIONS_TITLE_SUMMARIES_MAX_CHARS")
    truncated = _truncate_middle(inner, max_chars)
    return truncated if truncated.strip() else "(none)"


def _span_intersect_turn_range(sp: TurnSpan, t0: int, t1: int) -> Optional[TurnSpan]:
    lo = max(sp.start_turn, t0)
    hi = min(sp.end_turn, t1)
    if lo < hi:
        return TurnSpan(start_turn=lo, end_turn=hi)
    return None


def _indexed_user_durable_dialogue_zone(
    dialogue: List[dict], children: List[TurnsNode], t0: int, t1: int
) -> str:
    """仅 user_durable：窗口内带全局 turn 索引的对话行，供 reconcile 与 gate 共用。"""
    turn_indices: set[int] = set()
    for c in children:
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane != MEMORY_LANE_USER_DURABLE:
            continue
        for sp in c.spans or []:
            inter = _span_intersect_turn_range(sp, t0, t1)
            if inter is None:
                continue
            for k in range(inter.start_turn, inter.end_turn):
                if 0 <= k < len(dialogue):
                    turn_indices.add(k)
    if not turn_indices:
        return "(none)"
    lines: List[str] = []
    for k in sorted(turn_indices):
        lines.append(f"{k}. {_format_turn(dialogue[k])}")
    return "\n".join(lines)


def _indexed_general_qa_dialogue_zone(
    dialogue: List[dict], children: List[TurnsNode], t0: int, t1: int
) -> str:
    """仅 general_qa：窗口内带全局 turn 索引的对话行，供 general_qa extract 使用。"""
    turn_indices: set[int] = set()
    for c in children:
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane != MEMORY_LANE_GENERAL_QA:
            continue
        for sp in c.spans or []:
            inter = _span_intersect_turn_range(sp, t0, t1)
            if inter is None:
                continue
            for k in range(inter.start_turn, inter.end_turn):
                if 0 <= k < len(dialogue):
                    turn_indices.add(k)
    if not turn_indices:
        return "(none)"
    lines: List[str] = []
    for k in sorted(turn_indices):
        lines.append(f"{k}. {_format_turn(dialogue[k])}")
    return "\n".join(lines)


def _build_user_track_material_zone(
    dialogue: List[dict], children: List[TurnsNode], t0: int, t1: int
) -> str:
    """用户轨混合粒度，限制在 [t0, t1) 与 fragment 相交部分。"""
    parts: List[str] = []
    for c in children:
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane == MEMORY_LANE_GENERAL_QA:
            continue
        overlaps = False
        for sp in c.spans or []:
            if _span_intersect_turn_range(sp, t0, t1) is not None:
                overlaps = True
                break
        if not overlaps:
            continue
        if lane == MEMORY_LANE_USER_DURABLE:
            sub_spans = []
            for sp in c.spans or []:
                inter = _span_intersect_turn_range(sp, t0, t1)
                if inter is not None:
                    sub_spans.append(inter)
            raw = _dialogue_raw_from_spans(dialogue, sub_spans) if sub_spans else ""
            if (raw or "").strip():
                parts.append(raw.strip())
        elif lane == MEMORY_LANE_INTERACTION_META:
            continue
        else:
            sub_spans = []
            for sp in c.spans or []:
                inter = _span_intersect_turn_range(sp, t0, t1)
                if inter is not None:
                    sub_spans.append(inter)
            raw = _dialogue_raw_from_spans(dialogue, sub_spans) if sub_spans else ""
            if (raw or "").strip():
                parts.append(raw.strip())
    if not parts:
        return "(none)"
    return "\n".join(parts)


def _zone_has_user_durable(children: List[TurnsNode], t0: int, t1: int) -> bool:
    for c in children:
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) != MEMORY_LANE_USER_DURABLE:
            continue
        for sp in c.spans or []:
            if _span_intersect_turn_range(sp, t0, t1) is not None:
                return True
    return False


def _format_previous_pending_block(pending: str) -> str:
    s = (pending or "").strip()
    if not s:
        return "(none)"
    return s


def _format_zone_time_hint(dialogue: List[dict], t0: int, t1: int) -> str:
    """责任区 [t0, t1) 的时间提示，优先输出 ISO 区间。"""
    n = len(dialogue)
    lo = max(0, t0)
    hi = min(n, t1)
    if lo >= hi:
        return "(unknown)"
    ts0 = str((dialogue[lo].get("timestamp") or "")).strip()
    ts1 = str((dialogue[hi - 1].get("timestamp") or "")).strip()
    if not (ts0 or ts1):
        return "(unknown)"
    sv = (_parse_time_value(ts0).iso if ts0 else "") or ts0
    ev = (_parse_time_value(ts1).iso if ts1 else "") or ts1
    sv = (sv or "").strip()
    ev = (ev or "").strip()
    if sv and ev:
        return f"{sv} - {ev}"
    return sv or ev or "(unknown)"


def _format_prior_confirmed_snapshot(items: List[ConfirmedMemoryItem]) -> str:
    if not items:
        return "(none)"
    parts: List[str] = []
    for it in items:
        text = str(it.text or "").strip()
        if not text:
            continue
        parts.append(f"- {text}")
    return "\n".join(parts)


def _format_confirmed_memory_list_for_title(
    items: List[ConfirmedMemoryItem],
    children: List[TurnsNode],
) -> str:
    if not items:
        return "(none)"
    chapter_title_map: Dict[str, str] = {}
    for c in children:
        chapter_title_map[str(c.chapter_id)] = (c.title or "").strip() or "Unknown Chapter"
    grouped: Dict[str, List[str]] = defaultdict(list)
    ordered_titles: List[str] = []
    for it in items[:40]:
        fid = str(it.fragment_id or "").strip()
        title = chapter_title_map.get(fid, "Unknown Chapter")
        if title not in grouped:
            ordered_titles.append(title)
        grouped[title].append(it.text)
    lines: List[str] = []
    for title in ordered_titles:
        lines.append(title)
        lines.extend(f"- {txt}" for txt in grouped[title])
    return "\n".join(lines) if lines else "(none)"



def _normalize_text_for_id(text: str) -> str:
    # For strict determinism: normalize whitespace and case.
    s = (text or "").strip()
    s = " ".join(s.split())
    return s.lower()


def _extract_session_id_from_fragment_id(fragment_id: str) -> str:
    # Current fragment_id pattern is "frag_{i}", so this usually returns empty.
    raw = str(fragment_id or "").strip()
    marker = "_chapter_"
    if marker in raw:
        return raw.split(marker, 1)[0]
    return ""


def _compute_memory_point_id(
    *,
    session_id: str,
    fragment_id: str,
    source_turn_start: int,
    source_turn_end: int,
    text: str,
) -> str:
    """
    Strict deterministic memory_point_id:
    - includes session_id + fragment_id + exact turn_range + normalized text
    - any change in these fields => different id
    """
    norm_text = _normalize_text_for_id(text)
    payload = f"{session_id}|{fragment_id}|{source_turn_start}|{source_turn_end}|{norm_text}"
    # Use short prefix for compact meta; collision risk is negligible for this scale.
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_fragment_for_turn_range(
    turn_start: int,
    turn_end: int,
    children: List[TurnsNode],
    *,
    lane_name: str = MEMORY_LANE_USER_DURABLE,
) -> Tuple[str, bool]:
    """返回 (fragment_id, binding_ambiguous)；仅 lane_name。"""
    hits: List[str] = []
    for c in children:
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) != lane_name:
            continue
        for sp in c.spans or []:
            lo = max(sp.start_turn, turn_start)
            hi = min(sp.end_turn, turn_end)
            if lo < hi:
                hits.append(c.fragment_id)
                break
    uniq = list(dict.fromkeys(hits))
    if not uniq:
        return "", False
    if len(uniq) > 1:
        return uniq[0], True
    return uniq[0], False


def _enrich_confirmed_item_from_raw(
    raw: Dict[str, Any],
    dialogue: List[dict],
    children: List[TurnsNode],
    *,
    lane_name: str = MEMORY_LANE_USER_DURABLE,
) -> Optional[ConfirmedMemoryItem]:
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    n = len(dialogue)
    try:
        ts = int(raw.get("source_turn_start", 0))
        te = int(raw.get("source_turn_end", ts + 1))
    except (TypeError, ValueError):
        ts, te = 0, 1
    ts = max(0, min(ts, max(0, n - 1)))
    te = max(ts + 1, min(te, n))
    tsm = str((dialogue[ts].get("timestamp") or "")).strip() if ts < n else ""
    last = min(te - 1, n - 1)
    tem = str((dialogue[last].get("timestamp") or "")).strip() if last >= 0 and last < n else ""
    fid, amb = _resolve_fragment_for_turn_range(ts, te, children, lane_name=lane_name)
    session_id = _extract_session_id_from_fragment_id(fid)
    memory_point_id = _compute_memory_point_id(
        session_id=session_id,
        fragment_id=fid,
        source_turn_start=ts,
        source_turn_end=te,
        text=text,
    )
    return ConfirmedMemoryItem(
        text=text,
        source_turn_start=ts,
        source_turn_end=te,
        timestamp_start=tsm,
        timestamp_end=tem,
        fragment_id=fid,
        chapter_id="",
        memory_point_id=memory_point_id,
        binding_ambiguous=amb,
    )


def _merge_confirmed_lists(
    prev: List[ConfirmedMemoryItem],
    new_items: List[ConfirmedMemoryItem],
) -> List[ConfirmedMemoryItem]:
    seen: set[str] = set()
    out: List[ConfirmedMemoryItem] = []
    for it in prev + new_items:
        key = str(getattr(it, "memory_point_id", "") or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _memory_item_lane(item: ConfirmedMemoryItem, children: List[TurnsNode]) -> str:
    fid = str(item.fragment_id or "").strip()
    if not fid:
        return MEMORY_LANE_USER_DURABLE
    for c in children:
        source_ids = list((c.meta or {}).get("source_fragment_ids") or [])
        if fid in source_ids or fid == str(c.fragment_id or "").strip():
            return normalize_memory_lane((c.meta or {}).get("memory_lane"))
    return MEMORY_LANE_USER_DURABLE


def _lane_from_memory_dict(item: Dict[str, Any], children: List[TurnsNode]) -> str:
    fid = str(item.get("fragment_id") or "").strip()
    if not fid:
        return MEMORY_LANE_USER_DURABLE
    for c in children:
        source_ids = list((c.meta or {}).get("source_fragment_ids") or [])
        if fid in source_ids or fid == str(c.fragment_id or "").strip():
            return normalize_memory_lane((c.meta or {}).get("memory_lane"))
    return MEMORY_LANE_USER_DURABLE


def _repartition_confirmed_to_children(
    children: List[TurnsNode],
    confirmed_list: List[ConfirmedMemoryItem],
) -> None:
    by_id: Dict[str, List[ConfirmedMemoryItem]] = defaultdict(list)
    for m in confirmed_list:
        node_ref = str(m.chapter_id or "").strip()
        if node_ref:
            by_id[node_ref].append(m)
    for i, c in enumerate(children):
        node_id = str(c.chapter_id or "").strip()
        lst = by_id.get(node_id, [])
        dumped = [x.model_dump() for x in lst]
        children[i] = c.model_copy(
            update={
                "meta": {**(c.meta or {}), "confirmed_memories": dumped},
                "confirmed_only": [x.text for x in lst],
            }
        )


def _refresh_durable_children_summary_from_confirmed(
    llm: LLMClient,
    dialogue: List[dict],
    children: List[TurnsNode],
    *,
    previous_toc_block: str = "(none)",
) -> None:
    for idx, c in enumerate(children):
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane != MEMORY_LANE_USER_DURABLE:
            continue
        confirmed = list((c.meta or {}).get("confirmed_memories") or [])
        title, summary = _llm_fragment_title_and_summary_from_confirmed(
            llm,
            confirmed_memories=confirmed,
            previous_toc_block=previous_toc_block,
        )
        if confirmed and _looks_like_empty_facts_summary(summary):
            # Guardrail: when confirmed memories exist, avoid empty-facts summaries.
            fallback = _summary_from_confirmed_facts(confirmed)
            if fallback:
                summary = fallback
        if not (title or "").strip():
            title = c.title or c.fragment_id
        children[idx] = c.model_copy(update={"title": title, "summary": summary})


def _refresh_general_qa_children_summary_from_confirmed(
    llm: LLMClient,
    dialogue: List[dict],
    children: List[TurnsNode],
    *,
    previous_toc_block: str = "(none)",
) -> None:
    for idx, c in enumerate(children):
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane != MEMORY_LANE_GENERAL_QA:
            continue
        raw_text = _dialogue_raw_from_spans(dialogue, c.spans or [])
        confirmed = list((c.meta or {}).get("confirmed_memories") or [])
        points = [m for m in confirmed if _lane_from_memory_dict(m, children) == MEMORY_LANE_GENERAL_QA]
        title, summary = _llm_general_qa_fragment_title_and_summary_from_confirmed(
            llm,
            raw_text=raw_text,
            confirmed_memories=points,
            previous_toc_block=previous_toc_block,
        )
        if not (title or "").strip():
            title = c.title or c.fragment_id
        if not (summary or "").strip():
            summary = c.summary
        children[idx] = c.model_copy(update={"title": title, "summary": summary})


def _looks_like_empty_facts_summary(text: str) -> bool:
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


def _summary_from_confirmed_facts(confirmed: List[Dict[str, Any]], max_points: int = 3) -> str:
    points: List[str] = []
    seen: set[str] = set()
    for it in confirmed:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        points.append(text)
        if len(points) >= max_points:
            break
    if not points:
        return ""
    return " ".join(points)


def _parse_pending_text_json(text: str) -> str:
    try:
        obj = parse_json_object(text)
    except Exception:
        return ""
    raw_p = obj.get("pending_summary")
    if raw_p is None:
        raw_p = obj.get("pending_text")
    if raw_p is None:
        raw_p = obj.get("pending_paragraph")
    if isinstance(raw_p, str) and raw_p.strip():
        return " ".join(raw_p.split())
    raw = obj.get("pending_items")
    if not isinstance(raw, list):
        return ""
    parts: List[str] = []
    seen: set[str] = set()
    for p in raw:
        if isinstance(p, str):
            s = p.strip()
        elif isinstance(p, dict):
            s = str(p.get("text") or "").strip()
        else:
            continue
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        parts.append(s)
    return ". ".join(parts) if parts else ""


def _parse_merged_reconcile_json(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        obj = parse_json_object(text)
    except Exception:
        return "", []
    raw_p = obj.get("pending_summary")
    if raw_p is None:
        raw_p = obj.get("pending_text")
    if raw_p is None:
        raw_p = obj.get("pending_paragraph")
    pending = ""
    if isinstance(raw_p, str) and raw_p.strip():
        pending = " ".join(raw_p.split())
    raw_items = obj.get("confirmed_list")
    if raw_items is None:
        raw_items = obj.get("confirmed_items")
    items: List[Dict[str, Any]] = []
    if isinstance(raw_items, list):
        for x in raw_items:
            if isinstance(x, dict):
                items.append(x)
    return pending, items


def _parse_confirmed_list_json(text: str) -> List[Dict[str, Any]]:
    try:
        obj = parse_json_object(text)
    except Exception:
        return []
    raw_items = obj.get("confirmed_list")
    if raw_items is None:
        raw_items = obj.get("confirmed_items")
    out: List[Dict[str, Any]] = []
    if isinstance(raw_items, list):
        for x in raw_items:
            if isinstance(x, dict):
                out.append(x)
    return out


def _llm_reconcile_pending_and_confirmed_incremental(
    llm: LLMClient,
    previous_pending: str,
    prior_confirmed: List[ConfirmedMemoryItem],
    indexed_dialogue_zone: str,
    dialogue: List[dict],
    children: List[TurnsNode],
    t0: int,
    t1: int,
    *,
    session_id: str = "",
    context_provider: Optional[ReconcileContextProvider] = None,
) -> Tuple[str, List[ConfirmedMemoryItem]]:
    max_chars = _char_cap_from_env_or_shared("MEMBOOK_RECONCILE_MAX_CHARS")
    idx_zone = _truncate_middle((indexed_dialogue_zone or "").strip() or "(none)", max_chars)
    time_hint = _format_zone_time_hint(dialogue, t0, t1)
    # Cross-session only (in-session truth lives in PRIOR_CONFIRMED_LIST_SNAPSHOT).
    related_confirmed_context = "(none)"
    if context_provider:
        try:
            cross = context_provider(
                idx_zone,
                time_hint,
                session_id,
                prior_confirmed,
            )
            cross = (cross or "").strip()
            if cross and cross not in ("(none)",):
                related_confirmed_context = cross
        except Exception as e:
            log_progress(f"Reconcile context_provider failed ({type(e).__name__}): {e}; using in-session only")
    prompt = RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL.format(
        pending_text=_format_previous_pending_block(previous_pending),
        prior_confirmed_list_snapshot=_format_prior_confirmed_snapshot(prior_confirmed),
        related_confirmed_context=related_confirmed_context,
        indexed_dialogue_zone=idx_zone,
        time_hint=time_hint,
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Session reconcile merged (RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL)"
                if attempt == 0
                else f"Retry reconcile merged (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (
                llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or ""
            ).strip()
            if not out:
                _log_llm_retry_failure(
                    "RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            para, raw_items = _parse_merged_reconcile_json(out)
            new_struct: List[ConfirmedMemoryItem] = []
            for raw in raw_items:
                m = _enrich_confirmed_item_from_raw(raw, dialogue, children)
                if m is not None:
                    new_struct.append(m)
            merged = _merge_confirmed_lists(prior_confirmed, new_struct)
            n_sent = para.count(". ") + (1 if para.strip() else 0)
            log_progress(
                f"RECONCILE_MERGED: pending_text chars={len(para)}, approx_sentences={n_sent}, "
                f"new_confirmed={len(new_struct)}, total_confirmed={len(merged)}"
            )
            return para, merged
        except Exception as e:
            _log_llm_retry_failure(
                "RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL")
            continue
    kept = (previous_pending or "").strip()
    log_progress(
        "RECONCILE_MERGED: failed after retries; keeping previous pending and prior confirmed"
    )
    return kept, list(prior_confirmed)


def _llm_extract_general_qa_incremental(
    llm: LLMClient,
    prior_confirmed: List[ConfirmedMemoryItem],
    indexed_dialogue_zone: str,
    dialogue: List[dict],
    children: List[TurnsNode],
    t0: int,
    t1: int,
) -> List[ConfirmedMemoryItem]:
    max_chars = _char_cap_from_env_or_shared("MEMBOOK_RECONCILE_MAX_CHARS")
    idx_zone = _truncate_middle((indexed_dialogue_zone or "").strip() or "(none)", max_chars)
    time_hint = _format_zone_time_hint(dialogue, t0, t1)
    prompt = EXTRACT_GENERAL_QA_INCREMENTAL.format(
        prior_confirmed_list_snapshot=_format_prior_confirmed_snapshot(prior_confirmed),
        indexed_dialogue_zone=idx_zone,
        time_hint=time_hint,
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "General QA extract incremental (EXTRACT_GENERAL_QA_INCREMENTAL)"
                if attempt == 0
                else f"Retry general QA extract (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (
                llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or ""
            ).strip()
            if not out:
                _log_llm_retry_failure(
                    "EXTRACT_GENERAL_QA_INCREMENTAL",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            raw_items = _parse_confirmed_list_json(out)
            new_struct: List[ConfirmedMemoryItem] = []
            for raw in raw_items:
                m = _enrich_confirmed_item_from_raw(raw, dialogue, children, lane_name=MEMORY_LANE_GENERAL_QA)
                if m is not None:
                    new_struct.append(m)
            merged = _merge_confirmed_lists(prior_confirmed, new_struct)
            log_progress(
                f"GENERAL_QA_EXTRACT: new_confirmed={len(new_struct)}, total_confirmed={len(merged)}"
            )
            return merged
        except Exception as e:
            _log_llm_retry_failure(
                "EXTRACT_GENERAL_QA_INCREMENTAL",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("EXTRACT_GENERAL_QA_INCREMENTAL")
            continue
    log_progress("GENERAL_QA_EXTRACT: failed after retries; keeping prior confirmed")
    return list(prior_confirmed)


def _llm_session_summary_and_title_final(
    llm: LLMClient,
    prior_sessions_title_summaries: str,
    pending_text: str,
    confirmed_items: List[ConfirmedMemoryItem],
    children: List[TurnsNode],
    session_time_hint: str,
    *,
    fallback_title: str,
) -> Tuple[str, str]:
    """Session 收尾一次调用：由 confirmed_list + 时间锚（及 prior/pending 门控）产出 summary 与 title。"""
    max_prev = _char_cap_from_env_or_shared("MEMBOOK_SESSION_SUMMARY_MAX_CHARS")
    prior_cap = env("MEMBOOK_PRIOR_SESSIONS_TITLE_SUMMARIES_MAX_CHARS", "").strip()
    max_prior = int(prior_cap) if prior_cap else max_prev
    prior_raw = (prior_sessions_title_summaries or "").strip()
    if not prior_raw or prior_raw == "(none)":
        prior_block = "(none)"
    else:
        prior_block = _truncate_middle(prior_raw, max_prior)
    pend_block = _format_pending_exclusion_list(pending_text)
    confirmed_list_block = _format_confirmed_memory_list_for_title(confirmed_items or [], children)
    prompt = GENERATE_SESSION_SUMMARY_AND_TITLE_FINAL.format(
        prior_sessions_title_summaries=prior_block,
        confirmed_memory_list=confirmed_list_block,
        pending_text=pend_block,
        session_time_hint=session_time_hint,
    )
    op = "GENERATE_SESSION_SUMMARY_AND_TITLE_FINAL"
    for attempt in range(3):
        out = ""
        try:
            phase = (
                f"Session summary+title final ({op})"
                if attempt == 0
                else f"Retry session summary+title final (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (
                llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or ""
            ).strip()
            if not out:
                _log_llm_retry_failure(op, attempt, 3, "empty_llm_body")
                continue
            obj = parse_json_object(out)
            summary = str(obj.get("summary") or "").strip()
            summary = " ".join(summary.split()) if summary else ""
            title = apply_chronicle_title_time_prefix(
                str(obj.get("title") or "").strip(),
                session_time_hint,
            )
            fb = apply_chronicle_title_time_prefix(
                str(fallback_title or "").strip(),
                session_time_hint,
            )
            if title and summary:
                return summary, title
            if title:
                return summary, title
            if summary:
                return summary, fb or fallback_title
            keys = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            _log_llm_retry_failure(
                op,
                attempt,
                3,
                "parsed_json_but_empty_summary_and_title",
                detail=f"top_level_keys={keys} raw_head={_snippet_for_retry_log(out)}",
            )
        except Exception as e:
            _log_llm_retry_failure(
                op,
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback(op)
            continue
    log_progress(f"{op}: exhausted 3 attempts; using empty summary and fallback_title")
    fb = apply_chronicle_title_time_prefix(str(fallback_title or "").strip(), session_time_hint)
    return "", fb or fallback_title


def _run_one_audit_window(
    llm: LLMClient,
    dialogue: List[dict],
    children: List[TurnsNode],
    t0: int,
    t1: int,
    prev_pending: str,
    prev_confirmed: List[ConfirmedMemoryItem],
    prev_general_confirmed: List[ConfirmedMemoryItem],
    *,
    session_id: str = "",
    context_provider: Optional[ReconcileContextProvider] = None,
    general_qa_mode: GeneralQaMode = "nominal",
) -> Tuple[str, List[ConfirmedMemoryItem], List[ConfirmedMemoryItem]]:
    _ = general_qa_mode
    reconcile_on = env("MEMBOOK_SESSION_RECONCILE", "1").strip() == "1"
    indexed_zone_ud = _indexed_user_durable_dialogue_zone(dialogue, children, t0, t1)
    has_durable = _zone_has_user_durable(children, t0, t1)
    confirmed_acc = list(prev_confirmed)
    if reconcile_on and has_durable and (indexed_zone_ud or "").strip() not in ("", "(none)"):
        prev_pending, confirmed_acc = _llm_reconcile_pending_and_confirmed_incremental(
            llm,
            prev_pending,
            confirmed_acc,
            indexed_zone_ud,
            dialogue,
            children,
            t0,
            t1,
            session_id=session_id,
            context_provider=context_provider,
        )
    gq_acc = list(prev_general_confirmed)
    if general_qa_mode == "full":
        indexed_zone_gq = _indexed_general_qa_dialogue_zone(dialogue, children, t0, t1)
        if (indexed_zone_gq or "").strip() not in ("", "(none)"):
            gq_acc = _llm_extract_general_qa_incremental(
                llm,
                gq_acc,
                indexed_zone_gq,
                dialogue,
                children,
                t0,
                t1,
            )
    _repartition_confirmed_to_children(children, _merge_confirmed_lists(list(confirmed_acc), list(gq_acc)))
    return prev_pending, confirmed_acc, gq_acc


def run_session_audit_and_chronicle(
    llm: LLMClient,
    dialogue: List[dict],
    children: List[TurnsNode],
    window_spans: List[Tuple[int, int]],
    prior_sessions_title_summaries: str,
    time_span: Optional[TurnTimeRange],
    *,
    session_id: str,
    context_provider: Optional[ReconcileContextProvider] = None,
    general_qa_mode: GeneralQaMode = "nominal",
) -> Tuple[str, str, str, List[ConfirmedMemoryItem], List[ConfirmedMemoryItem]]:
    """统一审计 + 编年史：多窗或单窗 [(0,n)]；返回 (pending, summary, title, confirmed_list)。"""
    prev_pending: str = ""
    prev_confirmed: List[ConfirmedMemoryItem] = []
    prev_general_confirmed: List[ConfirmedMemoryItem] = []
    for t0, t1 in window_spans:
        if not (0 <= t0 < t1 <= len(dialogue) and t1 > t0):
            continue
        prev_pending, prev_confirmed, prev_general_confirmed = _run_one_audit_window(
            llm,
            dialogue,
            children,
            t0,
            t1,
            prev_pending,
            prev_confirmed,
            prev_general_confirmed,
            session_id=session_id,
            context_provider=context_provider,
            general_qa_mode=general_qa_mode,
        )
    _refresh_durable_children_summary_from_confirmed(
        llm,
        dialogue,
        children,
        previous_toc_block="(none)",
    )
    if general_qa_mode == "full":
        _refresh_general_qa_children_summary_from_confirmed(
            llm,
            dialogue,
            children,
            previous_toc_block="(none)",
        )
    hint = format_session_time_hint(time_span, dialogue)
    non_g = _non_general_children(children)
    fb_title = (non_g[0].title if non_g else None) or session_id
    summary_pending_text = prev_pending if general_qa_mode != "full" else ""
    summary_confirmed = (
        _merge_confirmed_lists(list(prev_confirmed), list(prev_general_confirmed))
        if general_qa_mode == "full"
        else list(prev_confirmed)
    )
    prev_summary, title = _llm_session_summary_and_title_final(
        llm,
        prior_sessions_title_summaries,
        summary_pending_text,
        summary_confirmed,
        children,
        hint,
        fallback_title=str(fb_title)
    )
    return prev_pending, prev_summary, title, prev_confirmed, prev_general_confirmed


def _dialogue_raw_from_spans(dialogue: List[dict], spans: List[TurnSpan]) -> str:
    if not spans:
        return ""
    ordered = sorted(spans, key=lambda sp: sp.start_turn)
    lines: List[str] = []
    for sp in ordered:
        for k in range(sp.start_turn, sp.end_turn):
            if 0 <= k < len(dialogue):
                lines.append(_format_turn(dialogue[k]))
    return "\n".join(lines)


def _non_general_children(children: List[TurnsNode]) -> List[TurnsNode]:
    return [
        c
        for c in children
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) != MEMORY_LANE_GENERAL_QA
    ]


def _build_user_durable_dialogue_for_reconcile(dialogue: List[dict], children: List[TurnsNode]) -> str:
    """仅 user_durable 片段：格式化轮次供 Session reconcile（不含 interaction_meta / general_qa）。"""
    parts: List[str] = []
    for i, c in enumerate(children):
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane != MEMORY_LANE_USER_DURABLE:
            continue
        raw = _dialogue_raw_from_spans(dialogue, c.spans or [])
        if (raw or "").strip():
            parts.append(f"[user_durable fragment index={i}]\n{raw.strip()}")
        else:
            parts.append(
                f"[user_durable fragment index={i}] (no raw spans; ToC only)\n"
                f"  title: {c.title or '(none)'}\n"
                f"  summary: {c.summary or '(none)'}"
            )
    if not parts:
        return "(none)"
    return "\n\n---\n\n".join(parts)


def _build_user_track_material(dialogue: List[dict], children: List[TurnsNode]) -> str:
    """用户轨拼装：user_durable 用 raw dialogue；interaction_meta 仅用章节 ToC（title/summary）。"""
    parts: List[str] = []
    for i, c in enumerate(children):
        lane = normalize_memory_lane((c.meta or {}).get("memory_lane"))
        if lane == MEMORY_LANE_GENERAL_QA:
            continue
        if lane == MEMORY_LANE_USER_DURABLE:
            raw = _dialogue_raw_from_spans(dialogue, c.spans or [])
            if (raw or "").strip():
                parts.append(raw.strip())
            else:
                parts.append(
                    "(no raw spans; ToC only)\n"
                    f"title: {c.title or '(none)'}\n"
                    f"summary: {c.summary or '(none)'}"
                )
        elif lane == MEMORY_LANE_INTERACTION_META:
            parts.append(
                "interaction_meta (ToC only)\n"
                f"title: {c.title or '(none)'}\n"
                f"summary: {c.summary or '(none)'}"
            )
        else:
            raw = _dialogue_raw_from_spans(dialogue, c.spans or [])
            if (raw or "").strip():
                parts.append(raw.strip())
    if not parts:
        return "(none)"
    return "\n\n---\n\n".join(parts)


def _format_pending_exclusion_list(pending_text: str) -> str:
    s = (pending_text or "").strip()
    if not s:
        return "(none)"
    return s


def _build_general_qa_chapters_toc(general_nodes: List[TurnsNode]) -> str:
    blocks: List[str] = []
    for c in general_nodes:
        title = c.title or "Untitled"
        summary = c.summary or "(none)"
        blocks.append(f"{title} - {summary}")
    return "\n".join(blocks)


def _build_interaction_meta_chapters_toc(meta_nodes: List[TurnsNode]) -> str:
    blocks: List[str] = []
    for c in meta_nodes:
        title = c.title or "Untitled"
        summary = c.summary or "(none)"
        blocks.append(f"{title} - {summary}")
    return "\n".join(blocks)


def _format_user_durable_fragments_for_consolidation(children: List[TurnsNode]) -> str:
    lines: List[str] = []
    for idx, c in enumerate(children):
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) != MEMORY_LANE_USER_DURABLE:
            continue
        title = (c.title or "").strip() or "(none)"
        summary = (c.summary or "").strip() or "(none)"
        confirmed = list((c.meta or {}).get("confirmed_memories") or [])
        points = []
        for item in confirmed[:5]:
            txt = str(item.get("text") or "").strip()
            if txt:
                points.append(txt)
        points_block = " | ".join(points) if points else "(none)"
        lines.append(
            f"[{idx}] fragment_id={c.fragment_id}\n"
            f"  title={title}\n"
            f"  summary={summary}\n"
            f"  confirmed_points={points_block}"
        )
    return "\n\n".join(lines) if lines else "(none)"


def _format_general_qa_fragments_for_merge(children: List[TurnsNode]) -> str:
    lines: List[str] = []
    for idx, c in enumerate(children):
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) != MEMORY_LANE_GENERAL_QA:
            continue
        title = (c.title or "").strip() or "(none)"
        summary = (c.summary or "").strip() or "(none)"
        confirmed = list((c.meta or {}).get("confirmed_memories") or [])
        points = []
        for item in confirmed[:5]:
            if _lane_from_memory_dict(item, children) != MEMORY_LANE_GENERAL_QA:
                continue
            txt = str(item.get("text") or "").strip()
            if txt:
                points.append(txt)
        points_block = " | ".join(points) if points else "(none)"
        lines.append(
            f"[{idx}] fragment_id={c.fragment_id}\n"
            f"  title={title}\n"
            f"  summary={summary}\n"
            f"  confirmed_points={points_block}"
        )
    return "\n\n".join(lines) if lines else "(none)"


def _parse_merge_fragments_to_chapters_groups(
    raw_text: str,
    allowed_indices: set[int],
) -> List[List[int]]:
    try:
        obj = parse_json_object(raw_text)
    except Exception:
        return []
    raw_groups = obj.get("merge_groups")
    if not isinstance(raw_groups, list):
        return []
    groups: List[List[int]] = []
    used: set[int] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, list):
            continue
        group: List[int] = []
        for x in raw_group:
            try:
                v = int(x)
            except Exception:
                continue
            if v in allowed_indices:
                group.append(v)
        group = sorted(set(group))
        if len(group) < 2:
            continue
        if any(v in used for v in group):
            continue
        groups.append(group)
        used.update(group)
    groups.sort(key=lambda g: g[0])
    return groups


def _llm_merge_fragments_to_chapters_groups(
    llm: LLMClient,
    children: List[TurnsNode],
) -> List[List[int]]:
    allowed_indices = {
        idx
        for idx, c in enumerate(children)
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) == MEMORY_LANE_USER_DURABLE
    }
    if len(allowed_indices) < 2:
        return []
    prompt = MERGE_FRAGMENTS_TO_CHAPTERS.format(
        fragments=_format_user_durable_fragments_for_consolidation(children)
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Merge fragments to chapters (MERGE_FRAGMENTS_TO_CHAPTERS)"
                if attempt == 0
                else f"Retry merge fragments to chapters (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or "").strip()
            if not out:
                _log_llm_retry_failure(
                    "MERGE_FRAGMENTS_TO_CHAPTERS",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            return _parse_merge_fragments_to_chapters_groups(out, allowed_indices)
        except Exception as e:
            _log_llm_retry_failure(
                "MERGE_FRAGMENTS_TO_CHAPTERS",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("MERGE_FRAGMENTS_TO_CHAPTERS")
    return []


def _llm_merge_general_fragments_to_chapters_groups(
    llm: LLMClient,
    children: List[TurnsNode],
) -> List[List[int]]:
    allowed_indices = {
        idx
        for idx, c in enumerate(children)
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) == MEMORY_LANE_GENERAL_QA
    }
    if len(allowed_indices) < 2:
        return []
    prompt = MERGE_FRAGMENTS_TO_CHAPTERS.format(
        fragments=_format_general_qa_fragments_for_merge(children)
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Merge general fragments to chapters (MERGE_FRAGMENTS_TO_CHAPTERS)"
                if attempt == 0
                else f"Retry merge general fragments to chapters (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or "").strip()
            if not out:
                _log_llm_retry_failure(
                    "MERGE_FRAGMENTS_TO_CHAPTERS_GENERAL",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            return _parse_merge_fragments_to_chapters_groups(out, allowed_indices)
        except Exception as e:
            _log_llm_retry_failure(
                "MERGE_FRAGMENTS_TO_CHAPTERS_GENERAL",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("MERGE_FRAGMENTS_TO_CHAPTERS_GENERAL")
    return []


def _llm_merged_user_durable_title_and_summary(
    llm: LLMClient,
    *,
    merged_confirmed_memory_points: str,
) -> Tuple[str, str]:
    prompt = GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE.format(
        merged_confirmed_memory_points=merged_confirmed_memory_points,
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Generate user_durable chapter title+summary (GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE)"
                if attempt == 0
                else f"Retry merge user_durable chapter title+summary (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or "").strip()
            if not out:
                _log_llm_retry_failure(
                    "GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE", attempt, 3, "empty_llm_body"
                )
                continue
            obj = parse_json_object(out)
            title = str(obj.get("title") or "").strip()
            summary = str(obj.get("summary") or "").strip()
            if title or summary:
                return title, summary
        except Exception as e:
            _log_llm_retry_failure(
                "GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE")
    return "", ""


def _consolidate_user_durable_session(
    llm: LLMClient,
    dialogue: List[dict],
    children: List[TurnsNode],
    confirmed_list: List[ConfirmedMemoryItem],
    *,
    session_id: str,
    previous_toc_block: str,
) -> Tuple[List[TurnsNode], List[ConfirmedMemoryItem], List[List[int]]]:
    merge_groups = _llm_merge_fragments_to_chapters_groups(llm, children)
    if not merge_groups:
        return children, confirmed_list, []

    group_map: Dict[int, int] = {}
    for gid, grp in enumerate(merge_groups):
        for idx in grp:
            group_map[idx] = gid

    merged_nodes: List[TurnsNode] = []
    source_fragment_ids_by_gid: Dict[int, set[str]] = defaultdict(set)
    source_fragment_titles_by_gid: Dict[int, Dict[str, str]] = defaultdict(dict)
    source_fragment_summaries_by_gid: Dict[int, Dict[str, str]] = defaultdict(dict)
    for gid, grp in enumerate(merge_groups):
        nodes = [children[idx] for idx in grp]
        all_spans: List[TurnSpan] = []
        all_confirmed_memories: List[Dict[str, Any]] = []
        for n in nodes:
            all_spans.extend(n.spans or [])
            nid = str(n.fragment_id)
            source_fragment_ids_by_gid[gid].add(nid)
            if nid:
                source_fragment_titles_by_gid[gid][nid] = str(n.title or "").strip()
                source_fragment_summaries_by_gid[gid][nid] = str(n.summary or "").strip()
            all_confirmed_memories.extend(list((n.meta or {}).get("confirmed_memories") or []))
        merged_spans = _merge_turn_spans(all_spans)
        merged_time_range = _turn_time_range_from_spans(dialogue, merged_spans)
        merged_points_block = _format_fragment_confirmed_memories_for_prompt(all_confirmed_memories)
        merged_title, merged_summary = _llm_merged_user_durable_title_and_summary(
            llm,
            merged_confirmed_memory_points=merged_points_block,
        )
        if not (merged_title or "").strip() or not (merged_summary or "").strip():
            merged_title, merged_summary = _llm_fragment_title_and_summary_from_confirmed(
                llm,
                confirmed_memories=all_confirmed_memories,
                previous_toc_block=previous_toc_block,
            )
        merged_nodes.append(
            TurnsNode(
                chapter_id="",
                fragment_id="",
                title=(merged_title or "").strip() or "Merged Durable",
                confirmed_only=[],
                pending="",
                turn_time_range=merged_time_range,
                spans=merged_spans,
                summary=(merged_summary or "").strip(),
                meta={
                    "memory_lane": MEMORY_LANE_USER_DURABLE,
                    "consolidated_user_durable": True,
                    "merged_from_chapter_indices": grp,
                    "merged_from_fragment_indices": grp,
                    "merged_from_fragment_ids": sorted(source_fragment_ids_by_gid[gid]),
                    "source_fragment_ids": sorted(source_fragment_ids_by_gid[gid]),
                    "source_fragment_titles": dict(source_fragment_titles_by_gid[gid]),
                    "source_fragment_summaries": dict(source_fragment_summaries_by_gid[gid]),
                },
            )
        )

    out_children: List[TurnsNode] = []
    inserted_groups: set[int] = set()
    for idx, child in enumerate(children):
        gid = group_map.get(idx)
        if gid is None:
            out_children.append(child)
            continue
        if gid in inserted_groups:
            continue
        out_children.append(merged_nodes[gid])
        inserted_groups.add(gid)
    return out_children, _merge_confirmed_lists([], list(confirmed_list)), merge_groups


def _assign_short_ids_and_mappings(
    children: List[TurnsNode],
    confirmed_list: List[ConfirmedMemoryItem],
    session_id: str,
) -> Tuple[List[TurnsNode], List[ConfirmedMemoryItem], Dict[str, Any]]:
    fragment_to_chapter: Dict[str, str] = {}
    reassigned_children: List[TurnsNode] = []
    for chap_idx, c in enumerate(children):
        chapter_id = f"chap_{chap_idx}"
        source_fragments = list((c.meta or {}).get("source_fragment_ids") or [])
        if not source_fragments:
            base_fragment = (c.fragment_id or "").strip()
            if base_fragment:
                source_fragments = [base_fragment]
        source_fragments = [f for f in source_fragments if f] or [f"frag_unbound_{chap_idx}"]
        for fid in source_fragments:
            fragment_to_chapter[fid] = chapter_id
        reassigned_children.append(
            c.model_copy(
                update={
                    "chapter_id": chapter_id,
                    "fragment_id": source_fragments[0],
                    "meta": {**(c.meta or {}), "source_fragment_ids": source_fragments},
                }
            )
        )

    memory_point_to_fragment: Dict[str, str] = {}
    reassigned_confirmed: List[ConfirmedMemoryItem] = []
    for mp_idx, m in enumerate(confirmed_list):
        mp_id = scoped_memory_point_id(session_id, mp_idx)
        fragment_id = str(m.fragment_id or "").strip()
        chapter_id = fragment_to_chapter.get(fragment_id, "")
        memory_point_to_fragment[mp_id] = fragment_id
        reassigned_confirmed.append(
            m.model_copy(
                update={
                    "memory_point_id": mp_id,
                    "fragment_id": fragment_id,
                    "chapter_id": chapter_id,
                }
            )
        )

    mappings = {
        "fragment_to_chapter": fragment_to_chapter,
        "memory_point_to_fragment": memory_point_to_fragment,
    }
    return reassigned_children, reassigned_confirmed, mappings


def _merge_turn_spans(spans: List[TurnSpan]) -> List[TurnSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda sp: (sp.start_turn, sp.end_turn))
    out: List[TurnSpan] = []
    for sp in ordered:
        if not out:
            out.append(TurnSpan(start_turn=sp.start_turn, end_turn=sp.end_turn))
            continue
        last = out[-1]
        if sp.start_turn <= last.end_turn:
            out[-1] = TurnSpan(start_turn=last.start_turn, end_turn=max(last.end_turn, sp.end_turn))
        else:
            out.append(TurnSpan(start_turn=sp.start_turn, end_turn=sp.end_turn))
    return out


def _turn_time_range_from_spans(dialogue: List[dict], spans: List[TurnSpan]) -> Optional[TurnTimeRange]:
    merged = _merge_turn_spans(spans)
    if not merged or not dialogue:
        return None
    st_turn = merged[0].start_turn
    en_turn = merged[-1].end_turn - 1
    if st_turn < 0 or st_turn >= len(dialogue):
        return None
    en_turn = max(st_turn, min(en_turn, len(dialogue) - 1))
    start_ts = str((dialogue[st_turn].get("timestamp") or "")).strip()
    end_ts = str((dialogue[en_turn].get("timestamp") or "")).strip()
    if not (start_ts and end_ts):
        return None
    return TurnTimeRange(start=_parse_time_value(start_ts), end=_parse_time_value(end_ts))


def _llm_merged_general_title_summary(
    llm: LLMClient,
    toc_block: str,
    *,
    merged_confirmed_memory_points: str = "(none)",
) -> Tuple[str, str]:
    prompt = GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA.format(
        general_chapters_toc=toc_block,
        merged_confirmed_memory_points=merged_confirmed_memory_points,
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Generate general_qa chapter title+summary (GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA)"
                if attempt == 0
                else f"Retry merge general_qa (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (
                llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or ""
            ).strip()
            if not out:
                _log_llm_retry_failure(
                    "GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            obj = parse_json_object(out)
            title = str(obj.get("title") or "").strip()
            summary = str(obj.get("summary") or "").strip()
            if title or summary:
                return title, summary
            keys = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            _log_llm_retry_failure(
                "GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA",
                attempt,
                3,
                "parsed_json_but_empty_title_and_summary",
                detail=f"top_level_keys={keys} raw_head={_snippet_for_retry_log(out)}",
            )
        except Exception as e:
            _log_llm_retry_failure(
                "GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA")
            continue
    log_progress(
        "GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA: exhausted 3 attempts; returning empty title and summary"
    )
    return "", ""


def _consolidate_general_qa_session(
    llm: LLMClient,
    dialogue: List[dict],
    children: List[TurnsNode],
    confirmed_list: List[ConfirmedMemoryItem],
    *,
    session_id: str,
    previous_toc_block: str,
) -> Tuple[List[TurnsNode], List[ConfirmedMemoryItem], List[List[int]]]:
    _ = session_id
    merge_groups = _llm_merge_general_fragments_to_chapters_groups(llm, children)
    if not merge_groups:
        return children, confirmed_list, []

    group_map: Dict[int, int] = {}
    for gid, grp in enumerate(merge_groups):
        for idx in grp:
            group_map[idx] = gid

    merged_nodes: List[TurnsNode] = []
    source_fragment_ids_by_gid: Dict[int, set[str]] = defaultdict(set)
    source_fragment_titles_by_gid: Dict[int, Dict[str, str]] = defaultdict(dict)
    source_fragment_summaries_by_gid: Dict[int, Dict[str, str]] = defaultdict(dict)
    for gid, grp in enumerate(merge_groups):
        nodes = [children[idx] for idx in grp]
        all_spans: List[TurnSpan] = []
        all_confirmed_memories: List[Dict[str, Any]] = []
        for n in nodes:
            all_spans.extend(n.spans or [])
            nid = str(n.fragment_id)
            source_fragment_ids_by_gid[gid].add(nid)
            if nid:
                source_fragment_titles_by_gid[gid][nid] = str(n.title or "").strip()
                source_fragment_summaries_by_gid[gid][nid] = str(n.summary or "").strip()
            all_confirmed_memories.extend(list((n.meta or {}).get("confirmed_memories") or []))
        merged_spans = _merge_turn_spans(all_spans)
        merged_time_range = _turn_time_range_from_spans(dialogue, merged_spans)
        points = [m for m in all_confirmed_memories if _lane_from_memory_dict(m, children) == MEMORY_LANE_GENERAL_QA]
        merged_points_block = _format_fragment_confirmed_memories_for_prompt(points)
        merged_toc = _build_general_qa_chapters_toc(nodes)
        merged_title, merged_summary = _llm_merged_general_title_summary(
            llm,
            merged_toc,
            merged_confirmed_memory_points=merged_points_block,
        )
        if not (merged_title or "").strip() or not (merged_summary or "").strip():
            merged_title, merged_summary = _llm_general_qa_fragment_title_and_summary_from_confirmed(
                llm,
                raw_text=_dialogue_raw_from_spans(dialogue, merged_spans),
                confirmed_memories=points,
                previous_toc_block=previous_toc_block,
            )
        merged_nodes.append(
            TurnsNode(
                chapter_id="",
                fragment_id="",
                title=(merged_title or "").strip() or "Merged General QA",
                confirmed_only=[],
                pending="",
                turn_time_range=merged_time_range,
                spans=merged_spans,
                summary=(merged_summary or "").strip(),
                meta={
                    "memory_lane": MEMORY_LANE_GENERAL_QA,
                    "consolidated_general_qa": True,
                    "merged_from_chapter_indices": grp,
                    "merged_from_fragment_indices": grp,
                    "merged_from_fragment_ids": sorted(source_fragment_ids_by_gid[gid]),
                    "source_fragment_ids": sorted(source_fragment_ids_by_gid[gid]),
                    "source_fragment_titles": dict(source_fragment_titles_by_gid[gid]),
                    "source_fragment_summaries": dict(source_fragment_summaries_by_gid[gid]),
                },
            )
        )

    out_children: List[TurnsNode] = []
    inserted_groups: set[int] = set()
    for idx, child in enumerate(children):
        gid = group_map.get(idx)
        if gid is None:
            out_children.append(child)
            continue
        if gid in inserted_groups:
            continue
        out_children.append(merged_nodes[gid])
        inserted_groups.add(gid)
    return out_children, _merge_confirmed_lists([], list(confirmed_list)), merge_groups


def _llm_merged_interaction_meta_title_summary(llm: LLMClient, toc_block: str) -> Tuple[str, str]:
    prompt = GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META.format(
        interaction_meta_chapters_toc=toc_block
    )
    for attempt in range(3):
        out = ""
        try:
            phase = (
                "Generate interaction_meta chapter title+summary (GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META)"
                if attempt == 0
                else f"Retry merge interaction_meta (attempt {attempt + 1}/3)"
            )
            log_llm_phase(phase, prompt)
            out = (
                llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or ""
            ).strip()
            if not out:
                _log_llm_retry_failure(
                    "GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META",
                    attempt,
                    3,
                    "empty_llm_body",
                )
                continue
            obj = parse_json_object(out)
            title = str(obj.get("title") or "").strip()
            summary = str(obj.get("summary") or "").strip()
            if title or summary:
                return title, summary
            keys = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            _log_llm_retry_failure(
                "GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META",
                attempt,
                3,
                "parsed_json_but_empty_title_and_summary",
                detail=f"top_level_keys={keys} raw_head={_snippet_for_retry_log(out)}",
            )
        except Exception as e:
            _log_llm_retry_failure(
                "GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META",
                attempt,
                3,
                "exception",
                detail=f"{type(e).__name__}: {e} | raw_head={_snippet_for_retry_log(out)}",
            )
            _log_retry_traceback("GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META")
            continue
    log_progress(
        "GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META: exhausted 3 attempts; returning empty title and summary"
    )
    return "", ""


def _replace_lane_fragments_with_merged(
    children: List[TurnsNode],
    lane_name: str,
    merged: TurnsNode,
) -> List[TurnsNode]:
    """去掉指定 lane 的全部碎片，追加一个 merged 节点。"""
    out: List[TurnsNode] = []
    for c in children:
        if normalize_memory_lane((c.meta or {}).get("memory_lane")) == lane_name:
            continue
        out.append(c)
    out.append(merged)
    return out


def _build_merged_general_turns_node(
    llm: LLMClient,
    general_nodes: List[TurnsNode],
    dialogue: List[dict],
    session_id: str,
    *,
    use_llm: bool = True,
) -> TurnsNode:
    """将本 session 内全部 general_qa 碎片折叠为一个 TurnsNode（ToC 级合并）。"""
    all_spans: List[TurnSpan] = []
    for n in general_nodes:
        for sp in n.spans or []:
            all_spans.append(sp)
    merged_spans = _merge_turn_spans(all_spans)
    start_turn = min((s.start_turn for s in merged_spans), default=0)
    end_turn = max((s.end_turn for s in merged_spans), default=0)
    mm = _turn_time_range_from_spans(dialogue, merged_spans)

    title, summary = "", ""
    if use_llm:
        toc = _build_general_qa_chapters_toc(general_nodes)
        title, summary = _llm_merged_general_title_summary(llm, toc)
    if not title:
        title = " | ".join(x.title for x in general_nodes[:3] if x.title) or "General Q&A"
    if not summary:
        summary = " | ".join(x.summary for x in general_nodes[:5] if x.summary) or ""
    meta = {
        "memory_lane": MEMORY_LANE_GENERAL_QA,
        "merged_general": True,
        "start_turn": start_turn,
        "end_turn": end_turn,
        "source_fragment_ids": [str(n.fragment_id) for n in general_nodes if str(n.fragment_id)],
    }
    return TurnsNode(
        chapter_id="",
        fragment_id="",
        title=title,
        confirmed_only=[],
        pending="",
        turn_time_range=mm,
        spans=merged_spans,
        summary=summary,
        meta=meta,
    )


def _build_merged_interaction_meta_turns_node(
    llm: LLMClient,
    meta_nodes: List[TurnsNode],
    dialogue: List[dict],
    session_id: str,
    *,
    use_llm: bool = True,
) -> TurnsNode:
    """将本 session 内全部 interaction_meta 碎片折叠为一个 TurnsNode（ToC 级合并）。"""
    all_spans: List[TurnSpan] = []
    for n in meta_nodes:
        for sp in n.spans or []:
            all_spans.append(sp)
    merged_spans = _merge_turn_spans(all_spans)
    start_turn = min((s.start_turn for s in merged_spans), default=0)
    end_turn = max((s.end_turn for s in merged_spans), default=0)
    mm = _turn_time_range_from_spans(dialogue, merged_spans)
    title, summary = "", ""
    if use_llm:
        toc = _build_interaction_meta_chapters_toc(meta_nodes)
        title, summary = _llm_merged_interaction_meta_title_summary(llm, toc)
    if not title:
        title = " | ".join(x.title for x in meta_nodes[:3] if x.title) or "Interaction Meta"
    if not summary:
        summary = " | ".join(x.summary for x in meta_nodes[:5] if x.summary) or ""
    meta = {
        "memory_lane": MEMORY_LANE_INTERACTION_META,
        "merged_interaction_meta": True,
        "start_turn": start_turn,
        "end_turn": end_turn,
        "source_fragment_ids": [str(n.fragment_id) for n in meta_nodes if str(n.fragment_id)],
    }
    return TurnsNode(
        chapter_id="",
        fragment_id="",
        title=title,
        confirmed_only=[],
        pending="",
        turn_time_range=mm,
        spans=merged_spans,
        summary=summary,
        meta=meta,
    )


def _reorder_children_by_lane(children: List[TurnsNode]) -> List[TurnsNode]:
    """统一 children 顺序：user_durable -> interaction_meta -> general_qa；同 lane 保持稳定顺序。"""
    lane_rank = {
        MEMORY_LANE_USER_DURABLE: 0,
        MEMORY_LANE_INTERACTION_META: 1,
        MEMORY_LANE_GENERAL_QA: 2,
    }
    return sorted(
        children,
        key=lambda c: lane_rank.get(
            normalize_memory_lane((c.meta or {}).get("memory_lane")),
            99,
        ),
    )
