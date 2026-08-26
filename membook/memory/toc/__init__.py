# -*- coding: utf-8 -*-
"""MemBook full ToC builder (aligned with HaluMem MemBook eval pipeline)."""

from .chapter_splitter import (
    MEMORY_LANE_GENERAL_QA,
    MEMORY_LANE_INTERACTION_META,
    MEMORY_LANE_USER_DURABLE,
    normalize_memory_lane,
    _format_turn,
)
from .json_llm import LLMClient, parse_json_object
from .policy import GeneralQaMode, normalize_general_qa_mode, should_skip_non_durable_title_summary
from .schemas import (
    ConfirmedMemoryItem,
    SessionNode,
    TimeValue,
    TocTree,
    TurnSpan,
    TurnsNode,
    TurnTimeRange,
    UserNode,
)
from .toc_builder import (
    ReconcileContextProvider,
    TocBuilder,
    format_prior_sessions_global_reconcile_context,
    format_prior_sessions_toc_block,
    format_session_time_hint,
)

__all__ = [
    "MEMORY_LANE_GENERAL_QA",
    "MEMORY_LANE_INTERACTION_META",
    "MEMORY_LANE_USER_DURABLE",
    "LLMClient",
    "ConfirmedMemoryItem",
    "GeneralQaMode",
    "SessionNode",
    "TimeValue",
    "ReconcileContextProvider",
    "TocBuilder",
    "TocTree",
    "TurnSpan",
    "TurnsNode",
    "TurnTimeRange",
    "UserNode",
    "_format_turn",
    "format_prior_sessions_global_reconcile_context",
    "format_prior_sessions_toc_block",
    "format_session_time_hint",
    "normalize_memory_lane",
    "normalize_general_qa_mode",
    "parse_json_object",
]
