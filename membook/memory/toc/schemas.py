# -*- coding: utf-8 -*-
"""ToC 树状记忆 Schema"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _coerce_pending_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if str(x).strip()]
        return ". ".join(parts) if parts else ""
    return " ".join(str(v).split()) if str(v).strip() else ""


class TimeValue(BaseModel):
    """既可读又可筛选：原始文本 + ISO + Unix 秒"""

    raw: str = Field(default="", description="原始时间字符串（HaluMem 格式）")
    iso: str = Field(default="", description="ISO8601 字符串，便于展示/跨系统")
    unix_sec: Optional[int] = Field(default=None, description="Unix 时间戳秒，便于筛选排序")


class TurnTimeRange(BaseModel):
    """turn 级时间区间（来自 HaluMem dialogue 的 timestamp 字符串）"""

    start: TimeValue = Field(default_factory=TimeValue)
    end: TimeValue = Field(default_factory=TimeValue)


class TurnSpan(BaseModel):
    """turn 索引范围（用于支持 chapter 的多段非连续 spans）"""
    start_turn: int = Field(default=0, description="inclusive")
    end_turn: int = Field(default=0, description="exclusive")


class ConfirmedMemoryItem(BaseModel):
    """会话级已确认记忆点；显式三层 ID：memory_point/source_fragment/chapter。"""

    text: str = Field(..., description="一句可确认的用户事实")
    memory_point_id: str = Field(
        default="",
        description="会话内短编号 mp_n；落盘为全局唯一 {session_id}/mp_n",
    )
    source_turn_start: int = Field(default=0, description="出处 turn 起点（含）")
    source_turn_end: int = Field(default=0, description="出处 turn 终点（不含），半开区间")
    timestamp_start: str = Field(default="", description="来自 dialogue[start].timestamp")
    timestamp_end: str = Field(default="", description="来自 dialogue[end-1].timestamp")
    fragment_id: str = Field(default="", description="来源 source fragment 编号，如 frag_3（审计追溯）")
    chapter_id: str = Field(default="", description="归属 chapter 编号，如 chap_1")
    binding_ambiguous: bool = Field(
        default=False,
        description="多个 fragment 的 span 与出处 turn 相交时为 True",
    )


class TurnsNode(BaseModel):
    """由若干 turns 组成的章节节点（Branch 层）"""
    chapter_id: str = Field(..., description="章节唯一 ID")
    fragment_id: str = Field(default="", description="首个 source fragment 编号（pre-merge，审计追溯）")
    title: str = Field(..., description="章节标题")
    entity_tags: List[str] = Field(default_factory=list)
    confirmed_only: List[str] = Field(
        default_factory=list,
        description="用户明确确认过的事实（当前 Session reconcile 不写入，保留字段供其它流程）",
    )
    pending: str = Field(
        default="",
        description="章级兼容字段（单段文本，与 summary 同构）；Session reconcile 后一般为空串，会话级见 SessionNode.session_pending",
    )

    @field_validator("pending", mode="before")
    @classmethod
    def _pending_coerce(cls, v: Any) -> str:
        return _coerce_pending_text(v)
    page_ids: List[int] = Field(default_factory=list)
    turn_time_range: Optional[TurnTimeRange] = Field(default=None)
    spans: List[TurnSpan] = Field(default_factory=list, description="chapter 覆盖的 turn 索引 spans（可非连续）")
    summary: str = Field(default="")
    meta: Dict[str, Any] = Field(default_factory=dict)


class SessionNode(BaseModel):
    """Session 节点（User 下一级）"""
    session_id: str = Field(...)
    user_id: str = Field(default="", description="用户标识")
    title: str = Field(default="", description="整段 session 的短标题（LLM，仅用户轨）")
    total_turns: int = Field(default=0, description="该 session 的总 turn 数")
    total_processing_ms: int = Field(default=0, description="构建该 session ToC 的总处理耗时（毫秒）")
    time_span: Optional[TurnTimeRange] = Field(default=None)
    summary: str = Field(default="", description="整段 session 的摘要（LLM，仅用户轨；不含 general_qa）")
    session_pending: str = Field(
        default="",
        description="会话级未确认命题（单段；句间建议 '. '；Session reconcile 产出，与 summary 同构）",
    )

    @field_validator("session_pending", mode="before")
    @classmethod
    def _session_pending_coerce(cls, v: Any) -> str:
        return _coerce_pending_text(v)
    confirmed_list: List[ConfirmedMemoryItem] = Field(
        default_factory=list,
        description="本会话已确认记忆点（扁平）；与各 Chapter meta.confirmed_memories 双写一致",
    )
    general_confirmed_list: List[ConfirmedMemoryItem] = Field(
        default_factory=list,
        description="本会话 general_qa 已确认记忆点（扁平）；仅在 general_qa full pipeline 时填充",
    )
    confirmed_memory_count: int = Field(
        default=0,
        description="本会话已确认记忆点数量（通常等于 len(confirmed_list)）",
    )
    children: List[TurnsNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class UserNode(BaseModel):
    """User 根节点（Root 层）"""

    user_id: str = Field(...)
    uuid: str = Field(default="")
    persona_info: str = Field(default="")
    sessions: List[SessionNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class TocTree(BaseModel):
    """完整 ToC 树（多 User）"""
    users: List[UserNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
