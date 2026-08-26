# -*- coding: utf-8 -*-
"""章节划分：LLMChapterSplitter（话题切换）"""

from __future__ import annotations
from typing import Any, List, Tuple, Dict
import re

from .json_llm import LLMClient, parse_json_object
from .progress_log import log_llm_phase, log_progress

# Legacy split prompt (no longer in prompts.py). Production path: TocBuilder + DETECT_TOPIC_SHIFTS.
_LEGACY_SPLIT_DIALOGUE_TO_FRAGMENTS = """\
Split INDEXED_DIALOGUE into topical fragments. Return ONLY JSON:
{{"fragments":[{{"start_turn":<int>,"end_turn":<int>,"title":<string>,"memory_lane":<optional string>}}]}}

INDEXED_DIALOGUE:
{indexed_dialogue}
"""

MEMORY_LANE_USER_DURABLE = "user_durable"
MEMORY_LANE_INTERACTION_META = "interaction_meta"
MEMORY_LANE_GENERAL_QA = "general_qa"
VALID_MEMORY_LANES = frozenset(
    {MEMORY_LANE_USER_DURABLE, MEMORY_LANE_INTERACTION_META, MEMORY_LANE_GENERAL_QA}
)


def normalize_memory_lane(raw: Any) -> str:
    """未知或非枚举值保守归为 general_qa，避免把噪声标成用户记忆。"""
    s = str(raw or "").strip().lower().replace("-", "_")
    if s in VALID_MEMORY_LANES:
        return s
    return MEMORY_LANE_GENERAL_QA


def merge_memory_lanes(a: str, b: str) -> str:
    """合并相邻同标题 fragment 的 lane：优先保留更偏用户持久记忆的标签。"""
    priority = {MEMORY_LANE_USER_DURABLE: 0, MEMORY_LANE_INTERACTION_META: 1, MEMORY_LANE_GENERAL_QA: 2}
    pa, pb = priority.get(a, 99), priority.get(b, 99)
    return a if pa <= pb else b


def _format_turn(turn: dict) -> str:
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    ts = turn.get("timestamp", "")
    return f"[{ts}]{role}: {content}" if ts else f"{role}: {content}"


def _indexed_dialogue(dialogue: List[dict]) -> str:
    lines = []
    for i, turn in enumerate(dialogue):
        lines.append(f"{i}. {_format_turn(turn)}")
    return "\n".join(lines)


class LLMChapterSplitter:
    """
    通过 LLM 将对话切成 topical fragments（连续 turn 段）。

    注：主流程已改用 ``TocBuilder`` + ``DETECT_TOPIC_SHIFTS`` 做话题切换切分，此类仅保留给旧测试与兼容。

    输出每段边界：(start_turn, end_turn, raw_text, title, memory_lane)
    - start_turn inclusive
    - end_turn exclusive
    """

    def __init__(self, llm: LLMClient, min_turns_per_chapter: int = 2):
        self.llm = llm
        self.min_turns_per_chapter = min_turns_per_chapter

    def split(self, dialogue: List[dict]) -> List[Tuple[int, int, str, str, str]]:
        if not dialogue:
            log_progress("Split: empty dialogue, 0 fragments")
            return []

        prompt = _LEGACY_SPLIT_DIALOGUE_TO_FRAGMENTS.format(
            indexed_dialogue=_indexed_dialogue(dialogue),
        )
        log_llm_phase("Splitting dialogue into topical fragments (legacy inline prompt)", prompt)
        obj = parse_json_object(self.llm.complete(prompt))
        # 兼容旧键名 "chapters"
        frags: List[Dict[str, Any]] = obj.get("fragments") or obj.get("chapters") or []
        if not frags:
            raise ValueError("LLM returned empty fragments.")

        n = len(dialogue)
        out = _normalize_chapters(frags, dialogue)

        # 覆盖性校验：无缝覆盖 0..n
        if out[0][0] != 0 or out[-1][1] != n:
            raise ValueError("Fragments must cover all turns from 0 to N.")
        for i in range(1, len(out)):
            if out[i - 1][1] != out[i][0]:
                raise ValueError("Fragments must not overlap or gap.")

        out = _refine_by_user_question_anchors(out, dialogue, self.min_turns_per_chapter)
        log_progress(f"Split done: {len(out)} fragment(s), covering {len(dialogue)} turn(s)")
        return out


def _normalize_chapters(
    chapters: List[Dict[str, Any]], dialogue: List[dict]
) -> List[Tuple[int, int, str, str, str]]:
    """对 LLM 章节边界做容错清洗，避免轻微越界导致整次构建失败。"""
    n = len(dialogue)
    cleaned: List[Tuple[int, int, str, str]] = []
    for ch in chapters:
        start = int(ch.get("start_turn", 0))
        end = int(ch.get("end_turn", 0))
        title = str(ch.get("title") or "")
        lane = normalize_memory_lane(ch.get("memory_lane"))
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end <= start:
            continue
        cleaned.append((start, end, title, lane))

    if not cleaned:
        raise ValueError(f"No valid chapters after normalization. n={n}")

    cleaned.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, int, str, str, str]] = []
    cursor = 0
    for start, end, title, lane in cleaned:
        if cursor >= n:
            break
        if start > cursor:
            gap_raw = "\n".join(_format_turn(dialogue[k]) for k in range(cursor, start))
            out.append((cursor, start, gap_raw, "Untitled", MEMORY_LANE_GENERAL_QA))
            cursor = start
        if end <= cursor:
            continue
        fixed_start = max(start, cursor)
        fixed_end = min(end, n)
        raw_text = "\n".join(_format_turn(dialogue[k]) for k in range(fixed_start, fixed_end))
        out.append((fixed_start, fixed_end, raw_text, title or "Untitled", lane))
        cursor = fixed_end

    if cursor < n:
        tail_raw = "\n".join(_format_turn(dialogue[k]) for k in range(cursor, n))
        out.append((cursor, n, tail_raw, "Untitled", MEMORY_LANE_GENERAL_QA))

    merged: List[Tuple[int, int, str, str, str]] = []
    for start, end, raw_text, title, lane in out:
        if merged and merged[-1][1] == start and merged[-1][3] == title:
            ps, pe, pr, pt, pl = merged[-1]
            merged_lane = merge_memory_lanes(pl, lane)
            merged[-1] = (
                ps,
                end,
                f"{pr}\n{raw_text}" if pr and raw_text else pr or raw_text,
                pt,
                merged_lane,
            )
        else:
            merged.append((start, end, raw_text, title, lane))
    return merged


def _refine_by_user_question_anchors(
    chapters: List[Tuple[int, int, str, str, str]],
    dialogue: List[dict],
    min_turns_per_chapter: int,
) -> List[Tuple[int, int, str, str, str]]:
    """
    对“用户问题密集会话”做轻量后处理：
    当用户在章节内部提出新的问题/请求时，优先在该 turn 处切分。
    """
    n = len(dialogue)
    if n == 0:
        return chapters

    anchor_idxs = [i for i, turn in enumerate(dialogue) if _is_user_question_turn(turn)]
    if not anchor_idxs:
        return chapters

    refined: List[Tuple[int, int, str, str, str]] = []
    for start, end, _, title, lane in chapters:
        split_points = [
            i
            for i in anchor_idxs
            if start < i < end and (i - start) >= min_turns_per_chapter and (end - i) >= min_turns_per_chapter
        ]
        if not split_points:
            raw_text = "\n".join(_format_turn(dialogue[k]) for k in range(start, end))
            refined.append((start, end, raw_text, title, lane))
            continue

        cuts = [start] + split_points + [end]
        for a, b in zip(cuts[:-1], cuts[1:]):
            raw_text = "\n".join(_format_turn(dialogue[k]) for k in range(a, b))
            refined.append((a, b, raw_text, title, lane))

    merged: List[Tuple[int, int, str, str, str]] = []
    for start, end, raw_text, title, lane in refined:
        if merged and (end - start) < min_turns_per_chapter:
            ps, pe, pr, pt, pl = merged[-1]
            merged[-1] = (
                ps,
                end,
                f"{pr}\n{raw_text}" if pr and raw_text else pr or raw_text,
                pt,
                merge_memory_lanes(pl, lane),
            )
        else:
            merged.append((start, end, raw_text, title, lane))
    return merged


def _is_user_question_turn(turn: dict) -> bool:
    if str(turn.get("role", "")).lower() != "user":
        return False
    text = str(turn.get("content") or "")
    low = text.lower()
    ask_signal = ("?" in text) or bool(
        re.search(r"\b(find|solve|prove|evaluate|show|help|how do we|how to|what is|could you|can you)\b", low)
    )
    return ask_signal


def _merge_short_assistant_only_interruption(
    chapters: List[Tuple[int, int, str, str, str]],
    dialogue: List[dict],
    max_turns: int = 2,
) -> List[Tuple[int, int, str, str, str]]:
    """
    合并短暂“纯 assistant 噪声段”到前一个章节：
    - 这些段通常不改变用户的主问题/主主题
    - 通过结构判断（该段内没有 user turn），避免做 story/math 特判
    """
    if len(chapters) <= 2:
        return chapters

    merged: List[Tuple[int, int, str, str, str]] = []
    for idx, (start, end, raw_text, title, lane) in enumerate(chapters):
        if (
            merged
            and idx >= 1
            and (end - start) <= max_turns
            and all(str(dialogue[k].get("role", "")).lower() != "user" for k in range(start, end))
        ):
            ps, pe, pr, pt, pl = merged[-1]
            merged[-1] = (
                ps,
                end,
                f"{pr}\n{raw_text}" if pr and raw_text else pr or raw_text,
                pt,
                merge_memory_lanes(pl, lane),
            )
        else:
            merged.append((start, end, raw_text, title, lane))
    return merged
