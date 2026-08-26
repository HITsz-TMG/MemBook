# -*- coding: utf-8 -*-
"""Reconcile-time cross-session context: gate LLM + format retrieved memory points."""

from __future__ import annotations

from membook.compat import env

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from membook.memory.toc.json_llm import LLMClient, parse_json_object
from membook.prompts.memory_prompts import RECONCILE_CONTEXT_GATE_AND_REQUESTS
from membook.schemas import Hit


def format_prior_confirmed_snapshot_for_gate(items: List[Any]) -> str:
    """Compact text for gate prompt from in-session ConfirmedMemoryItem list."""
    if not items:
        return "(none)"
    parts: List[str] = []
    for it in items:
        t = str(getattr(it, "text", "") or "").strip()
        if t:
            parts.append(f"- {t}")
    return "\n".join(parts) if parts else "(none)"


def _toc_json_object_enabled() -> bool:
    v = env("MEMBOOK_TOC_JSON_OBJECT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def run_reconcile_context_gate(
    llm: LLMClient,
    *,
    indexed_dialogue_zone: str,
    time_hint: str,
    prior_confirmed: List[Any],
) -> List[str]:
    """Single LLM call: retrieval questions; empty list means no cross-session context needed."""
    prompt = RECONCILE_CONTEXT_GATE_AND_REQUESTS.format(
        indexed_dialogue_zone=indexed_dialogue_zone,
        prior_confirmed_in_session_snapshot=format_prior_confirmed_snapshot_for_gate(prior_confirmed),
        time_hint=(time_hint or "").strip() or "(unknown)",
    )
    out = (llm.complete(prompt, response_json_object=_toc_json_object_enabled()) or "").strip()
    return parse_reconcile_gate_and_requests(out)


def parse_reconcile_gate_and_requests(text: str) -> List[str]:
    """Parse JSON from RECONCILE_CONTEXT_GATE_AND_REQUESTS (only ``requests`` is used)."""
    reqs: List[str] = []
    try:
        obj = parse_json_object(text or "")
        raw = obj.get("requests")
        if isinstance(raw, list):
            for x in raw:
                s = str(x).strip()
                if s:
                    reqs.append(s)
    except Exception:
        return []
    return reqs[:5]


_MP_DATE_PREFIX_RE = re.compile(
    r"^\s*\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\s*:\s*",
    re.IGNORECASE,
)


def _strip_mp_date_prefix(snippet: str) -> str:
    """Remove leading YYYY/MM/DD: or YYYY-MM-DD: from memory-point snippet text."""
    text = (snippet or "").strip()
    if not text:
        return ""
    return _MP_DATE_PREFIX_RE.sub("", text, count=1).strip() or text


def _session_sort_key(hs: List[Hit]) -> str:
    return max((str((x.meta or {}).get("timestamp_start") or "") for x in hs), default="")


def format_integrate_evidence_block(hits: List[Hit]) -> str:
    """
    Format QA Integrate EVIDENCE_CONTEXT: session title line, then bullet memory-point lines
    (``- `` + snippet body with leading YYYY/MM/DD: or YYYY-MM-DD: stripped; temporal anchor
    is the session title line); sessions newest-first; blank line between session blocks.
    """
    if not hits:
        return "(none)"

    by_sid: Dict[str, List[Hit]] = defaultdict(list)
    unscoped: List[Hit] = []
    for h in hits:
        sid = str((h.meta or {}).get("session_id") or "").strip()
        if sid:
            by_sid[sid].append(h)
        else:
            unscoped.append(h)

    blocks: List[str] = []
    sessions: List[Tuple[str, str, List[Hit]]] = []
    for sid, hs in by_sid.items():
        sessions.append((_session_sort_key(hs), sid, hs))
    sessions.sort(key=lambda x: x[0], reverse=True)

    for _max_ts, sid, hs in sessions:
        title = str((hs[0].meta or {}).get("session_title") or "").strip()
        if not title:
            title = sid or "Untitled Session"
        lines: List[str] = [title]
        hs_sorted = sorted(
            hs,
            key=lambda x: str((x.meta or {}).get("timestamp_start") or ""),
            reverse=True,
        )
        for h in hs_sorted:
            body = _strip_mp_date_prefix((h.snippet or "").strip())
            if body:
                lines.append(f"- {body}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))

    if unscoped:
        lines = ["Untitled"]
        for h in unscoped:
            body = _strip_mp_date_prefix((h.snippet or "").strip())
            if body:
                lines.append(f"- {body}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else "(none)"


def format_grouped_memory_points_block(hits: List[Hit]) -> str:
    """
    Group point hits by session; sessions newest-first; points per session newest-first.
    """
    if not hits:
        return "(none)"
    by_sid: Dict[str, List[Hit]] = defaultdict(list)
    for h in hits:
        sid = str((h.meta or {}).get("session_id") or "").strip()
        if sid:
            by_sid[sid].append(h)

    sessions: List[Tuple[str, str, List[Hit]]] = []
    for sid, hs in by_sid.items():
        max_ts = max((str((x.meta or {}).get("timestamp_start") or "") for x in hs), default="")
        sessions.append((max_ts, sid, hs))
    sessions.sort(key=lambda x: x[0], reverse=True)

    out_lines: List[str] = []
    for _max_ts, sid, hs in sessions:
        title = str((hs[0].meta or {}).get("session_title") or "").strip() or "Untitled Session"
        ts0 = str((hs[0].meta or {}).get("timestamp_start") or "")
        day = ""
        if ts0:
            day = ts0.split("T", 1)[0].strip() if "T" in ts0 else ts0[:10]
        if not day:
            day = "unknown-date"
        out_lines.append(f"[{day}] {title}")
        hs_sorted = sorted(
            hs,
            key=lambda x: str((x.meta or {}).get("timestamp_start") or ""),
            reverse=True,
        )
        for h in hs_sorted:
            snippet = (h.snippet or "").strip()
            if snippet:
                out_lines.append(f"- {snippet}")
    return "\n".join(out_lines) if out_lines else "(none)"
