# -*- coding: utf-8 -*-
"""Round-trip between HaluMem-style dialogue list and legacy single-string memorize() format."""

from __future__ import annotations

import re
from typing import Any, Dict, List

_LINE = re.compile(
    r"^(?:\[(?P<ts>[^\]]*)\])?(?P<role>[^:]+):\s*(?P<content>.*)$",
    re.MULTILINE,
)


def dialogue_from_gam_text(message: str) -> List[Dict[str, Any]]:
    """
    Inverse of eval/halumem/eval_gam.format_dialogue_to_text: one turn per non-empty line.
    Multiline content: not supported (same limitation as line-based format).
    """
    out: List[Dict[str, Any]] = []
    for raw in (message or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        role = (m.group("role") or "unknown").strip()
        content = (m.group("content") or "").strip()
        ts = (m.group("ts") or "").strip()
        turn: Dict[str, Any] = {"role": role, "content": content}
        if ts:
            turn["timestamp"] = ts
        out.append(turn)
    return out
