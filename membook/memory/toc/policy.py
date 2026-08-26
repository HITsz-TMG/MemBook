# -*- coding: utf-8 -*-
"""Policy knobs for MemBook ToC build/persist behavior."""

from __future__ import annotations

from typing import Literal

GeneralQaMode = Literal["off", "nominal", "full"]


def normalize_general_qa_mode(raw: str | None) -> GeneralQaMode:
    key = str(raw or "").strip().lower()
    if key in ("off", "disabled", "none", "no"):
        return "off"
    if key in ("full", "on", "enabled", "1", "true", "yes"):
        return "full"
    return "nominal"


def should_skip_non_durable_title_summary(lane: str, general_qa_mode: GeneralQaMode) -> bool:
    """Skip fragment/merge title+summary LLM for lanes that do not persist in nominal mode."""
    lane_norm = str(lane or "").strip().lower()
    if lane_norm == "user_durable":
        return False
    if lane_norm == "interaction_meta":
        return True
    if lane_norm == "general_qa":
        return general_qa_mode != "full"
    return False

