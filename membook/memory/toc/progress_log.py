# -*- coding: utf-8 -*-
"""Optional build progress logs (English).

Enabled when any of:
  MEMBOOK_LIVE_LLM=1   (live tree demo / real API — prints by default)
  MEMBOOK_BUILD_VERBOSE=1  (verbose without live)

LoCoMo batch eval (``eval/membook/locomo_membook_test.py``) sets ``MEMBOOK_LOCOMO_QUIET_BUILD=1`` by default so
``log_progress_always`` is suppressed unless ``MEMBOOK_BUILD_VERBOSE`` or ``MEMBOOK_LIVE_LLM`` is on.
Unset ``MEMBOOK_LOCOMO_QUIET_BUILD`` or set to ``0``/``false`` to restore always-on stage banners outside that entry.

LLM retry diagnostics (see toc_builder ``_log_llm_retry_failure``): same visibility as above.
Set MEMBOOK_RETRY_TRACEBACK=1 to append a short traceback tail after each logged exception retry.

Context caps (see toc_builder ``_shared_truncation_char_cap``): if ``MEMBOOK_SESSION_SUMMARY_MAX_CHARS``
/ ``MEMBOOK_RECONCILE_MAX_CHARS`` are unset, a char budget is derived from
``MEMBOOK_CONTEXT_WINDOW_TOKENS`` (default 128000, gpt-4o / gpt-4o-mini class) minus
``MEMBOOK_LLM_RESERVE_TOKENS`` (default 36000, completion + template + other fields), then
``min(MEMBOOK_SESSION_SUMMARY_BUDGET_TOKENS, remainder//6) * MEMBOOK_CHARS_PER_TOKEN_ESTIMATE``
(default budget 7000 tokens, 4 chars/token ~ historical 28k chars). Explicit
``MEMBOOK_SESSION_SUMMARY_MAX_CHARS`` still overrides that for rolling summary; ``MEMBOOK_PRIOR_SESSIONS_TITLE_SUMMARIES_MAX_CHARS``
overrides prior-only when set, else prior uses the same cap as rolling summary.
"""

from __future__ import annotations

from membook.compat import env

import os


def verbose_enabled() -> bool:
    v = env("MEMBOOK_BUILD_VERBOSE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def build_console_enabled() -> bool:
    """Console progress for live runs or explicit verbose."""
    if env("MEMBOOK_LIVE_LLM", "").strip() == "1":
        return True
    return verbose_enabled()


def _locomo_quiet_build() -> bool:
    v = (env("MEMBOOK_LOCOMO_QUIET_BUILD") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def log_progress(message: str) -> None:
    if not build_console_enabled():
        return
    print(f"[MemBook] {message}", flush=True)


def log_progress_always(message: str) -> None:
    """Stage markers; normally always printed.

    When ``MEMBOOK_LOCOMO_QUIET_BUILD`` is enabled (LoCoMo entry default), suppress unless the user
    turned on ``MEMBOOK_BUILD_VERBOSE`` or ``MEMBOOK_LIVE_LLM``.
    """
    if _locomo_quiet_build() and not build_console_enabled():
        return
    print(f"[MemBook] {message}", flush=True)


def log_llm_phase(phase: str, prompt: str) -> None:
    if not build_console_enabled():
        return
    prompt_chars = len(prompt or "")
    estimated_tokens = prompt_chars // 4
    print(f"[MemBook] {phase} (~{estimated_tokens} tokens)", flush=True)
