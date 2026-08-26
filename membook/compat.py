# -*- coding: utf-8 -*-
"""Compatibility helpers for the MemBook rename (MYMEM_* / import gam).

Prefer ``MEMBOOK_*`` environment variables. Reading ``MYMEM_*`` still works
for one transitional release and emits a ``DeprecationWarning``.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional


def _normalize_suffix(name: str) -> str:
    """Accept ``FOO``, ``MEMBOOK_FOO``, or ``MYMEM_FOO`` → ``FOO``."""
    if name.startswith("MEMBOOK_"):
        return name[len("MEMBOOK_") :]
    if name.startswith("MYMEM_"):
        return name[len("MYMEM_") :]
    return name


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read ``MEMBOOK_{name}``, falling back to ``MYMEM_{name}``.

    ``name`` may be a bare suffix (``USE_BM25``) or a full key
    (``MEMBOOK_USE_BM25`` / ``MYMEM_USE_BM25``).
    """
    suffix = _normalize_suffix(name)
    new_key = f"MEMBOOK_{suffix}"
    old_key = f"MYMEM_{suffix}"

    if new_key in os.environ:
        return os.environ.get(new_key)
    if old_key in os.environ:
        warnings.warn(
            f"Environment variable {old_key} is deprecated; use {new_key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return os.environ.get(old_key)
    return default
