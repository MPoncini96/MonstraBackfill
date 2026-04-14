"""
Base bot_id (no _alpha1 / _alpha2 / _beta1 suffix) + bot_type for composite keys.

Used by the worker, DB layer, and migration script.
"""

from __future__ import annotations

import re

BOT_TYPE_ALPHA1 = "alpha1"
BOT_TYPE_ALPHA2 = "alpha2"
BOT_TYPE_BETA1 = "beta1"
BOT_TYPE_GAMMA1 = "gamma1"

_SUFFIX_ALPHA1 = re.compile(r"_alpha1$", re.IGNORECASE)
_SUFFIX_ALPHA2 = re.compile(r"_alpha2$", re.IGNORECASE)
_SUFFIX_BETA1 = re.compile(r"_beta1$", re.IGNORECASE)
_SUFFIX_GAMMA1 = re.compile(r"_gamma1$", re.IGNORECASE)


def strip_bot_type_suffix(raw_id: str) -> str:
    """Remove trailing _alpha1 / _alpha2 / _beta1 / _gamma1 from a stored id."""
    s = raw_id.strip()
    s = _SUFFIX_ALPHA1.sub("", s)
    s = _SUFFIX_ALPHA2.sub("", s)
    s = _SUFFIX_BETA1.sub("", s)
    s = _SUFFIX_GAMMA1.sub("", s)
    return s


def infer_bot_type_from_suffix(raw_id: str) -> str:
    """Infer strategy from legacy suffixed id; default alpha1 if no suffix."""
    s = raw_id.strip().lower()
    if s.endswith("_alpha2"):
        return BOT_TYPE_ALPHA2
    if s.endswith("_beta1"):
        return BOT_TYPE_BETA1
    if s.endswith("_gamma1"):
        return BOT_TYPE_GAMMA1
    if s.endswith("_alpha1"):
        return BOT_TYPE_ALPHA1
    return BOT_TYPE_ALPHA1
