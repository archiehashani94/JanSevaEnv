"""
root_causes.py
Load and query root cause definitions from taxonomy.json.
taxonomy.json is the single source of truth — no hardcoded cause logic here.
"""

import json
from pathlib import Path
from typing import Optional

_TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy.json"
_taxonomy_cache: Optional[dict] = None


def _load_taxonomy() -> dict:
    global _taxonomy_cache
    if _taxonomy_cache is None:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            _taxonomy_cache = json.load(f)
    return _taxonomy_cache


def get_all_causes() -> list[dict]:
    """Return full list of cause objects."""
    return _load_taxonomy()["causes"]


def get_cause(cause_id: str) -> Optional[dict]:
    """Return a single cause dict by ID, or None if not found."""
    for cause in get_all_causes():
        if cause["id"] == cause_id:
            return cause
    return None


def get_cause_ids() -> list[str]:
    """Return list of all cause IDs."""
    return [c["id"] for c in get_all_causes()]


def get_causes_by_category(category: str) -> list[dict]:
    """Return all causes that belong to the given category."""
    return [c for c in get_all_causes() if c["category"] == category]


def get_causes_for_scheme(scheme: str) -> list[dict]:
    """Return causes applicable to a specific scheme."""
    return [c for c in get_all_causes() if scheme in c.get("applicable_schemes", [])]


def get_signal_questions(cause_id: str) -> list[str]:
    """Return signal question IDs for a cause (strongly indicative questions)."""
    cause = get_cause(cause_id)
    if cause is None:
        return []
    return cause.get("signal_questions", [])


def get_diagnostic_questions(cause_id: str) -> list[str]:
    """Return all diagnostic question IDs for a cause."""
    cause = get_cause(cause_id)
    if cause is None:
        return []
    return cause.get("diagnostic_questions", [])


def get_category(cause_id: str) -> Optional[str]:
    """Return the category of a cause."""
    cause = get_cause(cause_id)
    return cause["category"] if cause else None


def causes_same_category(cause_id_a: str, cause_id_b: str) -> bool:
    """Check if two causes belong to the same category."""
    return get_category(cause_id_a) == get_category(cause_id_b)
