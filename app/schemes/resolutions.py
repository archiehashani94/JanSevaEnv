"""
resolutions.py
Load and query resolution definitions from taxonomy.json.
taxonomy.json is the single source of truth — no hardcoded resolution logic here.
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


def get_all_resolutions() -> list[dict]:
    """Return full list of resolution objects."""
    return _load_taxonomy()["resolutions"]


def get_resolution(resolution_id: str) -> Optional[dict]:
    """Return a single resolution dict by ID, or None if not found."""
    for res in get_all_resolutions():
        if res["id"] == resolution_id:
            return res
    return None


def get_resolution_ids() -> list[str]:
    """Return list of all resolution IDs."""
    return [r["id"] for r in get_all_resolutions()]


def get_resolution_for_cause(cause_id: str) -> Optional[dict]:
    """Return the resolution linked to a cause via taxonomy.json."""
    causes = _load_taxonomy()["causes"]
    for cause in causes:
        if cause["id"] == cause_id:
            return get_resolution(cause["resolution_id"])
    return None


def get_resolution_steps(resolution_id: str) -> list[str]:
    """Return ordered resolution steps for a given resolution ID."""
    res = get_resolution(resolution_id)
    if res is None:
        return []
    return res.get("steps", [])


def get_resolution_authority(resolution_id: str) -> str:
    """Return the authority responsible for this resolution."""
    res = get_resolution(resolution_id)
    return res.get("authority", "Unknown") if res else "Unknown"


def get_expected_days(resolution_id: str) -> int:
    """Return expected resolution time in working days."""
    res = get_resolution(resolution_id)
    return res.get("expected_resolution_days", 0) if res else 0
