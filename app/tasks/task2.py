"""
task2.py — Medium: Old Age / Widow Pension Stopped

Scheme: OAP / WAP (Old Age Pension / Widow Pension)
Difficulty: Medium
Max steps: 15
Cases: 5

The agent must diagnose why an elderly or widow pensioner's monthly payment stopped.
Root causes span multiple categories (pension, banking, KYC, administrative) with some
misleading signals (e.g., beneficiary submitted life certificate but was marked deceased).
The agent needs 4–7 questions to reliably isolate the correct cause.
"""

from __future__ import annotations

from app.rewards.reward_fn import compute_final_score
from app.schemes.root_causes import get_signal_questions

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_ID = "task2"
TASK_NAME = "Old Age / Widow Pension Stopped"
TASK_DESCRIPTION = (
    "An elderly or widow beneficiary reports that their monthly pension under the "
    "Indira Gandhi National Old Age Pension Scheme (IGNOAPS) or Widow Pension Scheme "
    "has stopped without explanation. The agent must navigate multiple categories of "
    "possible causes — from missed life certificates and deceased flags to Aadhaar "
    "mismatches and district-level approval pending — and identify the correct one. "
    "Some cases contain misleading signals that require deeper investigation."
)
DIFFICULTY = "medium"
SCHEME = "OAP"  # Primary scheme; WAP also appears in cases
MAX_STEPS = 15
NUM_CASES = 5

# Causes that appear in task2 cases (from cases.json)
TASK_CAUSES = [
    "life_certificate_pending",
    "pensioner_marked_deceased",
    "aadhaar_mismatch",
    "bank_account_frozen",
    "district_level_pending",
]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(
    submitted_cause: str,
    submitted_resolution: str,
    questions_asked: list[str],
    steps_used: int,
    true_cause: str,
    true_resolution: str,
) -> float:
    """
    Grade an agent's performance on a task2 case.

    Medium difficulty scoring:
      - The efficiency window is wider (15 steps), so the bonus is harder to maximise.
      - Misleading signals mean an agent that asked the right signal question
        despite distractors earns the signal bonus even at more steps.

    Correct cause + resolution + efficient + signal Q -> up to 1.0
    Correct cause + resolution, no signal/efficiency  -> 0.70
    Correct cause only                                -> 0.50-0.55
    Same category, wrong cause                        -> 0.20
    Asked signal Q but wrong cause                    -> 0.10
    Completely wrong                                  -> 0.00

    Returns:
        Float in [0.0, 1.0].
    """
    return compute_final_score(
        submitted_cause=submitted_cause,
        submitted_resolution=submitted_resolution,
        questions_asked=questions_asked,
        steps_used=steps_used,
        max_steps=MAX_STEPS,
        true_cause=true_cause,
        true_resolution=true_resolution,
    )


def get_task_info() -> dict:
    """Return task metadata as a dict."""
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "description": TASK_DESCRIPTION,
        "difficulty": DIFFICULTY,
        "scheme": SCHEME,
        "max_steps": MAX_STEPS,
        "num_cases": NUM_CASES,
        "task_causes": TASK_CAUSES,
    }


def get_min_steps_hint() -> int:
    """
    Minimum questions needed to reliably identify any task2 cause.
    Medium cases need 3-5 questions to eliminate misleading signals.
    """
    return 3


def get_signal_coverage() -> dict[str, list[str]]:
    """Return signal questions for each cause in this task."""
    return {cause: get_signal_questions(cause) for cause in TASK_CAUSES}
