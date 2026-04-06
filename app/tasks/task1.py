"""
task1.py — Easy: PM-KISAN Payment Not Received

Scheme: PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)
Difficulty: Easy
Max steps: 10
Cases: 5

The agent must identify why a farmer is not receiving PM-KISAN installments.
Root causes are primarily from banking and KYC categories with clear signals.
The agent needs to ask 2–4 targeted questions to isolate the cause.
"""

from __future__ import annotations

from app.rewards.reward_fn import compute_final_score
from app.schemes.root_causes import get_signal_questions

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_ID = "task1"
TASK_NAME = "PM-KISAN Payment Not Received"
TASK_DESCRIPTION = (
    "An Indian farmer reports not receiving their PM-KISAN installment. "
    "The agent must ask targeted diagnostic questions to identify the root cause "
    "(e.g., Aadhaar not seeded, bank not linked, IFSC outdated) and suggest the "
    "correct resolution. Causes are straightforward with clear confirmation signals."
)
DIFFICULTY = "easy"
SCHEME = "PM-KISAN"
MAX_STEPS = 10
NUM_CASES = 5

# Causes that appear in task1 cases (from cases.json)
TASK_CAUSES = [
    "aadhaar_not_seeded",
    "bank_account_not_linked",
    "incorrect_bank_details",
    "land_records_mismatch",
    "npci_mapping_error",
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
    Grade an agent's performance on a task1 case.

    Scoring (inherits from compute_final_score):
      Correct cause + resolution + efficient + signal Q asked → 1.0
      Correct cause + resolution, no efficiency/signal        → 0.70
      Correct cause only                                      → 0.50–0.55
      Same category (banking/KYC), wrong cause                → 0.20
      Asked signal Q but wrong cause                          → 0.10
      Completely wrong                                        → 0.00

    Args:
        submitted_cause: Cause ID submitted by the agent.
        submitted_resolution: Resolution ID submitted by the agent.
        questions_asked: List of question IDs asked during the episode.
        steps_used: Number of steps taken.
        true_cause: Ground truth cause ID for the case.
        true_resolution: Ground truth resolution ID for the case.

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
    Minimum questions needed to reliably identify any task1 cause.
    Each cause has at least 1 signal question — a smart agent needs 1–3 questions.
    """
    return 1


def get_signal_coverage() -> dict[str, list[str]]:
    """Return signal questions for each cause in this task."""
    return {cause: get_signal_questions(cause) for cause in TASK_CAUSES}
