"""
task3.py — Hard: MGNREGA Wages Not Received

Scheme: MGNREGA
Difficulty: Hard
Max steps: 20
Cases: 5

The agent must identify why a MGNREGA labourer's wages were not credited.
The MGNREGA payment chain has multiple failure points (work measurement ->
muster roll -> FTO generation -> bank transfer), creating a long causal chain.
Some cases have misleading surface signals (e.g., FTO generated, bank appears
active) but the root cause is buried deeper (Aadhaar inactive, IFSC changed).
The agent needs 5-10 systematic questions covering the entire payment pipeline.
"""

from __future__ import annotations

from app.rewards.reward_fn import compute_final_score
from app.schemes.root_causes import get_signal_questions

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_ID = "task3"
TASK_NAME = "MGNREGA Wages Not Received"
TASK_DESCRIPTION = (
    "A MGNREGA labourer reports that their wages have not been credited despite "
    "completing assigned work. The MGNREGA payment pipeline has several potential "
    "failure points: work completion, technical measurement, muster roll entry, "
    "FTO generation by the Programme Officer, and final bank transfer. "
    "Some cases include deliberately misleading signals - e.g., the FTO is generated "
    "but Aadhaar is deactivated, or the bank account appears active but the IFSC "
    "changed after a branch merger. The agent must systematically probe the entire "
    "payment chain to locate the true root cause. Frontier models score ~0.6-0.75 "
    "on this task; random agents score ~0.0-0.1."
)
DIFFICULTY = "hard"
SCHEME = "MGNREGA"
MAX_STEPS = 20
NUM_CASES = 5

# Causes that appear in task3 cases (from cases.json)
TASK_CAUSES = [
    "fto_generation_pending",
    "muster_roll_error",
    "work_not_measured",
    "aadhaar_inactive",
    "rejection_by_bank",
]

# The payment pipeline in order - useful for structured investigation
MGNREGA_PIPELINE_STAGES = [
    {"stage": 1, "name": "Work Approval",      "signal_questions": ["Q46"]},
    {"stage": 2, "name": "Work Measurement",   "signal_questions": ["Q30"]},
    {"stage": 3, "name": "Muster Roll Entry",  "signal_questions": ["Q31"]},
    {"stage": 4, "name": "FTO Generation",     "signal_questions": ["Q32"]},
    {"stage": 5, "name": "Bank Transfer",      "signal_questions": ["Q33", "Q03", "Q08"]},
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
    Grade an agent's performance on a task3 case.

    Hard difficulty notes:
      - 20-step budget means efficiency bonus requires completion in <=10 steps
        for the full 0.20 bonus, scaling linearly to 0 at step 20.
      - Misleading signals (e.g., bank active but IFSC wrong) mean an agent
        that naively stops at surface level will get cause wrong despite
        asking relevant questions -> partial credit 0.10.
      - Systematic pipeline traversal (ask Q46->Q30->Q31->Q32->Q33) is the
        intended strategy and earns signal bonuses at each stage.

    Correct cause + resolution + efficient + signal Q -> up to 1.0
    Correct cause + resolution, late/no signal        -> 0.70
    Correct cause only                                -> 0.50-0.55
    Same MGNREGA category, wrong cause                -> 0.20
    Asked signal Q but wrong conclusion               -> 0.10
    Completely off (e.g., diagnosed KYC issue)        -> 0.00

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
        "payment_pipeline": MGNREGA_PIPELINE_STAGES,
    }


def get_min_steps_hint() -> int:
    """
    Minimum questions for reliable task3 diagnosis.
    The MGNREGA pipeline has 5 stages; a systematic agent needs at least 5 questions.
    """
    return 5


def get_signal_coverage() -> dict[str, list[str]]:
    """Return signal questions for each cause in this task."""
    return {cause: get_signal_questions(cause) for cause in TASK_CAUSES}
