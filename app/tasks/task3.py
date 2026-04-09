"""
task3.py  —  Hard Task: MGNREGA Wages Not Received
Scheme: MGNREGA | Max steps: 20 | Cases: 5
Root causes: fto_generation_pending, muster_roll_error, work_not_measured,
             aadhaar_inactive, rejection_by_bank

The MGNREGA payment pipeline has 5 stages:
  GP Approval → Work Measurement → Muster Roll → FTO Generation → Bank Transfer
Each stage can fail. Some cases have deliberate misleading signals.
"""

from app.schemes.root_causes import get_signal_questions, get_category


TASK_META = {
    "task_id": "task3",
    "name": "MGNREGA Wages & NFSA-PDS Grievances",
    "description": (
        "Hard multi-scheme cases covering MGNREGA wages (5-stage payment pipeline: "
        "GP Approval → Measurement → Muster Roll → FTO → Bank Transfer) and NFSA-PDS "
        "ration distribution failures. Cases include deliberate misleading signals — "
        "e.g. FTO generated but Aadhaar inactive, biometric failure vs dealer diversion, "
        "or portal sync errors masking eligibility issues."
    ),
    "difficulty": "hard",
    "scheme": "MGNREGA",
    "max_steps": 20,
    "num_cases": 11,
    "grader": "app.tasks.task3:grade",
    "has_grader": True,
}

TASK_CASES = ["T3_001", "T3_002", "T3_003", "T3_004", "T3_005",
              "T3_006", "T3_007", "T3_008", "T3_009", "T3_010", "T3_011"]


def grade(
    submitted_cause: str,
    submitted_resolution: str,
    questions_asked: list[str],
    steps_used: int,
    true_cause: str,
    true_resolution: str,
) -> float:
    """
    Rule-based grader for Task 3 (hard).

    Identical scoring formula — the difficulty comes from the case design,
    not a different scoring rule. Efficiency bonus still rewards fewer steps.
    """
    MAX_STEPS = TASK_META["max_steps"]
    asked_set = set(questions_asked)
    signal_qs = set(get_signal_questions(true_cause))
    asked_signal = bool(signal_qs & asked_set)

    cause_correct = submitted_cause == true_cause
    resolution_correct = submitted_resolution == true_resolution
    same_category = (
        get_category(submitted_cause) == get_category(true_cause)
        and submitted_cause != true_cause
    )

    if cause_correct and resolution_correct:
        efficiency = max(0.0, 1.0 - steps_used / MAX_STEPS) * 0.20
        return round(min(1.0, 0.70 + efficiency + (0.10 if asked_signal else 0.0)), 4)
    if cause_correct:
        return round(0.50 + (0.05 if asked_signal else 0.0), 4)
    if same_category:
        return 0.20
    if asked_signal:
        return 0.10
    return 0.0
