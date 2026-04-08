"""
task2.py  —  Medium Task: Old Age / Widow Pension Stopped
Scheme: OAP / WAP | Max steps: 15 | Cases: 5
Root causes: life_certificate_pending, pensioner_marked_deceased, aadhaar_mismatch,
             bank_account_frozen, district_level_pending
"""

from app.schemes.root_causes import get_signal_questions, get_category


TASK_META = {
    "task_id": "task2",
    "name": "Old Age / Widow Pension Stopped",
    "description": (
        "An elderly or widow beneficiary's monthly pension stopped without notice. "
        "Causes span pension, banking, KYC, and administrative categories. "
        "Some cases have misleading surface signals requiring 4-7 targeted questions."
    ),
    "difficulty": "medium",
    "scheme": "OAP",
    "max_steps": 15,
    "num_cases": 12,
}

TASK_CASES = ["T2_001", "T2_002", "T2_003", "T2_004", "T2_005",
              "T2_006", "T2_007", "T2_008", "T2_009", "T2_010",
              "T2_011", "T2_012"]


def grade(
    submitted_cause: str,
    submitted_resolution: str,
    questions_asked: list[str],
    steps_used: int,
    true_cause: str,
    true_resolution: str,
) -> float:
    """
    Rule-based grader for Task 2 (medium).

    Scoring identical to Task 1 but with a higher max_steps budget (15),
    so efficiency bonus rewards systematic but not wasteful investigation.
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
