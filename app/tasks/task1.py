"""
task1.py  —  Easy Task: PM-KISAN Payment Not Received
Scheme: PM-KISAN | Max steps: 10 | Cases: 5
Root causes: aadhaar_not_seeded, bank_account_not_linked, incorrect_bank_details,
             land_records_mismatch, npci_mapping_error
"""

from app.schemes.root_causes import get_signal_questions, get_category


TASK_META = {
    "task_id": "task1",
    "name": "PM-KISAN Payment Not Received",
    "description": (
        "A farmer is not receiving PM-KISAN installments. "
        "Root causes span banking and KYC categories with clear confirmation signals. "
        "1-3 targeted questions are sufficient to reach a correct diagnosis."
    ),
    "difficulty": "easy",
    "scheme": "PM-KISAN",
    "max_steps": 10,
    "num_cases": 12,
    "grader": "rule_based",
}

TASK_CASES = ["T1_001", "T1_002", "T1_003", "T1_004", "T1_005",
              "T1_006", "T1_007", "T1_008", "T1_009", "T1_010",
              "T1_011", "T1_012"]


def grade(
    submitted_cause: str,
    submitted_resolution: str,
    questions_asked: list[str],
    steps_used: int,
    true_cause: str,
    true_resolution: str,
) -> float:
    """
    Rule-based grader for Task 1 (easy).

    Scoring:
      Correct cause + resolution + efficiency + signal question: up to 1.0
      Correct cause + resolution (base):                         0.70
      Correct cause only + signal asked:                         0.55
      Correct cause only:                                        0.50
      Same category, wrong cause:                                0.20
      Signal question asked, wrong cause:                        0.10
      Completely wrong:                                          0.00
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
