"""
reward_fn.py
Reward shaping for JanSevaEnv.

Step reward (ask_question):
  +0.05  signal question for the true cause
  +0.02  diagnostic question (relevant but not definitive)
  -0.01  irrelevant question
   0.00  repeated question

Episode score (submit_diagnosis) — up to 1.0:
  Base 0.70 for correct cause + resolution
  + efficiency bonus up to 0.20 (scales with steps remaining)
  + 0.10 signal bonus if agent asked at least one signal question
  Partial credit for correct cause only, same category, or signal asked
"""

from __future__ import annotations
from app.schemes.root_causes import get_signal_questions, get_diagnostic_questions, get_category


def compute_step_reward(
    question_id: str,
    questions_asked_so_far: list[str],
    true_cause_id: str,
) -> float:
    """Immediate reward for asking a question."""
    if question_id in questions_asked_so_far:
        return 0.0

    signal_qs = set(get_signal_questions(true_cause_id))
    diagnostic_qs = set(get_diagnostic_questions(true_cause_id))

    if question_id in signal_qs:
        return 0.05
    elif question_id in diagnostic_qs:
        return 0.02
    else:
        return -0.01


def compute_final_score(
    submitted_cause: str,
    submitted_resolution: str,
    questions_asked: list[str],
    steps_used: int,
    max_steps: int,
    true_cause: str,
    true_resolution: str,
) -> float:
    """
    Final graded episode score (0.0-1.0).

    Breakdown:
      Correct cause + resolution: base 0.70
      Efficiency bonus: 0-0.20 (linear with steps remaining)
      Signal question bonus: 0.10
    Partial credit:
      Correct cause only: 0.50 (+ 0.05 if signal asked)
      Same category, wrong cause: 0.20
      Asked signal but wrong cause: 0.10
      Completely wrong: 0.00
    """
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
        efficiency = max(0.0, 1.0 - steps_used / max_steps) * 0.20
        signal_bonus = 0.10 if asked_signal else 0.0
        return round(min(1.0, 0.70 + efficiency + signal_bonus), 4)

    if cause_correct:
        return round(0.50 + (0.05 if asked_signal else 0.0), 4)

    if same_category:
        return 0.20

    if asked_signal:
        return 0.10

    return 0.0


def compute_trajectory_reward(question_ids_in_order: list[str], true_cause_id: str) -> float:
    """Sum step rewards across a full question trajectory. Useful for offline evaluation."""
    total = 0.0
    asked_so_far: list[str] = []
    for qid in question_ids_in_order:
        total += compute_step_reward(qid, asked_so_far, true_cause_id)
        asked_so_far.append(qid)
    return round(total, 4)
