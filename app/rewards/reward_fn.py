"""
reward_fn.py
Reward shaping for JanSevaEnv.

Design principles:
  - Partial progress signal every step (not just end-of-episode binary).
  - Relevant questions rewarded, irrelevant questions lightly penalised.
  - Final reward is the graded episode score (0.0–1.0) set at diagnosis time.
  - Total episode reward ≤ 1.0.

Step reward (ask_question):
  +0.05  if question is a signal question for the true cause
  +0.02  if question is in diagnostic_questions (but not signal) for the true cause
  -0.01  if question is irrelevant to the true cause
   0.00  if question was already asked (no reward for repetition)

Final reward (submit_diagnosis) — see compute_final_score():
  Correct cause + correct resolution + efficiency + used signal question → up to 1.0
  Correct cause only → 0.50–0.55
  Same category as true cause → 0.20
  Asked signal question but wrong cause → 0.10
  No useful signal, wrong cause → 0.00
"""

from __future__ import annotations
from app.schemes.root_causes import (
    get_signal_questions,
    get_diagnostic_questions,
    get_category,
)


# ---------------------------------------------------------------------------
# Step-level reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    question_id: str,
    questions_asked_so_far: list[str],
    true_cause_id: str,
) -> float:
    """
    Return the immediate reward for asking a question.

    Args:
        question_id: The question ID the agent chose.
        questions_asked_so_far: Questions already asked this episode (before this step).
        true_cause_id: Ground truth cause for the current case.

    Returns:
        Float reward in range [-0.01, +0.05].
    """
    # No reward for repeating a question
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


# ---------------------------------------------------------------------------
# Final episode score (used as episode_score in Reward model)
# ---------------------------------------------------------------------------

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
    Compute the final graded score for the episode (0.0–1.0).

    Scoring breakdown:
      Base score for correct cause + resolution: 0.70
      Efficiency bonus (scales with steps remaining): 0–0.20
      Signal question bonus (agent investigated correctly): 0.10
      Total maximum: 1.00

    Partial credit:
      Correct cause, wrong resolution: 0.50 (+ 0.05 if signal asked)
      Same category, wrong cause: 0.20
      Signal question asked but cause wrong: 0.10
      Completely off: 0.00

    Args:
        submitted_cause: Cause ID the agent diagnosed.
        submitted_resolution: Resolution ID the agent suggested.
        questions_asked: All question IDs asked during episode.
        steps_used: Number of steps consumed.
        max_steps: Maximum steps for the task.
        true_cause: Ground truth cause ID.
        true_resolution: Ground truth resolution ID.

    Returns:
        Float in [0.0, 1.0].
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
        base = 0.70
        # Efficiency bonus: full 0.20 if done in first half of steps, scales down
        efficiency = max(0.0, 1.0 - steps_used / max_steps) * 0.20
        signal_bonus = 0.10 if asked_signal else 0.0
        return round(min(1.0, base + efficiency + signal_bonus), 4)

    if cause_correct:
        # Correct root cause but wrong resolution
        return round(0.50 + (0.05 if asked_signal else 0.0), 4)

    if same_category:
        # Partially on track — right category, wrong specific cause
        return 0.20

    if asked_signal:
        # Agent found the right clue but drew wrong conclusion
        return 0.10

    return 0.0


# ---------------------------------------------------------------------------
# Helper: compute cumulative step rewards for a full history
# ---------------------------------------------------------------------------

def compute_trajectory_reward(
    question_ids_in_order: list[str],
    true_cause_id: str,
) -> float:
    """
    Replay a trajectory of questions and sum up step rewards.
    Useful for debugging or offline evaluation.
    """
    total = 0.0
    asked_so_far: list[str] = []
    for qid in question_ids_in_order:
        total += compute_step_reward(qid, asked_so_far, true_cause_id)
        asked_so_far.append(qid)
    return round(total, 4)
