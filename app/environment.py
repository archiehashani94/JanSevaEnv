"""
environment.py
Core JanSevaEnv RL environment — implements the OpenEnv interface.

  reset(task_id, case_id=None) -> Observation
  step(action)                 -> StepResult
  state()                      -> State
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

from app.models import (
    Action, AskQuestionAction, Observation, QAPair,
    Reward, State, StepResult, SubmitDiagnosisAction,
)
from app.schemes.policies import get_scheme_questions
from app.schemes.root_causes import get_all_causes
from app.schemes.resolutions import get_all_resolutions
from app.rewards.reward_fn import compute_step_reward, compute_final_score

_CASES_PATH = Path(__file__).parent / "data" / "cases.json"

TASK_MAX_STEPS: dict[str, int] = {
    "task1": 10,
    "task2": 15,
    "task3": 20,
}

_DEFAULT_ANSWER = "That information is not available in the current case records."


class JanSevaEnv:
    """
    JanSevaEnv: Indian welfare & pension grievance resolution environment.

    Episode flow:
      1. reset(task_id)  -> loads a case, returns initial Observation
      2. step(AskQuestionAction) -> returns answer + partial reward, repeat as needed
      3. step(SubmitDiagnosisAction) -> graded, episode ends, done=True
    """

    def __init__(self) -> None:
        self._cases: dict = self._load_cases()
        self._state: Optional[_EpisodeState] = None

    # ------------------------------------------------------------------
    # OpenEnv public interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str, case_id: Optional[str] = None) -> Observation:
        """Start a new episode. Returns the initial observation."""
        if task_id not in self._cases:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be task1, task2, or task3.")

        task_cases = self._cases[task_id]
        if case_id is not None:
            matched = [c for c in task_cases if c["case_id"] == case_id]
            if not matched:
                raise ValueError(f"Case '{case_id}' not found in {task_id}.")
            case = matched[0]
        else:
            case = random.choice(task_cases)

        max_steps = TASK_MAX_STEPS[task_id]
        self._state = _EpisodeState(task_id=task_id, case=case, max_steps=max_steps, is_custom=False)
        return self._build_observation()

    def reset_custom(self, grievance_text: str, scheme: str) -> Observation:
        """Start a custom episode where the user provides their own grievance text."""
        case = {
            "case_id": "CUSTOM",
            "grievance_text": grievance_text,
            "scheme": scheme,
            "question_answers": {},  # empty — user provides answers in real-time
            "ground_truth": None,    # no ground truth; no scoring
            "difficulty": "custom",
        }
        self._state = _EpisodeState(task_id="custom", case=case, max_steps=20, is_custom=True)
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Process one agent action. Returns StepResult."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        if isinstance(action, AskQuestionAction):
            return self._handle_ask_question(action)
        elif isinstance(action, SubmitDiagnosisAction):
            return self._handle_submit_diagnosis(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def state(self) -> State:
        """Return the current internal episode state."""
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        s = self._state
        return State(
            case_id=s.case["case_id"],
            task_id=s.task_id,
            scheme=s.case["scheme"],
            step_number=s.step_number,
            max_steps=s.max_steps,
            qa_history=list(s.qa_history),
            questions_asked=list(s.questions_asked),
            done=s.done,
            submitted_cause=s.submitted_cause,
            submitted_resolution=s.submitted_resolution,
            cumulative_reward=round(s.cumulative_reward, 4),
        )

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_ask_question(self, action: AskQuestionAction) -> StepResult:
        s = self._state
        available_qs = get_scheme_questions(s.case["scheme"])

        if action.question_id not in available_qs:
            reward_val = -0.02
            s.cumulative_reward += reward_val
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(step_reward=reward_val, cumulative_reward=round(s.cumulative_reward, 4)),
                done=False,
                info={"error": f"Question '{action.question_id}' not available for scheme '{s.case['scheme']}'."},
            )

        question_text = available_qs[action.question_id]

        if s.is_custom:
            # Custom mode: answer comes from the user via action.custom_answer
            answer = action.custom_answer or ""
            step_reward = 0.01  # small fixed reward for engaging with a question
        else:
            qa_map: dict = s.case.get("question_answers", {})
            answer = qa_map.get(action.question_id, _DEFAULT_ANSWER)
            step_reward = compute_step_reward(
                question_id=action.question_id,
                questions_asked_so_far=list(s.questions_asked),
                true_cause_id=s.case["ground_truth"]["cause"],
            )

        s.qa_history.append(QAPair(question_id=action.question_id, question_text=question_text, answer=answer))
        s.questions_asked.append(action.question_id)
        s.step_number += 1
        s.cumulative_reward += step_reward

        if s.step_number >= s.max_steps:
            s.done = True

        return StepResult(
            observation=self._build_observation(),
            reward=Reward(step_reward=round(step_reward, 4), cumulative_reward=round(s.cumulative_reward, 4)),
            done=s.done,
            info={"question_id": action.question_id, "answer": answer, "steps_remaining": max(0, s.max_steps - s.step_number)},
        )

    def _handle_submit_diagnosis(self, action: SubmitDiagnosisAction) -> StepResult:
        s = self._state

        if s.is_custom:
            # Custom mode: no ground truth — just record what was submitted
            s.submitted_cause = action.cause_id
            s.submitted_resolution = action.resolution_id
            s.step_number += 1
            s.done = True
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(
                    step_reward=0.0,
                    cumulative_reward=round(s.cumulative_reward, 4),
                    episode_score=None,
                ),
                done=True,
                info={
                    "submitted_cause": action.cause_id,
                    "submitted_resolution": action.resolution_id,
                    "true_cause": None,
                    "true_resolution": None,
                    "cause_correct": None,
                    "resolution_correct": None,
                    "episode_score": None,
                    "steps_used": s.step_number,
                    "is_custom": True,
                },
            )

        true_cause = s.case["ground_truth"]["cause"]
        true_resolution = s.case["ground_truth"]["resolution"]

        episode_score = compute_final_score(
            submitted_cause=action.cause_id,
            submitted_resolution=action.resolution_id,
            questions_asked=list(s.questions_asked),
            steps_used=s.step_number,
            max_steps=s.max_steps,
            true_cause=true_cause,
            true_resolution=true_resolution,
        )

        s.submitted_cause = action.cause_id
        s.submitted_resolution = action.resolution_id
        s.step_number += 1
        s.cumulative_reward += episode_score
        s.done = True

        return StepResult(
            observation=self._build_observation(),
            reward=Reward(
                step_reward=round(episode_score, 4),
                cumulative_reward=round(s.cumulative_reward, 4),
                episode_score=episode_score,
            ),
            done=True,
            info={
                "submitted_cause": action.cause_id,
                "submitted_resolution": action.resolution_id,
                "true_cause": true_cause,
                "true_resolution": true_resolution,
                "cause_correct": action.cause_id == true_cause,
                "resolution_correct": action.resolution_id == true_resolution,
                "episode_score": episode_score,
                "steps_used": s.step_number,
            },
        )

    def _build_observation(self) -> Observation:
        s = self._state
        scheme_qs = get_scheme_questions(s.case["scheme"])

        if s.is_custom:
            # Custom mode: show ALL scheme questions — user answers them in real-time
            available_qs = {
                qid: text
                for qid, text in scheme_qs.items()
                if qid not in s.questions_asked
            }
        else:
            # Preset case: filter to only questions this case actually answers.
            # Prevents "information not available" responses.
            case_answered = set(s.case.get("question_answers", {}).keys())
            already_asked = set(s.questions_asked)
            available_qs = {
                qid: text
                for qid, text in scheme_qs.items()
                if qid in case_answered or qid in already_asked
            }

        return Observation(
            case_id=s.case["case_id"],
            grievance_text=s.case["grievance_text"],
            scheme=s.case["scheme"],
            step_number=s.step_number,
            max_steps=s.max_steps,
            qa_history=list(s.qa_history),
            available_questions=available_qs,
            available_causes=[{"id": c["id"], "label": c["label"], "resolution_id": c["resolution_id"]} for c in get_all_causes()],
            available_resolutions=[{"id": r["id"], "label": r["label"]} for r in get_all_resolutions()],
            done=s.done,
        )

    @staticmethod
    def _load_cases() -> dict:
        with open(_CASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)


class _EpisodeState:
    """Mutable internal state for one running episode."""

    def __init__(self, task_id: str, case: dict, max_steps: int, is_custom: bool = False) -> None:
        self.task_id = task_id
        self.case = case
        self.max_steps = max_steps
        self.is_custom = is_custom
        self.step_number: int = 0
        self.qa_history: list[QAPair] = []
        self.questions_asked: list[str] = []
        self.done: bool = False
        self.submitted_cause: Optional[str] = None
        self.submitted_resolution: Optional[str] = None
        self.cumulative_reward: float = 0.0
