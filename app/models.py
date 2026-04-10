"""
models.py
OpenEnv-compliant Pydantic models for JanSevaEnv.
"""

from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


class QAPair(BaseModel):
    question_id: str = Field(..., description="Question ID from the question bank (e.g. Q06)")
    question_text: str = Field(..., description="Full text of the question asked")
    answer: str = Field(..., description="Answer returned by the environment for this case")


class Observation(BaseModel):
    case_id: str
    grievance_text: str
    scheme: str
    step_number: int
    max_steps: int
    qa_history: list[QAPair] = Field(default_factory=list)
    available_questions: dict[str, str]
    available_causes: list[dict]
    available_resolutions: list[dict]
    done: bool = False


class AskQuestionAction(BaseModel):
    action_type: Literal["ask_question"] = "ask_question"
    question_id: str
    custom_answer: Optional[str] = None


class SubmitDiagnosisAction(BaseModel):
    action_type: Literal["submit_diagnosis"] = "submit_diagnosis"
    cause_id: str
    resolution_id: str
    reasoning: Optional[str] = None


# ✅ FIXED HERE
Action = Union[AskQuestionAction, SubmitDiagnosisAction]


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    episode_score: Optional[float] = None


class State(BaseModel):
    case_id: str
    task_id: str
    scheme: str
    step_number: int
    max_steps: int
    qa_history: list[QAPair]
    questions_asked: list[str]
    done: bool
    submitted_cause: Optional[str] = None
    submitted_resolution: Optional[str] = None
    cumulative_reward: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[Literal["task1", "task2", "task3"]] = "task1"
    case_id: Optional[str] = None


class CustomResetRequest(BaseModel):
    grievance_text: str
    scheme: Optional[str] = None


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    scheme: str
    max_steps: int
    num_cases: int