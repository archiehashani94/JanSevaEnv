"""
models.py
OpenEnv-compliant Pydantic models for JanSevaEnv.

Action types:
  - ask_question  : agent asks a diagnostic question by its ID (Q01–Q50)
  - submit_diagnosis : agent submits root cause + resolution (ends episode)
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Question / Answer pair (used in observation history)
# ---------------------------------------------------------------------------

class QAPair(BaseModel):
    question_id: str = Field(..., description="Question ID from the question bank (e.g. Q06)")
    question_text: str = Field(..., description="Full text of the question asked")
    answer: str = Field(..., description="Answer returned by the environment for this case")


# ---------------------------------------------------------------------------
# Observation — what the agent sees each step
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    case_id: str = Field(..., description="Unique identifier for the current case")
    grievance_text: str = Field(..., description="Original grievance statement from the beneficiary")
    scheme: str = Field(..., description="Welfare scheme this grievance relates to")
    step_number: int = Field(..., description="Current step within the episode (starts at 0)")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode")
    qa_history: list[QAPair] = Field(
        default_factory=list,
        description="All questions asked so far and the answers received"
    )
    available_questions: dict[str, str] = Field(
        ...,
        description="Filtered question bank for this scheme {question_id: question_text}"
    )
    available_causes: list[dict] = Field(
        ...,
        description="List of possible root causes {id, label} the agent can diagnose"
    )
    available_resolutions: list[dict] = Field(
        ...,
        description="List of possible resolutions {id, label} the agent can suggest"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")


# ---------------------------------------------------------------------------
# Action — what the agent sends each step
# ---------------------------------------------------------------------------

class AskQuestionAction(BaseModel):
    action_type: Literal["ask_question"] = "ask_question"
    question_id: str = Field(
        ...,
        description="ID of the question to ask from the available_questions bank"
    )


class SubmitDiagnosisAction(BaseModel):
    action_type: Literal["submit_diagnosis"] = "submit_diagnosis"
    cause_id: str = Field(
        ...,
        description="Root cause ID from available_causes"
    )
    resolution_id: str = Field(
        ...,
        description="Resolution ID from available_resolutions"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional free-text reasoning for the diagnosis (not graded)"
    )


# Union type for step() input
Action = AskQuestionAction | SubmitDiagnosisAction


# ---------------------------------------------------------------------------
# Reward — step-level and episode-level signal
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    step_reward: float = Field(
        ...,
        description="Immediate reward for the current action (+0.05 relevant, -0.01 irrelevant)"
    )
    cumulative_reward: float = Field(
        ...,
        description="Total reward accumulated so far in the episode"
    )
    episode_score: Optional[float] = Field(
        default=None,
        description="Final graded score (0.0–1.0), only set when done=True"
    )


# ---------------------------------------------------------------------------
# State — full internal state (for state() endpoint)
# ---------------------------------------------------------------------------

class State(BaseModel):
    case_id: str
    task_id: str
    scheme: str
    step_number: int
    max_steps: int
    qa_history: list[QAPair]
    questions_asked: list[str] = Field(
        description="List of question IDs asked so far"
    )
    done: bool
    submitted_cause: Optional[str] = None
    submitted_resolution: Optional[str] = None
    cumulative_reward: float = 0.0


# ---------------------------------------------------------------------------
# StepResult — full return from step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(
        default_factory=dict,
        description="Auxiliary diagnostic info (cause_correct, resolution_correct, etc.)"
    )


# ---------------------------------------------------------------------------
# Reset request / response
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Literal["task1", "task2", "task3"] = Field(
        ...,
        description="Task difficulty level to run"
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Specific case ID to load. If None, a random case is selected."
    )


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    scheme: str
    max_steps: int
    num_cases: int
