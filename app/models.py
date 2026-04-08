"""
models.py
OpenEnv-compliant Pydantic models for JanSevaEnv.

Action types:
  - ask_question     : agent asks a diagnostic question by its ID (Q01-Q50)
  - submit_diagnosis : agent submits root cause + resolution (ends episode)
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single question asked and the answer received."""
    question_id: str = Field(..., description="Question ID from the question bank (e.g. Q06)")
    question_text: str = Field(..., description="Full text of the question asked")
    answer: str = Field(..., description="Answer returned by the environment for this case")


class Observation(BaseModel):
    """Everything the agent sees at each step."""
    case_id: str = Field(..., description="Unique identifier for the current case")
    grievance_text: str = Field(..., description="Original grievance statement from the beneficiary")
    scheme: str = Field(..., description="Welfare scheme this grievance relates to")
    step_number: int = Field(..., description="Current step within the episode (starts at 0)")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode")
    qa_history: list[QAPair] = Field(default_factory=list, description="All Q&A pairs so far")
    available_questions: dict[str, str] = Field(..., description="Filtered question bank for this scheme")
    available_causes: list[dict] = Field(..., description="All 35 root causes the agent can diagnose")
    available_resolutions: list[dict] = Field(..., description="All 35 resolutions the agent can suggest")
    done: bool = Field(default=False, description="Whether the episode has ended")


class AskQuestionAction(BaseModel):
    """Agent asks one diagnostic question."""
    action_type: Literal["ask_question"] = "ask_question"
    question_id: str = Field(..., description="ID of the question to ask (e.g. Q06)")
    custom_answer: Optional[str] = Field(default=None, description="User-provided answer for custom (non-case) episodes")


class SubmitDiagnosisAction(BaseModel):
    """Agent submits final root cause + resolution. Ends the episode."""
    action_type: Literal["submit_diagnosis"] = "submit_diagnosis"
    cause_id: str = Field(..., description="Root cause ID from available_causes")
    resolution_id: str = Field(..., description="Resolution ID from available_resolutions")
    reasoning: Optional[str] = Field(default=None, description="Optional free-text reasoning (not graded)")


# Union type accepted by step()
Action = AskQuestionAction | SubmitDiagnosisAction


class Reward(BaseModel):
    """Reward signal returned each step."""
    step_reward: float = Field(..., description="Immediate reward for the current action")
    cumulative_reward: float = Field(..., description="Total reward accumulated in this episode")
    episode_score: Optional[float] = Field(default=None, description="Final graded score (0.0-1.0), set when done=True")


class State(BaseModel):
    """Full internal episode state (for state() endpoint)."""
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
    """Full return value from step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict, description="Diagnostic info: cause_correct, resolution_correct, etc.")


class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: Literal["task1", "task2", "task3"] = Field(..., description="Task difficulty level")
    case_id: Optional[str] = Field(default=None, description="Specific case ID (random if omitted)")


class CustomResetRequest(BaseModel):
    """Request body for POST /reset-custom — user writes their own problem."""
    grievance_text: str = Field(..., description="The user's own grievance description")
    scheme: Optional[str] = Field(default=None, description="Welfare scheme (auto-detected from text if omitted)")


class TaskInfo(BaseModel):
    """Metadata for a single task."""
    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    scheme: str
    max_steps: int
    num_cases: int
