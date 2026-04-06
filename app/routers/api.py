"""
api.py
FastAPI route definitions for JanSevaEnv.

Endpoints:
  GET  /health              → liveness check
  GET  /tasks               → list all task metadata
  GET  /tasks/{task_id}     → single task metadata
  POST /reset               → start new episode
  POST /step                → take one action
  GET  /state               → current episode state
  GET  /taxonomy/causes     → all root causes
  GET  /taxonomy/resolutions→ all resolutions
  GET  /taxonomy/questions  → full question bank
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.environment import JanSevaEnv
from app.models import (
    Action,
    AskQuestionAction,
    ResetRequest,
    StepResult,
    SubmitDiagnosisAction,
    TaskInfo,
)
from app.schemes.policies import get_question_bank, get_all_scheme_codes, SCHEMES
from app.schemes.root_causes import get_all_causes
from app.schemes.resolutions import get_all_resolutions
from app.tasks import task1, task2, task3

router = APIRouter()

# Single shared environment instance (stateful, one episode at a time)
_env = JanSevaEnv()

_TASK_MODULES = {
    "task1": task1,
    "task2": task2,
    "task3": task3,
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "environment": "JanSevaEnv", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@router.get("/tasks", response_model=list[dict], tags=["tasks"])
def list_tasks():
    """Return metadata for all three tasks."""
    return [mod.get_task_info() for mod in _TASK_MODULES.values()]


@router.get("/tasks/{task_id}", response_model=dict, tags=["tasks"])
def get_task(task_id: str):
    if task_id not in _TASK_MODULES:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return _TASK_MODULES[task_id].get_task_info()


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@router.post("/reset", tags=["environment"])
def reset(request: ResetRequest):
    """
    Start a new episode.

    Body:
      task_id: "task1" | "task2" | "task3"
      case_id: optional specific case ID (e.g. "T1_001")

    Returns initial Observation.
    """
    try:
        observation = _env.reset(task_id=request.task_id, case_id=request.case_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return observation.model_dump()


@router.post("/step", response_model=dict, tags=["environment"])
def step(action_data: dict):
    """
    Take one action in the current episode.

    Body (ask_question):
      {"action_type": "ask_question", "question_id": "Q06"}

    Body (submit_diagnosis):
      {"action_type": "submit_diagnosis", "cause_id": "aadhaar_not_seeded",
       "resolution_id": "seed_aadhaar", "reasoning": "optional text"}

    Returns StepResult (observation, reward, done, info).
    """
    action_type = action_data.get("action_type")
    try:
        if action_type == "ask_question":
            action: Action = AskQuestionAction(**action_data)
        elif action_type == "submit_diagnosis":
            action = SubmitDiagnosisAction(**action_data)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action_type '{action_type}'. Must be 'ask_question' or 'submit_diagnosis'."
            )
    except (ValidationError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result: StepResult = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result.model_dump()


@router.get("/state", tags=["environment"])
def get_state():
    """Return the current internal episode state."""
    try:
        return _env.state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Taxonomy / reference endpoints
# ---------------------------------------------------------------------------

@router.get("/taxonomy/causes", tags=["taxonomy"])
def list_causes():
    """Return all 35 root causes from taxonomy.json."""
    return get_all_causes()


@router.get("/taxonomy/resolutions", tags=["taxonomy"])
def list_resolutions():
    """Return all 35 resolutions from taxonomy.json."""
    return get_all_resolutions()


@router.get("/taxonomy/questions", tags=["taxonomy"])
def list_questions():
    """Return the full diagnostic question bank (Q01–Q50)."""
    return get_question_bank()


@router.get("/taxonomy/schemes", tags=["taxonomy"])
def list_schemes():
    """Return all supported welfare scheme definitions."""
    return SCHEMES
