from typing import Union
from fastapi import APIRouter
from typing import Union
from app.models import AskQuestionAction, SubmitDiagnosisAction
"""
api.py
FastAPI route definitions for JanSevaEnv.
All routes delegate directly to the JanSevaEnv environment instance.
"""

from fastapi import APIRouter, HTTPException
from app.models import ResetRequest, CustomResetRequest, AskQuestionAction, SubmitDiagnosisAction, StepResult, Observation, State
from app.environment import JanSevaEnv
from app.schemes.root_causes import get_all_causes
from app.schemes.resolutions import get_all_resolutions
from app.schemes.policies import get_question_bank, SCHEMES
from app.tasks.task1 import TASK_META as T1_META
from app.tasks.task2 import TASK_META as T2_META
from app.tasks.task3 import TASK_META as T3_META

router = APIRouter()

# Single shared environment instance (session-based; one episode at a time)
_env = JanSevaEnv()

TASK_META_MAP = {
    "task1": T1_META,
    "task2": T2_META,
    "task3": T3_META,
}


# ---------------------------------------------------------------------------
# Episode control
# ---------------------------------------------------------------------------

def _detect_scheme(text: str) -> str:
    """Keyword-based scheme detection from free-form grievance text."""
    t = text.lower()
    if any(k in t for k in ["mgnrega", "mnrega", "nrega", "job card", "muster roll", "fto", "labourer", "wages unpaid", "wage payment"]):
        return "MGNREGA"
    if any(k in t for k in ["widow pension", "widow assistance", "wap", "ignwps"]):
        return "WAP"
    if any(k in t for k in ["old age pension", "old age", "oap", "jeevan pramaan", "life certificate", "annual certificate", "pensioner"]):
        return "OAP"
    if any(k in t for k in ["ration", "pds", "fair price shop", "fps", "food grain", "nfsa", "ration card", "foodgrain", "wheat", "rice quota"]):
        return "NFSA-PDS"
    if any(k in t for k in ["disability pension", "disabled", "dap", "handicap", "divyang"]):
        return "DAP"
    if any(k in t for k in ["pm-kisan", "pmkisan", "kisan samman", "farmer installment", "kisan nidhi"]):
        return "PM-KISAN"
    if any(k in t for k in ["farmer", "kisan", "agriculture", "installment", "pm kisan", "farm"]):
        return "PM-KISAN"
    if any(k in t for k in ["pension", "pensioner", "monthly pension"]):
        return "OAP"
    return "PM-KISAN"  # safe default


@router.post("/reset", response_model=Observation, summary="Start a new episode")
def reset_episode(request: ResetRequest):
    """
    Start a new episode for the given task.
    Optionally specify a case_id to load a specific case; otherwise a random one is selected.
    """
    try:
        obs = _env.reset(task_id=request.task_id, case_id=request.case_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset-custom", response_model=Observation, summary="Start a custom episode with user-written problem")
def reset_custom(request: CustomResetRequest):
    """
    Start a custom episode. The user provides their own grievance text.
    Scheme is auto-detected from the text if not supplied.
    Questions are drawn from the full scheme question bank — the user answers each one.
    No scoring at the end (no ground truth).
    """
    scheme = request.scheme or _detect_scheme(request.grievance_text)
    from app.schemes.policies import SCHEMES
    if scheme not in SCHEMES:
        scheme = "PM-KISAN"
    try:
        obs = _env.reset_custom(grievance_text=request.grievance_text, scheme=scheme)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/detect-scheme", summary="Auto-detect scheme from grievance text")
def detect_scheme_endpoint(text: str):
    """Return the auto-detected scheme for a grievance description."""
    return {"scheme": _detect_scheme(text), "text_preview": text[:120]}


@router.post("/step", response_model=StepResult, summary="Take one action")
from typing import Union

def step(action: Union[AskQuestionAction, SubmitDiagnosisAction]):
    """
    Take one action in the current episode.
    - ask_question: provide question_id
    - submit_diagnosis: provide cause_id + resolution_id (ends episode)
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/state", response_model=State, summary="Get current episode state")
def get_state():
    """Return the full internal state of the current episode."""
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

@router.get("/tasks", summary="List all tasks")
def list_tasks():
    """Return metadata for all three tasks."""
    return {"tasks": list(TASK_META_MAP.values())}


@router.get("/tasks/{task_id}", summary="Get a single task's metadata")
def get_task(task_id: str):
    """Return metadata for a specific task (task1, task2, or task3)."""
    if task_id not in TASK_META_MAP:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return TASK_META_MAP[task_id]


# ---------------------------------------------------------------------------
# Taxonomy endpoints
# ---------------------------------------------------------------------------

@router.get("/taxonomy/causes", summary="List all 35 root causes")
def list_causes():
    """Return all root cause definitions from taxonomy.json."""
    return {"causes": get_all_causes()}


@router.get("/taxonomy/resolutions", summary="List all 35 resolutions")
def list_resolutions():
    """Return all resolution definitions from taxonomy.json."""
    return {"resolutions": get_all_resolutions()}


@router.get("/taxonomy/questions", summary="List the full question bank")
def list_questions():
    """Return the full diagnostic question bank (Q01-Q50)."""
    return {"questions": get_question_bank()}


@router.get("/taxonomy/schemes", summary="List all scheme definitions")
def list_schemes():
    """Return welfare scheme metadata."""
    return {"schemes": SCHEMES}
