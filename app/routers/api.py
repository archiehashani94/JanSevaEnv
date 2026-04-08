"""
api.py
FastAPI route definitions for JanSevaEnv.
All routes delegate directly to the JanSevaEnv environment instance.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Union  # ✅ FIXED (moved here)

from app.models import (
    ResetRequest,
    CustomResetRequest,
    AskQuestionAction,
    SubmitDiagnosisAction,
    StepResult,
    Observation,
    State,
)
from app.environment import JanSevaEnv
from app.schemes.root_causes import get_all_causes
from app.schemes.resolutions import get_all_resolutions
from app.schemes.policies import get_question_bank, SCHEMES
from app.tasks.task1 import TASK_META as T1_META
from app.tasks.task2 import TASK_META as T2_META
from app.tasks.task3 import TASK_META as T3_META
from app.document_extractor import process_document

router = APIRouter()

# Single shared environment instance
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
    t = text.lower()
    if any(k in t for k in ["mgnrega", "mnrega", "nrega"]):
        return "MGNREGA"
    if any(k in t for k in ["widow", "wap"]):
        return "WAP"
    if any(k in t for k in ["old age", "oap"]):
        return "OAP"
    if any(k in t for k in ["ration", "pds"]):
        return "NFSA-PDS"
    if any(k in t for k in ["disability", "dap"]):
        return "DAP"
    if any(k in t for k in ["pm-kisan", "kisan"]):
        return "PM-KISAN"
    return "PM-KISAN"


@router.post("/reset", response_model=Observation)
def reset_episode(request: Optional[ResetRequest] = None):
    try:
        task_id = (request.task_id if request and request.task_id else None) or "task1"
        case_id = request.case_id if request else None
        return _env.reset(task_id=task_id, case_id=case_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset-custom", response_model=Observation)
def reset_custom(request: CustomResetRequest):
    scheme = request.scheme or _detect_scheme(request.grievance_text)
    if scheme not in SCHEMES:
        scheme = "PM-KISAN"
    try:
        return _env.reset_custom(
            grievance_text=request.grievance_text, scheme=scheme
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/detect-scheme")
def detect_scheme_endpoint(text: str):
    return {"scheme": _detect_scheme(text)}


# ✅ FIXED STEP FUNCTION
@router.post("/step", response_model=StepResult)
def step(action: Union[AskQuestionAction, SubmitDiagnosisAction]):
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/state", response_model=State)
def get_state():
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@router.get("/tasks")
def list_tasks():
    return {"tasks": list(TASK_META_MAP.values())}


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    if task_id not in TASK_META_MAP:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASK_META_MAP[task_id]


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@router.get("/taxonomy/causes")
def list_causes():
    return {"causes": get_all_causes()}


@router.get("/taxonomy/resolutions")
def list_resolutions():
    return {"resolutions": get_all_resolutions()}


@router.get("/taxonomy/questions")
def list_questions():
    return {"questions": get_question_bank()}


@router.get("/taxonomy/schemes")
def list_schemes():
    return {"schemes": SCHEMES}


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

@router.post("/process-document")
async def process_document_endpoint(
    file: UploadFile = File(...),
    doc_type: str = Form("Unknown"),
    scheme: Optional[str] = Form(None),
):
    try:
        file_bytes = await file.read()
        return process_document(
            file_bytes=file_bytes,
            filename=file.filename or "upload",
            doc_type=doc_type,
            scheme=scheme,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}