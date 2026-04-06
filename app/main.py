"""
main.py
FastAPI application entry point for JanSevaEnv.

Run locally:
  uvicorn app.main:app --reload --port 7860

Docker:
  docker build -t janseva-env .
  docker run -p 7860:7860 janseva-env
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.api import router

app = FastAPI(
    title="JanSevaEnv",
    description=(
        "An OpenEnv-compliant reinforcement learning environment that simulates "
        "Indian welfare and pension grievance resolution. An AI agent learns to "
        "identify the correct root cause from ~35 predefined issues by asking "
        "step-by-step diagnostic questions, then suggests the correct resolution. "
        "Evaluation is fully rule-based — no LLM judges."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root():
    return {
        "name": "JanSevaEnv",
        "version": "1.0.0",
        "description": "Indian welfare & pension grievance resolution RL environment",
        "docs": "/docs",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "health": "GET /health",
            "taxonomy": {
                "causes": "GET /taxonomy/causes",
                "resolutions": "GET /taxonomy/resolutions",
                "questions": "GET /taxonomy/questions",
                "schemes": "GET /taxonomy/schemes",
            },
        },
    }
