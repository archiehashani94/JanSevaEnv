from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from app.routers.api import router
from app.models import Action, Observation, State

app = FastAPI(
    title="JanSevaEnv",
    description="Indian welfare & pension grievance resolution RL environment.",
    version="1.0.0",
)

# API routes
app.include_router(router)

# Static frontend
_STATIC_DIR = Path(__file__).parent / "static"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(_STATIC_DIR / "index.html"))

# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}


# OpenEnv runtime API: metadata
@app.get("/metadata")
def metadata():
    return {
        "name": "JanSevaEnv",
        "description": (
            "Indian welfare and pension grievance resolution RL environment. "
            "The agent investigates beneficiary grievances across PM-KISAN, OAP/WAP, "
            "MGNREGA, NFSA-PDS schemes by asking diagnostic questions and submitting a root-cause diagnosis."
        ),
        "version": "1.0.0",
        "tasks": ["task1", "task2", "task3"],
    }


# OpenEnv runtime API: schema
@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


# OpenEnv runtime API: MCP (JSON-RPC stub)
@app.post("/mcp")
async def mcp(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {"name": "JanSevaEnv", "tools": []},
    })


def main():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()