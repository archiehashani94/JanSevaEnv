"""
main.py
FastAPI application entry point for JanSevaEnv.
Mounts the API router and serves the static frontend from app/static/.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.routers.api import router

app = FastAPI(
    title="JanSevaEnv",
    description="Indian welfare & pension grievance resolution RL environment.",
    version="1.0.0",
)

# Mount API routes
app.include_router(router)

# Serve the frontend from app/static/
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "service": "JanSevaEnv"}
