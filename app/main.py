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

# API routes
app.include_router(router)

# Static frontend
_STATIC_DIR = Path(__file__).parent / "static"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(_STATIC_DIR / "index.html"))

# Health check (IMPORTANT for hackathon)
@app.get("/health")
def health():
    return {"status": "healthy"}

def main():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()