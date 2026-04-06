# JanSevaEnv — Dockerfile
# Hugging Face Spaces compatible (port 7860, non-root user)
# Build: docker build -t janseva-env .
# Run:   docker run -p 7860:7860 janseva-env

FROM python:3.11-slim

# HF Spaces expects a non-root user named 'user'
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY --chown=user:user requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi==0.135.3 \
        uvicorn==0.42.0 \
        pydantic==2.12.5 \
        openai==2.30.0 \
        python-dotenv==1.2.2 \
        pyyaml==6.0.3

# Copy application code
COPY --chown=user:user . .

# Switch to non-root user
USER user

# HF Spaces uses port 7860
EXPOSE 7860

# Health check — ensures the container is ready before scoring
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
