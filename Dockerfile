# ═══════════════════════════════════════════════════════════════════════════════
# 个性化健康规划多智能体平台 v2.0 — Enterprise Multi-Stage Docker Build
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a virtual env
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL project="cs599-health-multi-agent"
LABEL version="2.0.0"
LABEL description="Personalized Health Planning Multi-Agent Platform — Enterprise Edition"
LABEL course="CS599 Enterprise Application Software Design & Development"
LABEL org.opencontainers.image.source="https://github.com/user/cs599-project"

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/

# Copy configuration templates
COPY .env.example ./

# Create data directories for persistence
RUN mkdir -p /app/data /app/data/chroma /app/logs \
    && chmod -R 755 /app/data

# ── Ports ────────────────────────────────────────────────────────────────────
# 8501: Streamlit UI
# 8000: FastAPI Server
EXPOSE 8501 8000

# ── Health Check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || \
        curl -f http://localhost:8501/_stcore/health || exit 1

# ── Volumes ──────────────────────────────────────────────────────────────────
VOLUME ["/app/data", "/app/logs"]

# ── Entry Points ─────────────────────────────────────────────────────────────

# Default: Streamlit UI (for demos)
CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# To run the FastAPI server instead, override CMD:
# docker run -p 8000:8000 --env-file .env health-agents \
#   uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4

# To run CLI:
# docker run -it --env-file .env health-agents python src/main.py
