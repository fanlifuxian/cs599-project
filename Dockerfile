# 个性化健康规划多智能体平台
# Multi-agent health planning platform Docker image

FROM python:3.11-slim

LABEL project="cs599-health-multi-agent"
LABEL description="Personalized Health Planning Multi-Agent Platform"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY .env.example ./

# Data directory for memory persistence
RUN mkdir -p /app/data

# Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run Streamlit UI
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
