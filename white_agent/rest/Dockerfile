# White Agent - REST API
# FastAPI-based server for custom orchestrator integration

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-white-agent.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-white-agent.txt

# Copy application code
COPY white_agent/ white_agent/

# Environment
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run FastAPI server directly
CMD ["sh", "-c", "uvicorn white_agent.rest.server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8080}"]
