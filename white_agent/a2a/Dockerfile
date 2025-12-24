# White Agent - A2A Protocol (AgentBeats-compliant)
# Uses earthshaker controller with a2a SDK

FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-white-agent-agentbeats.txt .

# Install Python dependencies (includes earthshaker and a2a SDK)
RUN pip install --no-cache-dir -r requirements-white-agent-agentbeats.txt

# Copy application code
COPY white_agent/ white_agent/

# Copy run script and Procfile for earthshaker
COPY white_agent/a2a/run.sh run.sh
COPY Procfile.white-agent Procfile
RUN chmod +x run.sh

# Environment
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/status || exit 1

# Use AgentBeats controller
CMD ["sh", "-c", "agentbeats run_ctrl"]
