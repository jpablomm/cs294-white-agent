#!/bin/bash
# White Agent - REST API Runner

export PYTHONPATH="${PYTHONPATH:-/app}"
export PYTHONUNBUFFERED=1

PORT=${PORT:-9002}
HOST=${HOST:-0.0.0.0}

echo "=== White Agent (REST) Starting ===" >&2
echo "HOST: $HOST" >&2
echo "PORT: $PORT" >&2

python3 -m uvicorn white_agent.rest.server:app --host "$HOST" --port "$PORT"
