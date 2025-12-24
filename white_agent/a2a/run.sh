#!/bin/bash
# White Agent - A2A Protocol Runner
# For use with AgentBeats earthshaker controller

export PYTHONPATH="${PYTHONPATH:-/app}"
export PYTHONUNBUFFERED=1

echo "=== White Agent (A2A) Starting ===" >&2
echo "HOST: ${HOST:-0.0.0.0}" >&2
echo "AGENT_PORT: ${AGENT_PORT:-8001}" >&2
echo "AGENT_URL: ${AGENT_URL:-not set}" >&2

python3 -m white_agent.a2a.server
