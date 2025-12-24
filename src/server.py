#!/usr/bin/env python3
"""
White Agent - A2A Protocol Server for OSWorld Desktop Automation

Uses PromptAgent for multi-model support (GPT-4V, Claude, Gemini, Qwen).
"""

import argparse
import os
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from white_agent.a2a.server import PromptAgentExecutor

# Configuration from environment
MODEL = os.environ.get("MODEL", "gpt-4o")


def main():
    parser = argparse.ArgumentParser(description="Run the White Agent A2A server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skills = [
        AgentSkill(
            id="desktop-automation",
            name="Desktop Automation",
            description=f"Execute desktop automation actions (click, type, hotkey, scroll) using {MODEL}",
            tags=["automation", "desktop", "gui", "osworld"],
            examples=[],
        ),
        AgentSkill(
            id="vision-reasoning",
            name="Vision-Language Reasoning",
            description=f"Analyze screenshots and determine appropriate actions using {MODEL}",
            tags=["vision", "reasoning", "screenshot"],
            examples=[],
        ),
    ]

    agent_card = AgentCard(
        name="osworld-white-agent",
        description=f"White agent for executing desktop automation tasks using {MODEL}. "
                    "Receives observations (screenshots, instructions) and returns actions.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['application/json'],
        default_output_modes=['application/json'],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PromptAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
