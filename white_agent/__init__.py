"""
White Agent - Multi-model vision-language agent for desktop automation.

This package provides a complete agent implementation for OSWorld benchmark tasks,
supporting multiple LLM providers (GPT-4V, Claude, Gemini, Qwen, etc.).

Structure:
- prompt_agent.py: Main PromptAgent class (forked from OSWorld)
- prompts.py: System prompts for different observation/action types
- accessibility_tree_wrap/: Utilities for parsing accessibility trees
- core.py: Action/observation parsing utilities
- a2a/: AgentBeats-compliant A2A protocol server
- rest/: FastAPI REST server for custom orchestrators

Usage:
    # Direct agent usage
    from white_agent.prompt_agent import PromptAgent
    agent = PromptAgent(model="gpt-4o", observation_type="screenshot")
    response, actions = agent.predict(instruction, observation)

    # A2A Server (AgentBeats)
    from white_agent.a2a import start_agent
    start_agent(host="0.0.0.0", port=8001)

    # REST Server (Custom orchestrators)
    from white_agent.rest import app
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

from .prompt_agent import PromptAgent
from .core import parse_actions, parse_observation

__all__ = [
    "PromptAgent",
    "parse_actions",
    "parse_observation",
]
