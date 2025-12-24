#!/usr/bin/env python3
"""
White Agent - A2A Protocol Server

Uses the a2a SDK (A2AStarletteApplication) for AgentBeats compliance.
Directly uses PromptAgent for multi-model support (GPT-4V, Claude, Gemini, Qwen, etc.)

Based on: https://github.com/agentbeats/agentify-example-tau-bench
"""

import base64
import json
import logging
import os
from typing import Dict, Any, List, Optional

import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

from white_agent.prompt_agent import PromptAgent
from white_agent.core import parse_observation, parse_actions

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL = os.environ.get("MODEL", os.environ.get("GPT5_MODEL", "gpt-5.1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", os.environ.get("GPT4V_TEMPERATURE", "1.0")))
# For Qwen models, set OSWORLD_OBS_TYPE=screenshot (pure visual grounding)
OBSERVATION_TYPE = os.environ.get("OSWORLD_OBS_TYPE", "screenshot_a11y_tree")
ACTION_SPACE = os.environ.get("ACTION_SPACE", "pyautogui")
MAX_TRAJECTORY_LENGTH = int(os.environ.get("MAX_TRAJECTORY_LENGTH", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))


def prepare_agent_card(url: str, model: str) -> AgentCard:
    """Prepare A2A-compliant agent card."""
    skills = [
        AgentSkill(
            id="desktop-automation",
            name="Desktop Automation",
            description=f"Execute desktop automation actions (click, type, hotkey, scroll) using {model}",
            tags=["automation", "desktop", "gui", "osworld"],
            examples=[],
        ),
        AgentSkill(
            id="vision-reasoning",
            name="Vision-Language Reasoning",
            description=f"Analyze screenshots and determine appropriate actions using {model}",
            tags=["vision", "reasoning", "screenshot"],
            examples=[],
        ),
    ]

    return AgentCard(
        name=f"osworld_agent_{model.replace('-', '_')}",
        description=f"White agent for executing desktop automation tasks using {model}. "
                    "Receives observations (screenshots, instructions) and returns actions for OSWorld workflows.",
        url=url,
        version="2.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(),
        skills=skills,
    )


class PromptAgentExecutor(AgentExecutor):
    """
    A2A Agent executor using PromptAgent directly.

    Supports all models that PromptAgent supports (GPT-4V, Claude, Gemini, Qwen, etc.)
    """

    def __init__(self):
        self.agent: PromptAgent | None = None
        self.ctx_id_to_history: Dict[str, list] = {}
        logger.info(
            f"PromptAgentExecutor initialized "
            f"(model={MODEL}, obs_type={OBSERVATION_TYPE}, agent will be created on first use)"
        )

    def _get_agent(self) -> PromptAgent:
        """Get or create the PromptAgent instance."""
        if self.agent is None:
            logger.info(f"Creating PromptAgent: model={MODEL}, obs_type={OBSERVATION_TYPE}")
            self.agent = PromptAgent(
                model=MODEL,
                temperature=TEMPERATURE,
                observation_type=OBSERVATION_TYPE,
                action_space=ACTION_SPACE,
                max_trajectory_length=MAX_TRAJECTORY_LENGTH,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
            )
            logger.info("PromptAgent created successfully")
        return self.agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a task using PromptAgent."""
        try:
            agent = self._get_agent()

            # Reset agent trajectory if this is a new context (new task)
            # This prevents cross-task contamination where trajectory from
            # previous assessments would be included in the LLM prompt
            if context.context_id not in self.ctx_id_to_history:
                agent.reset()
                self.ctx_id_to_history[context.context_id] = []
                logger.info(f"Reset agent trajectory for new context: {context.context_id}")

            # Get user input (the task message)
            user_input = context.get_user_input()
            logger.info(f"Received task: {user_input[:100]}..." if len(user_input) > 100 else f"Received task: {user_input}")

            # Try to parse as JSON (for structured observation format)
            try:
                task_data = json.loads(user_input)
                observation = parse_observation(task_data)
            except json.JSONDecodeError:
                # Plain text instruction - no screenshot
                observation = None
                instruction = user_input

            if observation:
                instruction = observation["instruction"]
                screenshot_bytes = observation["screenshot"]
                accessibility_tree = observation.get("accessibility_tree")

                # Build observation dict for agent
                obs_for_agent = {"screenshot": screenshot_bytes}
                if accessibility_tree:
                    obs_for_agent["accessibility_tree"] = accessibility_tree

                # Log trajectory state before prediction (for debugging)
                traj_len = len(agent.observations)
                max_traj = agent.max_trajectory_length
                logger.info(
                    f"Trajectory state: {traj_len} steps in history, "
                    f"sending last {min(traj_len, max_traj)} to LLM"
                )

                logger.info(f"Calling PromptAgent ({MODEL}) with instruction: {instruction[:80]}...")
                response, actions = agent.predict(instruction, obs_for_agent)

                # Parse actions using core.py's robust parser directly on raw response
                # It handles JSON, pyautogui code, DONE/FAIL, and code blocks
                action = parse_actions(response)

                # Build response
                result = {
                    "action": action,
                    "reasoning": response,
                    "raw_response": response,
                    "done": action.get("op") == "done"
                }
                response_text = json.dumps(result)
            else:
                response_text = f"Received instruction: {instruction}. Please provide an observation (screenshot) to proceed."

            # Send response through event queue
            await event_queue.enqueue_event(
                new_agent_text_message(
                    response_text,
                    context_id=context.context_id
                )
            )

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            error_response = json.dumps({
                "error": str(e),
                "status": "failed"
            })
            await event_queue.enqueue_event(
                new_agent_text_message(
                    error_response,
                    context_id=context.context_id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        logger.info(f"Task cancelled: {context.context_id}")
        if context.context_id in self.ctx_id_to_history:
            del self.ctx_id_to_history[context.context_id]


# Backwards compatibility aliases
UnifiedAgentExecutor = PromptAgentExecutor
GPT4VAgentExecutor = PromptAgentExecutor


def create_app():
    """Create the A2A application for uvicorn."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", os.environ.get("PORT", "8080")))
    agent_url = os.getenv("AGENT_URL", f"http://{host}:{port}")

    logger.info(f"Creating White Agent A2A app (Model: {MODEL}, URL: {agent_url})")

    # Create agent card
    card = prepare_agent_card(agent_url, MODEL)

    # Create request handler with our executor
    request_handler = DefaultRequestHandler(
        agent_executor=PromptAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    # Health check endpoint (required for Cloud Run)
    async def health_check(request):
        return JSONResponse({
            "status": "healthy",
            "agent_type": "white",
            "protocol": "a2a",
            "model": MODEL,
            "observation_type": OBSERVATION_TYPE,
        })

    # Legacy agent-card endpoint for backwards compatibility
    async def agent_card_endpoint(request):
        return JSONResponse(card.model_dump(by_alias=True, exclude_none=True))

    # Stateless /decide endpoint for green agent compatibility
    # This mirrors the REST server's /decide endpoint
    async def decide_endpoint(request):
        """
        Stateless action decision endpoint for green agent.

        Creates fresh PromptAgent per request, rebuilds trajectory,
        and returns action + thought + trajectory_step.
        """
        try:
            body = await request.json()

            frame_id = body.get("frame_id", 0)
            image_png_b64 = body.get("image_png_b64", "")
            instruction = body.get("instruction", "")
            accessibility_tree = body.get("accessibility_tree")
            stuck_feedback = body.get("stuck_feedback")
            trajectory = body.get("trajectory", [])
            # Use model from request if provided, otherwise use default
            request_model = body.get("model", MODEL)

            logger.info(f"[/decide] Frame {frame_id}: Processing with {len(trajectory)} history steps, model={request_model}, obs_type={OBSERVATION_TYPE}, a11y_tree={'yes' if accessibility_tree else 'no'} ({len(accessibility_tree) if accessibility_tree else 0} chars)")

            # Create fresh agent for this request (stateless)
            agent = PromptAgent(
                model=request_model,
                temperature=TEMPERATURE,
                observation_type=OBSERVATION_TYPE,
                action_space=ACTION_SPACE,
                max_trajectory_length=MAX_TRAJECTORY_LENGTH,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
            )

            # Rebuild trajectory from request (for stateless operation)
            for step in trajectory:
                # Each step should have: accessibility_tree, action, thought
                step_a11y = step.get("accessibility_tree", "")
                step_action = step.get("action", {})
                step_thought = step.get("thought", "")

                # Add to agent's trajectory (without screenshot since we don't store those)
                # observations must be dicts with screenshot and accessibility_tree keys
                agent.observations.append({
                    "screenshot": None,
                    "accessibility_tree": step_a11y
                })
                agent.actions.append(str(step_action))
                agent.thoughts.append(step_thought)

            # Build current observation
            screenshot_bytes = base64.b64decode(image_png_b64) if image_png_b64 else None
            obs_for_agent = {}
            if screenshot_bytes:
                obs_for_agent["screenshot"] = screenshot_bytes
            if accessibility_tree:
                obs_for_agent["accessibility_tree"] = accessibility_tree

            # Inject stuck feedback if present
            final_instruction = instruction
            if stuck_feedback:
                logger.warning(f"[/decide] Stuck feedback for frame {frame_id}")
                final_instruction = f"{stuck_feedback}\n\nOriginal task: {instruction}"

            # Get prediction from LLM
            response, actions = agent.predict(final_instruction, obs_for_agent)

            # Parse action using robust parser
            parsed_action = parse_actions(response)

            # Build trajectory step for green agent to store
            trimmed_a11y = None
            if accessibility_tree:
                trimmed_a11y = accessibility_tree[:2000] if len(accessibility_tree) > 2000 else accessibility_tree

            trajectory_step = {
                "accessibility_tree": trimmed_a11y,
                "action": parsed_action,
                "thought": response,
            }

            logger.info(f"[/decide] Frame {frame_id}: Action={parsed_action.get('op', 'unknown')}")

            return JSONResponse({
                "action": parsed_action,
                "thought": response,
                "trajectory_step": trajectory_step,
            })

        except Exception as e:
            logger.error(f"[/decide] Failed: {e}", exc_info=True)
            error_action = {"op": "error", "args": {"message": str(e)}}
            error_step = {
                "accessibility_tree": None,
                "action": error_action,
                "thought": f"Error: {e}",
            }
            return JSONResponse({
                "action": error_action,
                "thought": f"Error: {e}",
                "trajectory_step": error_step,
            })

    # Get A2A routes and add custom routes
    a2a_routes = a2a_app.routes()
    custom_routes = [
        Route("/health", health_check, methods=["GET"]),
        Route("/agent-card", agent_card_endpoint, methods=["GET"]),
        Route("/decide", decide_endpoint, methods=["POST"]),
    ]

    # Create Starlette app with combined routes
    return Starlette(routes=a2a_routes + custom_routes)


# Module-level app for uvicorn
app = create_app()


def start_agent(host: str = "0.0.0.0", port: int = 8001):
    """
    Start the white agent server using A2A SDK.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    logger.info(f"Starting White Agent (A2A) on {host}:{port}")
    logger.info(f"Model: {MODEL}, Observation type: {OBSERVATION_TYPE}")

    # Use the module-level app which includes /decide endpoint
    # This ensures green agent can call /decide for stateless operation
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", os.environ.get("PORT", "8001")))
    start_agent(host=host, port=port)
