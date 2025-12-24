#!/usr/bin/env python3
"""
White Agent - REST API Server (Stateless for Cloud Run)

FastAPI-based REST server for custom orchestrator integration.
Uses PromptAgent for multi-model support (GPT-4V, Claude, Gemini, Qwen, etc.)

STATELESS DESIGN:
- No global agent state - fresh PromptAgent created per request
- Trajectory passed in request, rebuilt on each call
- Supports Cloud Run auto-scaling and concurrent requests
- Green agent owns trajectory state
"""

import logging
import uuid
import base64
from typing import Dict, Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from white_agent.prompt_agent import PromptAgent
from white_agent.core import parse_observation, parse_actions
from white_agent.config import (
    MODEL,
    TEMPERATURE,
    OBSERVATION_TYPE,
    ACTION_SPACE,
    MAX_TRAJECTORY_LENGTH,
    MAX_TOKENS,
    TOP_P,
    WHITE_AGENT_HOST,
    WHITE_AGENT_PORT,
    get_agent_url,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NOTE: This server is STATELESS for Cloud Run compatibility.
# - No global agent instance
# - No conversation contexts stored server-side
# - Trajectory is passed in each request by the green agent
# - Each request creates a fresh PromptAgent

# FastAPI app
app = FastAPI(
    title=f"White Agent (REST, Stateless) - {MODEL}",
    description=f"Stateless white agent server using PromptAgent with {MODEL}. "
                f"Designed for Cloud Run auto-scaling."
)


# =============================================================================
# Request/Response Models
# =============================================================================

class TrajectoryStep(BaseModel):
    """Single step in trajectory history (text only, no screenshots)."""
    accessibility_tree: Optional[str] = None  # Trimmed a11y tree summary
    action: Any  # The parsed action {op: str, args: dict}
    thought: str  # LLM response text


class Observation(BaseModel):
    """
    Direct observation request - includes trajectory for stateless operation.

    The green agent passes the accumulated trajectory with each request.
    This allows any white agent instance to handle any request.
    """
    frame_id: int
    image_png_b64: str
    instruction: str = ""
    accessibility_tree: Optional[str] = None
    done: bool = False
    stuck_feedback: Optional[str] = None  # Feedback when agent is stuck in a loop
    # Trajectory passed by green agent (no screenshots - text only)
    trajectory: List[TrajectoryStep] = []
    # Model override (optional) - if provided, use this model instead of default
    model: Optional[str] = None


class DecideResponse(BaseModel):
    """Response model - returns data for green agent to store."""
    action: Dict[str, Any]  # Parsed action {op: str, args: dict}
    thought: str  # Raw LLM response
    # Processed observation data for green agent to store in trajectory
    trajectory_step: TrajectoryStep


class A2ATask(BaseModel):
    """Task request model (A2A-compatible format)"""
    task_id: str
    context_id: Optional[str] = None
    message: str
    metadata: Optional[Dict[str, Any]] = None
    # Trajectory for stateless operation
    trajectory: List[TrajectoryStep] = []


class A2AMessage(BaseModel):
    """Response message model (A2A-compatible format)"""
    message_id: str
    task_id: str
    context_id: Optional[str] = None
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class AgentCardResponse(BaseModel):
    """Agent card for discovery"""
    name: str
    description: str
    url: str
    version: str
    model: str


# =============================================================================
# Helper Functions
# =============================================================================

def create_agent(model_override: Optional[str] = None) -> PromptAgent:
    """Create a fresh PromptAgent instance for this request."""
    use_model = model_override or MODEL
    return PromptAgent(
        model=use_model,
        temperature=TEMPERATURE,
        observation_type=OBSERVATION_TYPE,
        action_space=ACTION_SPACE,
        max_trajectory_length=MAX_TRAJECTORY_LENGTH,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )


def rebuild_trajectory(agent: PromptAgent, trajectory: List[TrajectoryStep]) -> None:
    """
    Rebuild agent trajectory from request data.

    Only uses the last MAX_TRAJECTORY_LENGTH steps.
    Screenshots are NOT included in trajectory - only current observation has screenshot.
    """
    steps_to_use = trajectory[-MAX_TRAJECTORY_LENGTH:] if trajectory else []

    for step in steps_to_use:
        # Past observations don't need screenshots - only current does
        agent.observations.append({
            "screenshot": None,
            "accessibility_tree": step.accessibility_tree,
        })
        agent.actions.append(step.action)
        agent.thoughts.append(step.thought)


def build_agent_url() -> str:
    """Build the agent URL from environment."""
    return get_agent_url()


# =============================================================================
# Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup - stateless server, no initialization needed"""
    logger.info(f"White Agent (REST, Stateless) starting - model={MODEL}")
    logger.info("Stateless mode: fresh agent created per request")
    logger.info("Cloud Run optimized: supports auto-scaling and concurrent requests")


@app.get("/health")
def health():
    """Health check - stateless service."""
    return {
        "status": "healthy",
        "protocol": "rest",
        "model": MODEL,
        "observation_type": OBSERVATION_TYPE,
        "stateless": True,
        "cloud_run_optimized": True,
    }


@app.get("/status")
def status():
    """Status endpoint"""
    return {
        "status": "running",
        "model": MODEL,
        "protocol": "rest",
        "stateless": True,
    }


@app.get("/agent-card")
@app.get("/.well-known/agent-card.json")
def get_agent_card() -> AgentCardResponse:
    """Agent card for discovery"""
    return AgentCardResponse(
        name=f"OSWorld Agent ({MODEL})",
        description=f"Stateless white agent for desktop automation using {MODEL}",
        url=build_agent_url(),
        version="3.0.0",  # Version bump for stateless architecture
        model=MODEL
    )


@app.post("/decide", response_model=DecideResponse)
def decide(obs: Observation) -> DecideResponse:
    """
    Stateless action decision endpoint.

    - Creates fresh PromptAgent per request
    - Rebuilds trajectory from request payload
    - Returns action + data for caller to store

    This design supports Cloud Run auto-scaling and concurrent requests.
    Any instance can handle any request since there's no server-side state.
    """
    try:
        # Create fresh agent for this request (no shared state)
        # Use model from request if provided, otherwise use default
        agent = create_agent(model_override=obs.model)

        # Rebuild trajectory from request
        rebuild_trajectory(agent, obs.trajectory)

        logger.info(f"Frame {obs.frame_id}: Processing with {len(obs.trajectory)} history steps, model={agent.model}")

        # Build current observation (this one HAS the screenshot)
        # NOTE: PromptAgent.predict() expects screenshot as raw bytes, not base64 string
        # It will encode to base64 internally via encode_image()
        screenshot_bytes = base64.b64decode(obs.image_png_b64)
        obs_for_agent = {"screenshot": screenshot_bytes}
        if obs.accessibility_tree:
            obs_for_agent["accessibility_tree"] = obs.accessibility_tree

        # Inject stuck feedback if present
        instruction = obs.instruction
        if obs.stuck_feedback:
            logger.warning(f"[LoopDetection] Stuck feedback for frame {obs.frame_id}")
            instruction = f"{obs.stuck_feedback}\n\nOriginal task: {obs.instruction}"

        # Get prediction from LLM
        response, actions = agent.predict(instruction, obs_for_agent)

        # Parse action using robust parser
        parsed_action = parse_actions(response)

        # Build trajectory step for green agent to store
        # Trim accessibility tree to reduce payload size
        trimmed_a11y = None
        if obs.accessibility_tree:
            trimmed_a11y = obs.accessibility_tree[:2000] if len(obs.accessibility_tree) > 2000 else obs.accessibility_tree

        trajectory_step = TrajectoryStep(
            accessibility_tree=trimmed_a11y,
            action=parsed_action,
            thought=response,
        )

        logger.info(f"Frame {obs.frame_id}: Action={parsed_action.get('op', 'unknown')}")

        return DecideResponse(
            action=parsed_action,
            thought=response,
            trajectory_step=trajectory_step,
        )

    except Exception as e:
        logger.error(f"Decide failed: {e}", exc_info=True)
        error_action = {"op": "error", "args": {"message": str(e)}}
        error_step = TrajectoryStep(
            accessibility_tree=None,
            action=error_action,
            thought=f"Error: {e}",
        )
        return DecideResponse(
            action=error_action,
            thought=f"Error: {e}",
            trajectory_step=error_step,
        )


@app.post("/task")
def handle_task(task: A2ATask) -> A2AMessage:
    """
    Handle task request (A2A-compatible format, stateless).

    Expected task.metadata:
    {
        "observation": {
            "frame_id": int,
            "image_png_b64": str,
            "instruction": str,
            "done": bool
        }
    }

    Trajectory is passed in task.trajectory for stateless operation.
    """
    context_id = task.context_id or task.task_id

    try:
        # Create fresh agent for this request
        agent = create_agent()

        # Rebuild trajectory from request
        rebuild_trajectory(agent, task.trajectory)

        # Parse observation from metadata
        if not task.metadata or "observation" not in task.metadata:
            raise ValueError("Task must have observation in metadata")

        observation = parse_observation(task.metadata)
        instruction = observation["instruction"]
        screenshot_bytes = observation["screenshot"]
        accessibility_tree = observation.get("accessibility_tree")

        step = len(task.trajectory)
        logger.info(f"Step {step}: Processing observation for '{instruction[:80]}...'")

        # Build observation for agent
        obs_for_agent = {"screenshot": screenshot_bytes}
        if accessibility_tree:
            obs_for_agent["accessibility_tree"] = accessibility_tree

        # Get prediction
        response, actions = agent.predict(instruction, obs_for_agent)

        # Parse actions using robust parser
        parsed_action = parse_actions(response)

        # Build trajectory step for response
        trimmed_a11y = None
        if accessibility_tree:
            trimmed_a11y = accessibility_tree[:2000] if len(accessibility_tree) > 2000 else accessibility_tree

        trajectory_step = TrajectoryStep(
            accessibility_tree=trimmed_a11y,
            action=parsed_action,
            thought=response,
        )

        # Check if done
        task_done = parsed_action.get("op") == "done" or observation.get("done", False)

        return A2AMessage(
            message_id=str(uuid.uuid4()),
            task_id=task.task_id,
            context_id=context_id,
            role="agent",
            content=f"Step {step}: {response}",
            metadata={
                "action": parsed_action,
                "step": step,
                "done": task_done,
                "raw_response": response,
                "trajectory_step": trajectory_step.dict(),
            }
        )

    except Exception as e:
        logger.error(f"Task processing failed: {e}", exc_info=True)
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            task_id=task.task_id,
            context_id=context_id,
            role="agent",
            content=f"Error: {e}",
            metadata={"status": "error", "error": str(e)}
        )


@app.post("/reset")
def reset():
    """Reset - no-op for stateless service."""
    logger.info("Reset called (no-op for stateless service)")
    return {"status": "ok", "message": "Stateless service - nothing to reset", "model": MODEL}


@app.get("/debug/contexts")
def debug_contexts():
    """Debug endpoint - stateless service has no contexts."""
    return {
        "status": "stateless",
        "message": "Stateless service - contexts managed by green agent",
        "model": MODEL,
    }


@app.get("/debug/trajectory")
def debug_trajectory():
    """Debug endpoint - trajectory is managed by green agent."""
    return {
        "status": "stateless",
        "message": "Trajectory managed by green agent, not white agent. "
                   "Pass trajectory in request to /decide endpoint.",
        "model": MODEL,
        "max_trajectory_length": MAX_TRAJECTORY_LENGTH,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=WHITE_AGENT_HOST, port=WHITE_AGENT_PORT)
