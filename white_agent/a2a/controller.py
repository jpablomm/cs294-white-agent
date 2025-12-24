"""
White Agent AgentBeats Controller with Catch-All Proxy

This module patches the earthshaker controller to add:
- Catch-all proxy for /to_agent/{agent_id}/{path} routes
- This allows the green agent to call /decide on the white agent

Usage in Dockerfile:
    CMD ["python", "-m", "white_agent.a2a.controller"]
"""

import os
import logging

import httpx
from fastapi import Request
from fastapi.responses import Response, JSONResponse

# Import the earthshaker controller app - this registers it
from agentbeats.controller import app as earthshaker_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_agent_port_by_id(agent_id: str) -> int | None:
    """Get the port for a specific agent by ID."""
    port_file = os.path.join(".ab", "agents", agent_id, "port")
    if os.path.exists(port_file):
        try:
            with open(port_file, "r") as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return None
    return None


# Patch: Add catch-all proxy for /to_agent/{agent_id}/{path}
# This allows green agent to call /decide, /health, etc.
@earthshaker_app.api_route(
    "/to_agent/{agent_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_to_agent(agent_id: str, path: str, request: Request):
    """
    Catch-all proxy for agent endpoints.

    Proxies requests like /to_agent/{id}/decide to localhost:{agent_port}/decide
    """
    agent_port = get_agent_port_by_id(agent_id)

    if agent_port is None:
        logger.warning(f"Agent {agent_id} not ready yet (no port file)")
        return JSONResponse(
            {"error": f"Agent {agent_id} is not ready yet. Try again in a few seconds."},
            status_code=503
        )

    # Build target URL
    agent_url = f"http://localhost:{agent_port}/{path}"
    logger.info(f"Proxying {request.method} /to_agent/{agent_id}/{path} -> {agent_url}")

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            # Forward the request
            response = await client.request(
                method=request.method,
                url=agent_url,
                content=await request.body(),
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ('host', 'content-length')
                },
            )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    k: v for k, v in response.headers.items()
                    if k.lower() not in ('content-length', 'transfer-encoding')
                },
            )
    except httpx.ConnectError:
        logger.warning(f"Agent {agent_id} port {agent_port} not accepting connections yet")
        return JSONResponse(
            {"error": f"Agent {agent_id} is starting up. Try again in a few seconds."},
            status_code=503
        )
    except httpx.TimeoutException:
        logger.error(f"Request to agent {agent_id} timed out")
        return JSONResponse(
            {"error": "Agent request timed out"},
            status_code=504
        )
    except Exception as e:
        logger.error(f"Error proxying to agent {agent_id}: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


logger.info("Catch-all proxy routes added to earthshaker controller")
logger.info("  /to_agent/{agent_id}/{path} -> proxies to agent localhost port")


def run_ctrl():
    """
    Run the enhanced controller.

    This is equivalent to `agentbeats run_ctrl` but with the patched routes.
    """
    from agentbeats.controller import main as controller_main
    controller_main()


if __name__ == "__main__":
    run_ctrl()
