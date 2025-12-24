#!/usr/bin/env python3
"""
White Agent - A2A Protocol Server for OSWorld Desktop Automation

Uses PromptAgent for multi-model support (GPT-4V, Claude, Gemini, Qwen).
Exposes /decide endpoint for green agent compatibility.
"""

import argparse
import os
import uvicorn

# Use the create_app from white_agent.a2a.server which includes /decide endpoint
from white_agent.a2a.server import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the White Agent A2A server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card (ignored, set via env)")
    args = parser.parse_args()

    # Set AGENT_URL environment variable if --card-url provided
    if args.card_url:
        os.environ["AGENT_URL"] = args.card_url

    # Create the app with A2A + /decide endpoint support
    app = create_app()

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
