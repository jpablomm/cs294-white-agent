# CS294 White Agent

OSWorld White Agent - A2A compliant agent for desktop automation tasks.

This agent receives observations (screenshots, accessibility trees) and returns actions for desktop automation.

## Features

- Multi-model support: GPT-4V, Claude, Gemini, Qwen
- A2A protocol compliant
- Vision-language reasoning for desktop automation

## Environment Variables

- `MODEL`: LLM model to use (default: gpt-4o)
- `OPENAI_API_KEY`: OpenAI API key (for GPT models)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Claude models)

## Running Locally

```bash
uv sync
uv run src/server.py
```

## Running with Docker

```bash
docker build -t white-agent .
docker run -p 9009:9009 -e OPENAI_API_KEY=$OPENAI_API_KEY white-agent
```
