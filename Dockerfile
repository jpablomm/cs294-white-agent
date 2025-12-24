FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY --chown=agent:agent pyproject.toml README.md ./
COPY --chown=agent:agent src src
COPY --chown=agent:agent white_agent white_agent

RUN uv sync

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
