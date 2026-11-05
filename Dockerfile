FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY src/voting_cli/cli.py ./

ENTRYPOINT ["uv", "run", "--no-sync", "python", "main.py"]
