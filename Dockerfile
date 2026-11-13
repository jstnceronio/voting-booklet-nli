FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

ENTRYPOINT ["uv", "run", "--no-sync", "voting-nli"]
