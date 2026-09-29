FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable \
 # Bake tokenizer files into the image so runs never download them.
 && /opt/venv/bin/python -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('cl100k_base', 'o200k_base')]"


FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 # The runner mounts the workspace owned by another uid; git must still read it.
 && git config --system --add safe.directory '*'

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/tiktoken /opt/tiktoken
ENV PATH=/opt/venv/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["qdrant-md-sync"]
