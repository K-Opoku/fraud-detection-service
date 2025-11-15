FROM python:3.12.1-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version .

RUN uv sync --locked


COPY src/main.py .
COPY src/final_model_pipeline.pkl .

EXPOSE 8000


ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]