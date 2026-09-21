FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src

EXPOSE 8002

# Default: mock provider so the image boots without secrets;
# override LLM_PROVIDER=openai + OPENAI_API_KEY at runtime for real calls.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8002"]
