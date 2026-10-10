# KudiGuard API + WhatsApp webhook (CPU). N-ATLAS runs separately on a GPU host: set KUDIGUARD_LLM_URL.
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir ".[semantic,speech]" \
    && useradd --create-home --uid 10001 kudiguard \
    && mkdir -p /app/var && chown kudiguard /app/var

USER kudiguard
ENV KUDIGUARD_DATA_DIR=/app/data \
    KUDIGUARD_DB_PATH=/app/var/kudiguard.db \
    KUDIGUARD_VOICE_DIR=/app/var/voice \
    KUDIGUARD_MODEL_CACHE=/models \
    KUDIGUARD_EMBEDDINGS_CACHE=/models/fastembed \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/healthz')"
CMD ["kudiguard", "serve", "--host", "0.0.0.0", "--port", "8010"]
