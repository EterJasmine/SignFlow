FROM node:24-bookworm-slim AS frontend
WORKDIR /frontend
COPY apps/web/package*.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GESTUREFLOW_DATA=/app/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY apps/api/ apps/api/
COPY apps/__init__.py apps/__init__.py
COPY packages/ packages/
COPY training/ training/
COPY models/*.task models/
COPY --from=frontend /frontend/dist apps/web/dist/
RUN useradd --create-home app && mkdir -p /app/data && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=3)"
CMD ["python","-m","uvicorn","apps.api.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
