# Multi-stage Dockerfile for full-stack deployment (Hugging Face Spaces / Render / Railway)
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# System deps for OpenCV headless & Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY models.py evaluate.py ./
COPY backend/ ./backend/
COPY checkpoints/ ./checkpoints/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV CHECKPOINTS_DIR=/app/checkpoints
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
