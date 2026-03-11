# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for ChromaDB + openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create directories for persistent data
RUN mkdir -p /app/chroma_db /app/cache /app/data

# ── Environment defaults (override with docker run -e) ────────────────────────
ENV PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    CACHE_DIR=/app/cache \
    DATA_PATH=/app/data/online_retail_II.xlsx \
    GEMINI_MODEL=gemini-1.5-flash \
    CACHE_ENABLED=true \
    LOG_LEVEL=INFO

# ── Entry point ───────────────────────────────────────────────────────────────
# Default: interactive CLI
ENTRYPOINT ["python", "main.py"]
# Single query example:
#   docker run ... ecommerce-analyst --query "Top 5 products by revenue"
