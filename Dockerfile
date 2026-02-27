# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps for Manim + pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libcairo2-dev \
    libpango1.0-dev \
    pkg-config \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir manim

# ── Final stage ───────────────────────────────────────────────────────────────
FROM builder AS final

COPY . /app
RUN mkdir -p /app/outputs /app/tmp

ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/app/outputs
ENV TEMP_DIR=/app/tmp

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
