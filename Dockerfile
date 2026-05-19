FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential pkg-config libcairo2-dev libpango1.0-dev \
    texlive texlive-latex-extra texlive-fonts-extra dvisvgm \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m runner && chown -R runner /app
USER runner
ENV PORT=8080
CMD exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}