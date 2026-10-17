# Single-VM image. Untested here (no Docker on the dev machine); see README "Deploy".
FROM python:3.13-slim
WORKDIR /app
# OpenCV/Docling runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HOST=0.0.0.0 SPEND_FILE=/app/state/spend.json PYTHONIOENCODING=utf-8
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["python", "server.py"]
