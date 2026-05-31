# ImpellerVision — FastAPI + MobileNetV2 (CPU inference imaji)
FROM python:3.11-slim

# OpenCV / Pillow runtime bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Torch'u CPU-only kur (imaj boyutunu kucuk tutar). Mimariye gore:
#   x86_64  -> PyTorch CPU index'i (CUDA'siz, kucuk)
#   aarch64 -> PyPI (ARM Linux wheel'leri zaten CPU)
COPY requirements.txt .
RUN ARCH="$(uname -m)"; \
    if [ "$ARCH" = "x86_64" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    else \
        pip install --no-cache-dir torch torchvision; \
    fi \
    && pip install --no-cache-dir \
        numpy pillow opencv-python-headless scikit-learn matplotlib \
        "grad-cam>=1.5" fastapi "uvicorn[standard]" python-multipart python-dotenv

# Uygulama + model + cikarim modulu
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
