# Optimized ML Build - Multimodal Price Predictor
FROM python:3.10-slim

# Install system dependencies for FAISS and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD THE HEAVY MODELS ---
# This caches the 3GB backbone models inside the image,
# so your live server starts instantly.
RUN python -c "from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, SiglipVisionModel; \
    AutoTokenizer.from_pretrained('microsoft/deberta-v3-large'); \
    AutoModel.from_pretrained('microsoft/deberta-v3-large'); \
    AutoImageProcessor.from_pretrained('google/siglip2-so400m-patch14-384'); \
    SiglipVisionModel.from_pretrained('google/siglip2-so400m-patch14-384')"

# Copy the minimum structure required for the production web app
COPY webapp/ webapp/
COPY checkpoints/ checkpoints/
COPY data/ data/
COPY steps/ steps/
COPY config.py .
COPY README.md .

# Hugging Face default port is 7860
EXPOSE 7860
ENV PORT=7860

# Launch server
CMD ["python", "webapp/server.py"]
