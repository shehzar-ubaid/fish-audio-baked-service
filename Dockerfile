FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 [cite: 1]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 [cite: 1]

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app

# Step 1: Clone Fish-Speech Code (Important)
RUN git clone https://github.com/fishaudio/fish-speech.git . && \
    pip3 install --no-cache-dir -e . 

# Step 2: Install Dependencies [cite: 2]
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt [cite: 2]

# Step 3: Bake Model Weights (No Token Needed)
RUN pip3 install modelscope && \
    python3 -c "from modelscope import snapshot_download; \
    snapshot_download('fishaudio/fish-speech-1.4', local_dir='checkpoints/s1-mini')"

# Step 4: Pre-download NLTK data for 100k Chars Tokenization
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY handler.py .
CMD ["python3", "-u", "handler.py"] [cite: 3]