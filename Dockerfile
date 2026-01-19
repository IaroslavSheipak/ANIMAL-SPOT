# ANIMAL-SPOT Docker Image
# Multi-stage build for smaller final image

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create requirements file
COPY requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt

# ============================================
# Stage 2: Production image
# ============================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="CETACEANS Project" \
      version="2.0" \
      description="ANIMAL-SPOT: Deep Learning for Bioacoustic Signal Detection"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANIMAL_SPOT_HOME=/app \
    PATH="/root/.local/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create app directory structure
WORKDIR /app

# Copy source code
COPY ANIMAL-SPOT/ /app/ANIMAL-SPOT/
COPY TRAINING/ /app/TRAINING/
COPY PREDICTION/ /app/PREDICTION/
COPY EVALUATION/ /app/EVALUATION/

# Create directories for data and outputs
RUN mkdir -p /data /output /cache /models

# Set Python path
ENV PYTHONPATH="/app/ANIMAL-SPOT:$PYTHONPATH"

# Default command
CMD ["python", "-c", "print('ANIMAL-SPOT container ready. Use docker run with specific commands.')"]

# ============================================
# Usage Examples:
# ============================================
# Build:
#   docker build -t animal-spot:latest .
#
# Training:
#   docker run --gpus all -v /path/to/data:/data -v /path/to/output:/output \
#     animal-spot:latest python /app/ANIMAL-SPOT/main.py \
#     --data_dir /data --model_dir /output/model \
#     --checkpoint_dir /output/checkpoints --log_dir /output/logs \
#     --summary_dir /output/summaries --num_classes 13 --max_train_epochs 150
#
# Prediction:
#   docker run --gpus all -v /path/to/audio:/data -v /path/to/model:/models \
#     -v /path/to/output:/output animal-spot:latest \
#     python /app/ANIMAL-SPOT/predict.py --model /models/ANIMAL-SPOT.pk \
#     --audio_dir /data --output_dir /output
#
# Two-stage prediction:
#   docker run --gpus all -v /path/to/audio:/data -v /path/to/models:/models \
#     -v /path/to/output:/output animal-spot:latest \
#     python /app/ANIMAL-SPOT/two_stage_predict.py \
#     --binary_model /models/binary_detector.pk \
#     --multiclass_model /models/multiclass.pk \
#     --input_dir /data --output_dir /output
#
# Optuna tuning:
#   docker run --gpus all -v /path/to/data:/data -v /path/to/output:/output \
#     animal-spot:latest python /app/ANIMAL-SPOT/optuna_tuning.py \
#     --data_dir /data --output_dir /output --num_classes 13 --n_trials 50
