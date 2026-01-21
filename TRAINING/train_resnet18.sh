#!/bin/bash
# Training script for ResNet-18 baseline (vanilla ANIMAL-SPOT)
# Purpose: Baseline comparison for cetacean vocalization classification

set -e  # Exit on error

# Activate virtual environment if exists (optional)
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
elif [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

# Set paths (use environment variables or defaults)
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_BASE="${OUTPUT_DIR:-./outputs}/resnet18_$(date +%Y-%m-%d_%H-%M)"

# Create output directories
mkdir -p $OUTPUT_BASE/model
mkdir -p $OUTPUT_BASE/checkpoints
mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/summaries

# Training parameters
BACKBONE="resnet"
RESNET_SIZE=18
BATCH_SIZE=8
LR=5e-6
MAX_EPOCHS=150
EARLY_STOPPING=20
NUM_WORKERS=8

# Run training
python3 ../ANIMAL-SPOT/main.py \
    --data_dir $DATA_DIR \
    --model_dir $OUTPUT_BASE/model \
    --checkpoint_dir $OUTPUT_BASE/checkpoints \
    --log_dir $OUTPUT_BASE/logs \
    --summary_dir $OUTPUT_BASE/summaries \
    --backbone $BACKBONE \
    --resnet $RESNET_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_train_epochs $MAX_EPOCHS \
    --early_stopping_patience_epochs $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --augmentation \
    --start_from_scratch \
    --epochs_per_eval 2 \
    --lr_patience_epochs 8 \
    --lr_decay_factor 0.5 \
    --beta1 0.5 \
    --num_classes 13 \
    --sequence_len 500 \
    --sr 44100 \
    --n_fft 1024 \
    --hop_length 172 \
    --n_freq_bins 256 \
    --freq_compression linear \
    --fmin 500 \
    --fmax 10000 \
    --min_max_norm \
    --filter_broken_audio \
    --debug

echo "Training complete! Results saved to: $OUTPUT_BASE"
echo "View TensorBoard: tensorboard --logdir $OUTPUT_BASE/summaries/"
