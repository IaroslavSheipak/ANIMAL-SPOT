#!/bin/bash
# Best configuration training script (99.49% test accuracy)
# This is the optimal configuration discovered through hyperparameter optimization
#
# Usage: ./train_best_config.sh
# Or with custom paths:
#   DATA_DIR=/path/to/data OUTPUT_DIR=/path/to/output ./train_best_config.sh

set -e  # Exit on error

# Activate virtual environment if exists (optional)
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
elif [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

# Set paths (use environment variables or defaults)
DATA_DIR="${DATA_DIR:-./data}"
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_BASE="${OUTPUT_DIR:-./outputs}/best_config_$(date +%Y-%m-%d_%H-%M)"

# Create output directories
mkdir -p $OUTPUT_BASE/model
mkdir -p $OUTPUT_BASE/checkpoints
mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/summaries
mkdir -p $CACHE_DIR

# Optimal training parameters (99.49% accuracy)
BACKBONE="resnet"
RESNET_SIZE=18
BATCH_SIZE=64
LR=3e-4
MAX_EPOCHS=100
EARLY_STOPPING=25
NUM_WORKERS=8

# Run training with best config
python3 ../ANIMAL-SPOT/main.py \
    --data_dir $DATA_DIR \
    --cache_dir $CACHE_DIR \
    --model_dir $OUTPUT_BASE/model \
    --checkpoint_dir $OUTPUT_BASE/checkpoints \
    --log_dir $OUTPUT_BASE/logs \
    --summary_dir $OUTPUT_BASE/summaries \
    --backbone $BACKBONE \
    --resnet $RESNET_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --scheduler onecycle \
    --label_smoothing 0.1 \
    --weighted_sampling \
    --sampling_strategy sqrt_inverse_freq \
    --rare_class_boost 1.5 \
    --max_train_epochs $MAX_EPOCHS \
    --early_stopping_patience_epochs $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --augmentation \
    --start_from_scratch \
    --epochs_per_eval 2 \
    --num_classes 13 \
    --sequence_len 1000 \
    --sr 44100 \
    --n_fft 1024 \
    --hop_length 172 \
    --n_freq_bins 256 \
    --freq_compression linear \
    --fmin 500 \
    --fmax 10000 \
    --min_max_norm \
    --filter_broken_audio

echo "Training complete! Results saved to: $OUTPUT_BASE"
echo "View TensorBoard: tensorboard --logdir $OUTPUT_BASE/summaries/"
echo ""
echo "Expected results: ~99.49% test accuracy"
