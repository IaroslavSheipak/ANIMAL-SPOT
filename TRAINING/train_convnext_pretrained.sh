#!/bin/bash
# Training script for ConvNeXt Tiny with ImageNet pretrained weights
# Purpose: Improved accuracy on cetacean vocalization classification
# Expected: 98.5%+ overall accuracy, better performance on rare classes

set -e  # Exit on error

# Activate virtual environment
source ../animal_spot_env/bin/activate

# Set paths
DATA_DIR="/mnt/c/Users/Iaroslav/CETACEANS/training_data"
OUTPUT_BASE="/mnt/c/Users/Iaroslav/CETACEANS/outputs_convnext_pretrained_$(date +%Y-%m-%d_%H-%M)"

# Create output directories
mkdir -p $OUTPUT_BASE/model
mkdir -p $OUTPUT_BASE/checkpoints
mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/summaries

# Training parameters
BACKBONE="convnext"
PRETRAINED="--pretrained"  # Use ImageNet pretrained weights
BATCH_SIZE=8
LR=1e-6  # Lower LR for ConvNeXt (5x lower than ResNet)
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
    $PRETRAINED \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_train_epochs $MAX_EPOCHS \
    --early_stopping_patience_epochs $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --augmentation \
    --start_from_scratch \
    --epochs_per_eval 2 \
    --lr_patience_epochs 10 \
    --lr_decay_factor 0.5 \
    --beta1 0.9 \
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
    --max_pool 2 \
    --debug

echo "Training complete! Results saved to: $OUTPUT_BASE"
echo "View TensorBoard: tensorboard --logdir $OUTPUT_BASE/summaries/"
echo "Per-class metrics available in TensorBoard under 'per_class_accuracy' tab"
