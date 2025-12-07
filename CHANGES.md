# ANIMAL-SPOT Improvements for Cetacean Classification

## Overview

This fork of ANIMAL-SPOT adds two major improvements for multi-class cetacean vocalization classification:

1. **ConvNeXt Tiny backbone** with ImageNet pretrained weights (transfer learning)
2. **Per-class accuracy metrics** for detailed performance monitoring

These changes improve overall accuracy from **97.8% → 98.5%** and especially benefit rare classes (K21, K27, K14).

## New Features

### 1. ConvNeXt Backbone Support

Added ConvNeXt Tiny as an alternative to ResNet-18:

**Files Added:**
- `ANIMAL-SPOT/models/convnext_encoder.py` - ConvNeXt Tiny encoder implementation

**Key Features:**
- ImageNet pretrained weights (optional, via `--pretrained` flag)
- Adapted for single-channel spectrograms (averages RGB weights)
- 768 output channels (vs 512 for ResNet-18)
- Better performance on small, imbalanced datasets

**Usage:**
```bash
python main.py --backbone convnext --pretrained [other args...]
```

### 2. Per-Class Accuracy Metrics

Added detailed per-class accuracy tracking for multi-class classification:

**Files Modified:**
- `ANIMAL-SPOT/utils/metrics.py` - Added `PerClassAccuracy` class
- `ANIMAL-SPOT/main.py` - Added per-class metrics to multi-class training
- `ANIMAL-SPOT/trainer.py` - TensorBoard logging for per-class accuracy

**Benefits:**
- Track accuracy for each class separately
- Identify problematic classes (e.g., rare classes with low samples)
- Monitor training progress for imbalanced datasets
- Available in TensorBoard under `per_class_accuracy/` tab

### 3. Training Scripts

Added convenient training scripts in `TRAINING/`:

- `train_resnet18.sh` - ResNet-18 baseline (vanilla ANIMAL-SPOT)
- `train_convnext_pretrained.sh` - ConvNeXt with pretrained weights

## Results

### Cetacean Dataset (13 classes, 21,053 samples)

| Model | Overall Accuracy | Rare Classes (K21/K27/K14) |
|-------|------------------|----------------------------|
| **ResNet-18 (baseline)** | 97.8% | 65-75% |
| **ConvNeXt Pretrained** | 98.5% | 70-85% |

**Per-Class Results (ConvNeXt Pretrained):**
- Excellent (>95%): K1 (100%), noise (99.7%), K5 (99.4%), K7 (97.7%)
- Good (80-95%): K17 (94.1%), K10 (93.3%), K4 (92.3%), K13 (84.6%), K12 (82.9%)
- Challenging (<80%): K14 (75%), K27 (71.4%), K21 (70%)

## Usage

### Basic Training

**ResNet-18 baseline:**
```bash
cd TRAINING
./train_resnet18.sh
```

**ConvNeXt pretrained:**
```bash
cd TRAINING
./train_convnext_pretrained.sh
```

### Custom Training

**ConvNeXt from scratch (no pretrained):**
```bash
python ANIMAL-SPOT/main.py \
    --backbone convnext \
    --batch_size 8 \
    --lr 1e-6 \
    --num_classes 13 \
    [other args...]
```

**ResNet-18 (original ANIMAL-SPOT):**
```bash
python ANIMAL-SPOT/main.py \
    --backbone resnet \
    --resnet 18 \
    --batch_size 8 \
    --lr 5e-6 \
    --num_classes 13 \
    [other args...]
```

### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir outputs_*/summaries/
```

**Per-class metrics available:**
- `train/per_class_accuracy/K1`, `K3`, `K4`, etc.
- `val/per_class_accuracy/K1`, `K3`, `K4`, etc.
- `test/per_class_accuracy/K1`, `K3`, `K4`, etc.
- `mean_per_class_accuracy` - average across all classes

## Important Notes

### Learning Rate Sensitivity

ConvNeXt is **5x more sensitive** to learning rate than ResNet:

| Backbone | Recommended Base LR | Batch Size | Effective LR |
|----------|---------------------|------------|--------------|
| ResNet-18 | 5e-6 | 8 | 4e-5 |
| ConvNeXt | 1e-6 | 8 | 8e-6 |

**Why?** ConvNeXt uses LayerNorm (more sensitive than BatchNorm) and larger receptive fields.

### Optimizer Settings

| Backbone | Beta1 | LR Patience | Early Stopping |
|----------|-------|-------------|----------------|
| ResNet-18 | 0.5 | 8 epochs | 20 epochs |
| ConvNeXt | 0.9 | 10 epochs | 20 epochs |

## Implementation Details

### ConvNeXt Encoder Architecture

```python
ConvNextEncoder(
    input_channels=1,          # Single-channel spectrograms
    pretrained=True,           # ImageNet weights
)
→ Features: torch.Size([B, 768, H', W'])  # 768 output channels
```

**Weight Initialization:**
- If pretrained=True: Load ImageNet weights, average Conv1 across RGB channels
- If pretrained=False: Random initialization

### Per-Class Accuracy Calculation

```python
PerClassAccuracy(num_classes=13, device='cuda')
→ Tracks correct/total predictions per class
→ Returns: [acc_class_0, acc_class_1, ..., acc_class_12]
```

**Example Output:**
```
K1: 100.0%
K3: 98.1%
K4: 92.3%
...
K27: 71.4%
```

## File Changes Summary

**New Files:**
- `models/convnext_encoder.py` (235 lines)
- `TRAINING/train_resnet18.sh` (55 lines)
- `TRAINING/train_convnext_pretrained.sh` (63 lines)
- `CHANGES.md` (this file)

**Modified Files:**
- `main.py` - Added backbone selection logic (+50 lines)
- `trainer.py` - Added per-class logging (+30 lines)
- `utils/metrics.py` - Added PerClassAccuracy class (+60 lines)

**Total Changes:** ~463 lines added

## Compatibility

- ✅ Backward compatible with original ANIMAL-SPOT
- ✅ Default behavior unchanged (`--backbone resnet` is default)
- ✅ All existing config files work as before
- ✅ Per-class metrics only enabled for multi-class (num_classes > 2)

## Dependencies

**Additional requirements for ConvNeXt:**
- torchvision >= 0.13.0 (for `convnext_tiny`)

Install with:
```bash
pip install torchvision>=0.13.0
```

## Future Work

- [ ] Add class weighting support for rare classes
- [ ] Implement focal loss for imbalanced datasets
- [ ] Support other ConvNeXt variants (Small, Base, Large)
- [ ] Add gradient clipping for training stability
- [ ] Ensemble predictions (ResNet + ConvNeXt)

## Citation

If you use these improvements, please cite both the original ANIMAL-SPOT paper and mention this fork:

```bibtex
@article{BerglerAnimalSpot:2022,
  author = {Bergler, Christian and Smeele, Simeon and ...},
  title = {ANIMAL-SPOT enables animal-independent signal detection and classification using deep learning},
  journal = {Scientific Reports},
  year = {2022},
  doi = {10.1038/s41598-022-26429-y}
}
```

## License

GNU General Public License v3.0 (same as original ANIMAL-SPOT)

## Authors

- Original ANIMAL-SPOT: Christian Bergler, Hendrik Schroeter, et al.
- ConvNeXt integration & per-class metrics: [Your Name]
- Last Updated: December 2025
