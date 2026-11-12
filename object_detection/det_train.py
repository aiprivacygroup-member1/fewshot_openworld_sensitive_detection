"""
SWAT-Style Two-Stage Few-Shot Object Detection Training.

Adapts the SWAT paper methodology to object detection:

  Stage 1: Train on mixed data (auto-labeled + hand-labeled).
           Auto-labeled data = "retrieval augmentation" providing diversity.
           Hand-labeled data = "few-shot" providing accuracy.
           Strong augmentation (mosaic + mixup), oversampled hand-labeled samples.

  Stage 2: Fine-tune on hand-labeled data only.
           Lighter augmentation, frozen backbone layers.
           Analogous to SWAT's "classifier re-training" stage.

  Ensemble: Weight-averaging between Stage 1 and Stage 2 models (WiSE-FT spirit).

Requirements:
  pip install ultralytics

Usage:
  # Full pipeline
  pythondet_train.py --data output/detection/dataset/data.yaml

  # Custom settings
  pythondet_train.py --data data.yaml \
      --stage1_epochs 50 --stage2_epochs 30 \
      --oversample_hand 3 --model yolov8s
"""

import os
import sys
import yaml
import json
import shutil
import copy
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# ==============================================================================
# Dataset Splitting (Hand-labeled only subset for Stage 2)
# ==============================================================================

def create_hand_only_dataset(dataset_dir, output_dir):
    """Create a YOLO dataset config using only hand-labeled images.

    Filters out auto-labeled samples (files starting with 'auto_') from train,
    keeping only hand-labeled samples ('hand_'). Validation is unchanged.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_train_dir = output_dir / 'images' / 'train'
    lbl_train_dir = output_dir / 'labels' / 'train'

    for d in [img_train_dir, lbl_train_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy only hand-labeled files
    src_img = dataset_dir / 'images' / 'train'
    src_lbl = dataset_dir / 'labels' / 'train'

    hand_count = 0
    if src_img.exists():
        for img_file in src_img.glob('*'):
            if img_file.stem.startswith('hand_'):
                ext = img_file.suffix
                shutil.copy2(img_file, img_train_dir / img_file.name)

                # Copy corresponding label
                label_file = src_lbl / f"{img_file.stem}.txt"
                if label_file.exists():
                    # Strip source comment line to get clean YOLO labels
                    clean_labels = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                clean_labels.append(line)

                    with open(lbl_train_dir / f"{img_file.stem}.txt", 'w') as f:
                        f.write('\n'.join(clean_labels))

                hand_count += 1

    # Copy validation (unchanged, already hand-only)
    val_img = output_dir / 'images' / 'val'
    val_lbl = output_dir / 'labels' / 'val'
    src_val_img = dataset_dir / 'images' / 'val'
    src_val_lbl = dataset_dir / 'labels' / 'val'

    if src_val_img.exists():
        val_img.mkdir(parents=True, exist_ok=True)
        val_lbl.mkdir(parents=True, exist_ok=True)
        for f in src_val_img.glob('*'):
            shutil.copy2(f, val_img / f.name)
        for f in src_val_lbl.glob('*'):
            if f.suffix == '.txt':
                clean_labels = []
                with open(f, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            clean_labels.append(line)
                with open(val_lbl / f.name, 'w') as fh:
                    fh.write('\n'.join(clean_labels))

    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['face', 'license_plate', 'badge'],
    }
    yaml_path = output_dir / 'data_hand_only.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Hand-only dataset: {hand_count} train images -> {yaml_path}")
    return str(yaml_path)


def create_oversampled_dataset(dataset_dir, output_dir, oversample_factor=3):
    """Create dataset with oversampled hand-labeled data for Stage 1.

    Duplicates hand-labeled samples to give them more training weight,
    mimicking the SWAT approach of weighting few-shot data higher.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_train_dir = output_dir / 'images' / 'train'
    lbl_train_dir = output_dir / 'labels' / 'train'

    for d in [img_train_dir, lbl_train_dir]:
        d.mkdir(parents=True, exist_ok=True)

    src_img = dataset_dir / 'images' / 'train'
    src_lbl = dataset_dir / 'labels' / 'train'

    hand_count = 0
    auto_count = 0

    if src_img.exists():
        for img_file in sorted(src_img.glob('*')):
            label_file = src_lbl / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue

            is_hand = img_file.stem.startswith('hand_')
            is_auto = img_file.stem.startswith('auto_')

            # Read clean labels (strip source comments)
            clean_labels = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            clean_labels.append(line)

            if is_auto:
                # Copy auto data once
                ext = img_file.suffix
                shutil.copy2(img_file, img_train_dir / img_file.name)
                with open(lbl_train_dir / f"{img_file.stem}.txt", 'w') as f:
                    f.write('\n'.join(clean_labels))
                auto_count += 1

            elif is_hand:
                # Duplicate hand data oversample_factor times
                ext = img_file.suffix
                for dup in range(oversample_factor):
                    new_name = f"hand_{hand_count:06d}_dup{dup}"
                    dst_img = img_train_dir / f"{new_name}{ext}"
                    dst_lbl = lbl_train_dir / f"{new_name}.txt"

                    shutil.copy2(img_file, dst_img)
                    # Remap labels with same class_ids
                    with open(dst_lbl, 'w') as f:
                        f.write('\n'.join(clean_labels))

                hand_count += 1

    # Copy validation
    val_img = output_dir / 'images' / 'val'
    val_lbl = output_dir / 'labels' / 'val'
    src_val_img = dataset_dir / 'images' / 'val'
    src_val_lbl = dataset_dir / 'labels' / 'val'

    if src_val_img.exists():
        val_img.mkdir(parents=True, exist_ok=True)
        val_lbl.mkdir(parents=True, exist_ok=True)
        for f in src_val_img.glob('*'):
            shutil.copy2(f, val_img / f.name)
        for f in src_val_lbl.glob('*'):
            if f.suffix == '.txt':
                clean_labels = []
                with open(f, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            clean_labels.append(line)
                with open(val_lbl / f.name, 'w') as fh:
                    fh.write('\n'.join(clean_labels))

    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['face', 'license_plate', 'badge'],
    }
    yaml_path = output_dir / 'data_oversampled.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    effective_hand = hand_count * oversample_factor
    print(f"Oversampled dataset: {effective_hand} effective-hand + "
          f"{auto_count} auto = {effective_hand + auto_count} total -> {yaml_path}")
    return str(yaml_path)


# ==============================================================================
# Model Weight Averaging (WiSE-FT style for detection)
# ==============================================================================

def average_model_weights(model_path_a, model_path_b, output_path, alpha=0.5):
    """Average the weights of two YOLO models (WiSE-FT for detection).

    Args:
        alpha: weight for model_a, (1-alpha) for model_b
    """
    ckpt_a = torch.load(model_path_a, map_location='cpu')
    ckpt_b = torch.load(model_path_b, map_location='cpu')

    if 'model' in ckpt_a and 'model' in ckpt_b:
        state_a = ckpt_a['model'].state_dict() if hasattr(ckpt_a['model'], 'state_dict') else ckpt_a['model']
        state_b = ckpt_b['model'].state_dict() if hasattr(ckpt_b['model'], 'state_dict') else ckpt_b['model']

        if not isinstance(state_a, dict):
            state_a = state_a.float()
            state_b = state_b.float()

        if isinstance(state_a, dict) and isinstance(state_b, dict):
            avg_state = {}
            for key in state_a:
                if key in state_b:
                    avg_state[key] = alpha * state_a[key] + (1.0 - alpha) * state_b[key]
                else:
                    avg_state[key] = state_a[key]

            # Create new checkpoint
            new_ckpt = copy.deepcopy(ckpt_a)
            if hasattr(new_ckpt['model'], 'load_state_dict'):
                new_ckpt['model'].load_state_dict(avg_state)
            else:
                new_ckpt['model'] = avg_state

            torch.save(new_ckpt, output_path)
            return True

    print("Weight averaging failed: incompatible checkpoint formats")
    return False


# ==============================================================================
# Training Orchestrator
# ==============================================================================

def train_stage1(args, data_yaml):
    """Stage 1: Train on mixed (auto + hand) data with strong augmentation.

    Uses oversampled hand-labeled data for higher few-shot weight.
    """
    print(f"\n{'='*60}")
    print("STAGE 1: Mixed Training (auto-labeled + hand-labeled)")
    print(f"{'='*60}")

    # Create oversampled dataset
    dataset_dir = Path(data_yaml).parent
    oversampled_dir = Path(args.output_dir) / 'dataset_stage1'
    stage1_data = create_oversampled_dataset(
        dataset_dir, oversampled_dir, args.oversample_hand
    )

    # Initialize model
    model_name = args.model
    pretrained_weight = args.pretrained_weights if hasattr(args, 'pretrained_weights') and args.pretrained_weights else True
    model = YOLO(f"{model_name}.pt" if pretrained_weight else f"{model_name}.yaml")

    # Stage 1 training
    results = model.train(
        data=stage1_data,
        epochs=args.stage1_epochs,
        imgsz=args.imgsz,
        batch=args.stage1_batch,
        lr0=args.stage1_lr0,
        lrf=args.stage1_lrf,
        device=args.device,
        workers=args.workers,
        optimizer='AdamW',
        # Strong augmentation (mosaic + mixup)
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.1 if args.stage1_augment == 'strong' else 0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        # Validation
        val=True,
        save=True,
        save_period=args.save_period,
        project=args.output_dir,
        name='stage1_mixed',
        exist_ok=True,
        seed=args.seed,
        verbose=True,
    )

    # Get best model path
    stage1_best = os.path.join(args.output_dir, 'stage1_mixed', 'weights', 'best.pt')
    if not os.path.exists(stage1_best):
        # Ultralytics might put it in a different location
        alt_path = os.path.join(args.output_dir, 'stage1_mixed', 'weights', 'last.pt')
        if os.path.exists(alt_path):
            stage1_best = alt_path

    print(f"\nStage 1 complete. Best model: {stage1_best}")
    return stage1_best, results


def train_stage2(args, data_yaml, stage1_best):
    """Stage 2: Fine-tune on hand-labeled data only.

    Uses lighter augmentation, frozen backbone, lower learning rate.
    """
    print(f"\n{'='*60}")
    print("STAGE 2: Hand-labeled Fine-tuning")
    print(f"{'='*60}")

    # Create hand-only dataset
    dataset_dir = Path(data_yaml).parent
    hand_only_dir = Path(args.output_dir) / 'dataset_stage2'
    stage2_data = create_hand_only_dataset(dataset_dir, hand_only_dir)

    # Load Stage 1 model
    model = YOLO(stage1_best)

    # Stage 2 training
    results = model.train(
        data=stage2_data,
        epochs=args.stage2_epochs,
        imgsz=args.imgsz,
        batch=args.stage2_batch,
        lr0=args.stage2_lr0,
        lrf=args.stage2_lrf,
        device=args.device,
        workers=args.workers,
        optimizer='AdamW',
        # Lighter augmentation — no mosaic/mixup
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        # Freeze backbone for Stage 2
        freeze=10 if args.stage2_freeze_backbone else 0,
        # Validation
        val=True,
        save=True,
        save_period=args.save_period,
        project=args.output_dir,
        name='stage2_hand',
        exist_ok=True,
        seed=args.seed,
        verbose=True,
    )

    stage2_best = os.path.join(args.output_dir, 'stage2_hand', 'weights', 'best.pt')
    if not os.path.exists(stage2_best):
        stage2_best = os.path.join(args.output_dir, 'stage2_hand', 'weights', 'last.pt')

    print(f"\nStage 2 complete. Best model: {stage2_best}")
    return stage2_best, results


def ensemble_models(args, stage1_best, stage2_best, val_data_yaml):
    """WiSE-FT style weight averaging for detection models.

    Finds best alpha by sweeping [0.3, 0.5, 0.7] and validating.
    """
    print(f"\n{'='*60}")
    print("MODEL ENSEMBLING (WiSE-FT Style)")
    print(f"{'='*60}")

    best_map = -1
    best_alpha = 0.5
    ensemble_dir = Path(args.output_dir) / 'ensemble'

    for alpha in [0.3, 0.5, 0.7]:
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        ensemble_path = str(ensemble_dir / f'ensemble_alpha{alpha:.1f}.pt')

        success = average_model_weights(
            stage1_best, stage2_best, ensemble_path, alpha=alpha
        )

        if success:
            model = YOLO(ensemble_path)
            metrics = model.val(data=val_data_yaml, device=args.device,
                                imgsz=args.imgsz, verbose=False)
            mAP50 = metrics.box.map50 if hasattr(metrics.box, 'map50') else 0.0
            print(f"  Alpha={alpha:.1f}: mAP@50={mAP50:.4f}")

            if mAP50 > best_map:
                best_map = mAP50
                best_alpha = alpha
        else:
            print(f"  Alpha={alpha:.1f}: averaging failed")

    # Save best ensemble
    best_path = str(ensemble_dir / 'best_ensemble.pt')
    best_ensemble_path = str(ensemble_dir / f'ensemble_alpha{best_alpha:.1f}.pt')
    if os.path.exists(best_ensemble_path):
        shutil.copy2(best_ensemble_path, best_path)
        print(f"\nBest ensemble: alpha={best_alpha:.1f}, mAP@50={best_map:.4f}")
        print(f"Saved: {best_path}")

    return best_path, best_alpha, best_map


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SWAT-Style Few-Shot Object Detection Training')

    # ---- Data ----
    parser.add_argument('--data', type=str,
                        default='output/detection/dataset/data.yaml',
                        help='Path to data.yaml from det_dataset.py')
    parser.add_argument('--output_dir', type=str,
                        default='output/detection')

    # ---- Model ----
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'])
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Override pretrained weights path')
    parser.add_argument('--imgsz', type=int, default=640)

    # ---- Stage 1 ----
    parser.add_argument('--stage1_epochs', type=int, default=50)
    parser.add_argument('--stage1_batch', type=int, default=16)
    parser.add_argument('--stage1_lr0', type=float, default=0.001)
    parser.add_argument('--stage1_lrf', type=float, default=0.01)
    parser.add_argument('--stage1_augment', type=str, default='strong',
                        choices=['strong', 'light'])
    parser.add_argument('--oversample_hand', type=int, default=3,
                        help='Duplicate hand-labeled samples N times in Stage 1')

    # ---- Stage 2 ----
    parser.add_argument('--stage2_epochs', type=int, default=30)
    parser.add_argument('--stage2_batch', type=int, default=8)
    parser.add_argument('--stage2_lr0', type=float, default=0.0005)
    parser.add_argument('--stage2_lrf', type=float, default=0.01)
    parser.add_argument('--stage2_augment', type=str, default='light')
    parser.add_argument('--stage2_freeze_backbone', type=lambda x: x.lower() == 'true',
                        default=True)

    # ---- General ----
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--skip_ensemble', action='store_true')
    parser.add_argument('--skip_stage2', action='store_true')

    args = parser.parse_args()

    # ---- Check ultralytics ----
    if not ULTRALYTICS_AVAILABLE:
        print("ERROR: ultralytics not installed.")
        print("  Run: pip install ultralytics")
        sys.exit(1)

    # ---- Check dataset ----
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset YAML not found: {args.data}")
        print("  Run: pythondet_dataset.py first")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Save config ----
    config_path = os.path.join(args.output_dir, 'train_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    start_time = datetime.now()

    # =====================================================================
    # Stage 1: Mixed Training
    # =====================================================================
    stage1_best, stage1_results = train_stage1(args, args.data)

    # =====================================================================
    # Stage 2: Hand-only Fine-tuning
    # =====================================================================
    if not args.skip_stage2:
        stage2_best, stage2_results = train_stage2(args, args.data, stage1_best)
    else:
        stage2_best = stage1_best

    # =====================================================================
    # Ensemble
    # =====================================================================
    ensemble_path = stage2_best
    best_alpha = 1.0
    ensemble_map = 0.0

    if not args.skip_ensemble and not args.skip_stage2:
        # Create hand-only val dataset for ensemble selection
        dataset_dir = Path(args.data).parent
        val_yaml = args.data  # Use same val set

        ensemble_path, best_alpha, ensemble_map = ensemble_models(
            args, stage1_best, stage2_best, val_yaml
        )

    # =====================================================================
    # Final Summary
    # =====================================================================
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Time elapsed:     {elapsed}")
    print(f"  Stage 1 model:    {stage1_best}")
    if not args.skip_stage2:
        print(f"  Stage 2 model:    {stage2_best}")
    if not args.skip_ensemble and not args.skip_stage2:
        print(f"  Ensemble (α={best_alpha:.1f}): {ensemble_path}")

    print(f"\n  All outputs saved to: {args.output_dir}")
    print(f"\n  Next step: evaluate the model")
    print(f"    pythondet_test.py \\")
    print(f"      --model {ensemble_path} \\")
    print(f"      --data {args.data}")


if __name__ == '__main__':
    main()
