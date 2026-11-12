"""
Detection dataset preparation for SWAT-style two-stage training.

Merges hand-labeled (few-shot) data with auto-labeled (retrieval-style) data
into a unified YOLO training structure, with source tracking for weighted loss.

Output structure:
  output/detection/dataset/
    images/
      train/    # symlinks/copies: hand-labeled train + auto-labeled
      val/      # symlinks/copies: hand-labeled val
    labels/
      train/    # YOLO format .txt labels
      val/
    data.yaml   # dataset config for YOLO training

Usage:
  pythondet_dataset.py
  pythondet_dataset.py --hand_dir data/my_hand_labels --auto_dir output/labels
"""

import os
import sys
import yaml
import shutil
import random
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')


def parse_yolo_label(label_path):
    """Parse a YOLO-format label file.

    Returns: list of (class_id, cx, cy, w, h) normalized
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                # Validate: values should be in [0, 1]
                if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    labels.append((class_id, cx, cy, w, h))
            except (ValueError, IndexError):
                continue

    return labels


def validate_yolo_label(labels, img_w=None, img_h=None):
    """Check if labels are valid (within image bounds, reasonable sizes)."""
    if not labels:
        return False
    for class_id, cx, cy, w, h in labels:
        if class_id not in (0, 1, 2):
            return False
        # Bounding box should be within [0,1]
        if cx - w/2 < -0.01 or cy - h/2 < -0.01:
            return False
        if cx + w/2 > 1.01 or cy + h/2 > 1.01:
            return False
        # Box shouldn't be too small (>1% of image)
        if w < 0.005 or h < 0.005:
            return False
    return True


def discover_hand_labeled_data(hand_dir):
    """Discover hand-labeled images and their YOLO labels.

    Expected structure:
      hand_dir/
        images/          # .jpg/.png images
        labels/          # .txt labels with same stem name

    Returns: list of (image_path, label_path, source='hand')
    """
    hand_dir = Path(hand_dir)

    # Try different common directory structures
    candidates = [
        hand_dir,                                 # images and labels in same dir
        hand_dir / 'images',                      # images/ + labels/
        hand_dir / 'train' / 'images',            # train/images/ + train/labels/
    ]

    pairs = []
    for img_dir_candidate in candidates:
        if not img_dir_candidate.exists():
            continue

        # Find corresponding label dir
        if img_dir_candidate.name == 'images':
            label_dir = img_dir_candidate.parent / 'labels'
        elif img_dir_candidate == hand_dir:
            label_dir = hand_dir
        else:
            label_dir = img_dir_candidate.parent.parent / 'labels'

        if not label_dir.exists():
            label_dir = img_dir_candidate  # Same dir fallback

        # Match images with labels
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG'):
            for img_path in img_dir_candidate.glob(ext):
                stem = img_path.stem
                label_path = label_dir / f"{stem}.txt"

                labels = parse_yolo_label(str(label_path))
                if validate_yolo_label(labels):
                    pairs.append((str(img_path), str(label_path), 'hand'))

    return pairs


def discover_auto_labeled_data(auto_label_dir, auto_image_dir):
    """Discover auto-labeled images and their YOLO labels.

    Expected structure:
      auto_label_dir/          # YOLO .txt files (from auto_label.py)
        img001.txt
        img002.txt
      auto_image_dir/          # Source images
        0/  img001.jpg
        1/  img002.jpg
        ...

    Returns: list of (image_path, label_path, source='auto')
    """
    auto_label_dir = Path(auto_label_dir)
    auto_image_dir = Path(auto_image_dir)

    if not auto_label_dir.exists():
        print(f"Auto label dir not found: {auto_label_dir}")
        return []

    pairs = []

    # Build image lookup: stem -> image_path
    image_lookup = {}
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        for img_path in auto_image_dir.rglob(ext):
            image_lookup[img_path.stem] = str(img_path)

    for label_file in auto_label_dir.glob('*.txt'):
        stem = label_file.stem
        if stem in image_lookup:
            labels = parse_yolo_label(str(label_file))
            if labels and validate_yolo_label(labels):
                pairs.append((image_lookup[stem], str(label_file), 'auto'))

    return pairs


def build_yolo_dataset(hand_pairs, auto_pairs, output_dir, val_ratio=0.2, seed=42):
    """Build a unified YOLO dataset structure.

    Args:
        hand_pairs: list of (img_path, label_path, 'hand')
        auto_pairs: list of (img_path, label_path, 'auto')
        output_dir: output root directory
        val_ratio: fraction of hand-labeled data for validation
        seed: random seed

    Returns:
        data_yaml_path: path to the generated data.yaml
        stats: dict with dataset statistics
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    img_train_dir = output_dir / 'images' / 'train'
    img_val_dir = output_dir / 'images' / 'val'
    lbl_train_dir = output_dir / 'labels' / 'train'
    lbl_val_dir = output_dir / 'labels' / 'val'

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split hand-labeled data into train/val
    random.shuffle(hand_pairs)
    n_val = max(1, int(len(hand_pairs) * val_ratio))
    hand_train = hand_pairs[n_val:]
    hand_val = hand_pairs[:n_val]

    stats = {
        'hand_train': len(hand_train),
        'hand_val': len(hand_val),
        'auto_train': len(auto_pairs),
        'total_train': len(hand_train) + len(auto_pairs),
        'total_val': len(hand_val),
    }

    # ---- Copy/symlink to train ----
    train_pairs = hand_train + auto_pairs
    _copy_files(train_pairs, img_train_dir, lbl_train_dir, prefix='')

    # ---- Copy/symlink to val (only hand-labeled) ----
    _copy_files(hand_val, img_val_dir, lbl_val_dir, prefix='')

    # ---- Create data.yaml ----
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['face', 'license_plate', 'badge'],
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return str(yaml_path), stats


def _copy_files(pairs, img_dir, lbl_dir, prefix=''):
    """Copy image and label files to target directories.

    Renames files to avoid collisions.
    """
    for idx, (img_path, label_path, source) in enumerate(pairs):
        ext = Path(img_path).suffix
        new_name = f"{source}_{idx:06d}"

        # Copy image
        dst_img = img_dir / f"{new_name}{ext}"
        if not dst_img.exists():
            try:
                shutil.copy2(img_path, dst_img)
            except OSError:
                try:
                    shutil.copyfile(img_path, dst_img)
                except:
                    continue

        # Copy and rename label
        dst_label = lbl_dir / f"{new_name}.txt"
        if not dst_label.exists():
            try:
                shutil.copy2(label_path, dst_label)
            except OSError:
                try:
                    shutil.copyfile(label_path, dst_label)
                except:
                    continue

        # Track source in label file (add a comment line for reference)
        # This enables weighted loss during Stage 1 training
        # We append source tag: #source=auto  or  #source=hand
        with open(dst_label, 'a') as f:
            f.write(f"\n#source={source}")


def print_stats(stats):
    """Print dataset statistics."""
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"  Hand-labeled train:  {stats['hand_train']}")
    print(f"  Auto-labeled train:  {stats['auto_train']}")
    print(f"  Total train:         {stats['total_train']}")
    print(f"  Validation:          {stats['total_val']} (hand-labeled only)")
    print(f"  Few-shot multiplier: {stats['auto_train'] / max(1, stats['hand_train']):.1f}x "
          f"(auto/hand ratio)")

    if stats['hand_train'] < 5:
        print(f"\n  WARNING: Only {stats['hand_train']} hand-labeled training images!")
        print(f"  Few-shot detection works best with at least 10-20 hand-labeled images.")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare detection dataset for SWAT training')
    parser.add_argument('--hand_dir', type=str,
                        default='data/det_hand_labeled',
                        help='Directory with hand-labeled images/ and labels/')
    parser.add_argument('--auto_label_dir', type=str,
                        default='output/labels',
                        help='Directory with auto-generated YOLO labels')
    parser.add_argument('--auto_image_dir', type=str,
                        default='data/face_badge_plate/raw_scenes',
                        help='Directory with raw scene images')
    parser.add_argument('--output_dir', type=str,
                        default='output/detection/dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ---- Discover data ----
    print("\nDiscovering hand-labeled data...")
    hand_pairs = discover_hand_labeled_data(args.hand_dir)
    print(f"  Found {len(hand_pairs)} hand-labeled image/label pairs")

    if len(hand_pairs) == 0:
        print(f"\nERROR: No hand-labeled data found in '{args.hand_dir}'.")
        print("\nExpected structure:")
        print(f"  {args.hand_dir}/")
        print(f"    images/          # .jpg/.png images")
        print(f"      face001.jpg")
        print(f"      plate001.jpg")
        print(f"    labels/          # YOLO .txt labels (same stem name)")
        print(f"      face001.txt    # contains: class_id cx cy w h")
        print(f"      plate001.txt")
        return

    print("\nDiscovering auto-labeled data...")
    auto_pairs = discover_auto_labeled_data(args.auto_label_dir, args.auto_image_dir)
    print(f"  Found {len(auto_pairs)} auto-labeled image/label pairs")

    # ---- Build YOLO dataset ----
    print("\nBuilding unified YOLO dataset...")
    yaml_path, stats = build_yolo_dataset(
        hand_pairs, auto_pairs, args.output_dir, args.val_ratio, args.seed
    )

    print_stats(stats)
    print(f"\nDataset YAML: {yaml_path}")
    print(f"\nNext step:")
    print(f"  pythondet_train.py \\")
    print(f"    --data {yaml_path} \\")
    print(f"    --hand_train_count {stats['hand_train']}")


if __name__ == '__main__':
    main()
