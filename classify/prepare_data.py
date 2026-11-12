"""
Data preparation script for 3-class few-shot classification.
Classes: face, license_plate, badge

Expected input directory structure:
  data/face_badge_plate/
    face/         # all face images
    license_plate/  # all license plate images
    badge/        # all employee badge images

Output:
  - Creates train/val/test split files (fewshotN_seedX.txt, test.txt)
  - Creates metrics JSON for prompt construction
  - Creates class name mapping
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Class definitions
CLASS_NAMES = {
    0: {
        "common_name": "human face",
        "most_common_name": "human face",
        "species": "human face",
        "description": "a human face"
    },
    1: {
        "common_name": "license plate",
        "most_common_name": "license plate",
        "species": "license plate",
        "description": "a vehicle license plate"
    },
    2: {
        "common_name": "employee badge",
        "most_common_name": "employee badge",
        "species": "employee badge",
        "description": "an employee ID badge or work badge"
    }
}

CLASS_FOLDERS = ['face', 'license_plate', 'badge']


def create_dataset_structure(data_root, shots, seed, val_ratio=0.2):
    """
    Create the dataset splits from class folders.

    Args:
        data_root: root directory containing class subfolders
        shots: number of few-shot examples per class
        seed: random seed for reproducibility
        val_ratio: ratio of remaining data to use as validation
    """
    data_root = Path(data_root)
    random.seed(seed)

    # Collect all images per class
    class_images = {}
    for class_id, folder_name in enumerate(CLASS_FOLDERS):
        folder_path = data_root / folder_name
        if not folder_path.exists():
            print(f"WARNING: Folder {folder_path} does not exist. Creating it...")
            folder_path.mkdir(parents=True, exist_ok=True)

        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(folder_path.glob(ext)))

        if len(images) == 0:
            print(f"WARNING: No images found in {folder_path}. "
                  f"Please add images to this folder before training.")

        class_images[class_id] = images
        print(f"Class '{folder_name}' (id={class_id}): {len(images)} images")

    # Check minimum shots requirement
    for class_id, images in class_images.items():
        if len(images) < shots + 2:
            raise ValueError(
                f"Class '{CLASS_FOLDERS[class_id]}' has only {len(images)} images, "
                f"but need at least {shots + 2} (shots={shots} + 1 val + 1 test)."
            )

    # Split each class into few-shot train, val, and test
    fewshot_entries = []
    val_entries = []
    test_entries = []

    for class_id, images in class_images.items():
        random.shuffle(images)
        folder_name = CLASS_FOLDERS[class_id]

        # Pick shots for few-shot training
        fewshot_imgs = images[:shots]
        remaining = images[shots:]

        # Split remaining into val and test
        n_val = max(1, int(len(remaining) * val_ratio))
        val_imgs = remaining[:n_val]
        test_imgs = remaining[n_val:]

        if len(test_imgs) == 0:
            val_imgs, test_imgs = val_imgs[:-1], val_imgs[-1:]

        for img_path in fewshot_imgs:
            rel_path = f"{folder_name}/{img_path.name}"
            fewshot_entries.append(f"{rel_path} {class_id} 1")

        for img_path in val_imgs:
            rel_path = f"{folder_name}/{img_path.name}"
            val_entries.append(f"{rel_path} {class_id} 1")

        for img_path in test_imgs:
            rel_path = f"{folder_name}/{img_path.name}"
            test_entries.append(f"{rel_path} {class_id} 1")

        print(f"Class '{folder_name}': train={len(fewshot_imgs)}, "
              f"val={len(val_imgs)}, test={len(test_imgs)}")

    # Save splits
    fewshot_file = data_root / f"fewshot{shots}_seed{seed}.txt"
    test_file = data_root / "test.txt"

    random.shuffle(fewshot_entries)
    with open(fewshot_file, 'w') as f:
        f.write('\n'.join(fewshot_entries))
    print(f"Saved few-shot split: {fewshot_file} ({len(fewshot_entries)} samples)")

    # For validation, also use the few-shot file (same as paper)
    # The paper uses the few-shot data for validation

    random.shuffle(test_entries)
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_entries))
    print(f"Saved test split: {test_file} ({len(test_entries)} samples)")

    return fewshot_file, test_file


def create_metrics_json(data_root):
    """Create metrics JSON for prompt template construction."""
    metrics = {}
    for class_id in range(3):
        metrics[str(class_id)] = {
            "common_name": CLASS_NAMES[class_id]["common_name"],
            "most_common_name": CLASS_NAMES[class_id]["most_common_name"],
            "species": CLASS_NAMES[class_id]["species"],
            "class_id": class_id
        }

    metrics_file = Path(data_root) / "face_badge_plate_metrics-LAION400M.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {metrics_file}")

    return metrics_file


def print_usage_instructions(data_root):
    """Print instructions for how to organize data."""
    print("""
============================================================
DATA PREPARATION INSTRUCTIONS
============================================================

Please organize your images in the following structure:

  data/face_badge_plate/
    face/              <- Put all face photos here
      person1.jpg
      person2.jpg
      ...
    license_plate/     <- Put all license plate photos here
      plate1.jpg
      plate2.jpg
      ...
    badge/             <- Put all employee badge photos here
      badge1.jpg
      badge2.jpg
      ...

Supported formats: .jpg, .jpeg, .png, .bmp

Each class should have at least shot+2 images.
For 5-shot training, you need at least 7 images per class.

Then run:
  pythonprepare_data.py --shots 5 --seed 1
============================================================
""")


def main():
    parser = argparse.ArgumentParser(description='Prepare custom 3-class dataset')
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate',
                        help='Root directory containing class subfolders')
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of few-shot examples per class')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for data splitting')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of non-fewshot data to use as validation')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        for folder in CLASS_FOLDERS:
            (data_root / folder).mkdir(exist_ok=True)
        print_usage_instructions(args.data_root)
        return

    # Check if class folders have images
    has_images = False
    for folder in CLASS_FOLDERS:
        folder_path = data_root / folder
        if folder_path.exists():
            imgs = list(folder_path.glob('*.[jp][pn][g]*')) + \
                   list(folder_path.glob('*.JPG')) + \
                   list(folder_path.glob('*.PNG')) + \
                   list(folder_path.glob('*.bmp'))
            if len(imgs) > 0:
                has_images = True
                break

    if not has_images:
        for folder in CLASS_FOLDERS:
            (data_root / folder).mkdir(parents=True, exist_ok=True)
        print_usage_instructions(args.data_root)
        return

    print(f"\nPreparing dataset from: {data_root}")
    print(f"Shots: {args.shots}, Seed: {args.seed}\n")

    # Create splits
    create_dataset_structure(data_root, args.shots, args.seed, args.val_ratio)

    # Create metrics JSON
    create_metrics_json(data_root)

    # Create prompts directory structure
    prompts_dir = data_root / 'prompts'
    prompts_dir.mkdir(exist_ok=True)

    pre_extracted_dir = data_root / 'pre_extracted'
    pre_extracted_dir.mkdir(exist_ok=True)

    print(f"\nData preparation complete!")
    print(f"Next step: run training with")
    print(f"  pythontrain.py --shots {args.shots} --seed {args.seed}")


if __name__ == '__main__':
    main()
