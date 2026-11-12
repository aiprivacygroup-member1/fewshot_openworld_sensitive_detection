"""
Generate synthetic test data for pipeline verification.
Creates random images in the expected directory structure so you can
verify the entire pipeline runs end-to-end before using real data.

Usage:
  pythondemo_data.py   # Creates 30 synthetic images per class
  pythondemo_data.py --images_per_class 5  # Minimal dataset
"""

import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import colorsys


def create_synthetic_image(class_name, index, size=(224, 224)):
    """Create a synthetic image with class label text."""
    # Generate a random but deterministic background color based on class
    seed = hash(class_name) + index
    random.seed(seed)

    hue_map = {'face': 0.07, 'license_plate': 0.55, 'badge': 0.0}
    hue = hue_map.get(class_name, 0.5)

    # Vary brightness and saturation for variety
    saturation = 0.6 + random.random() * 0.4
    brightness = 0.7 + random.random() * 0.3

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    bg_color = (int(r * 255), int(g * 255), int(b * 255))

    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    # Draw some distinguishing features per class
    if class_name == 'face':
        # Draw a simple face-like circle
        cx, cy = size[0] // 2, size[1] // 2
        r = size[0] // 4
        draw.ellipse([cx - r, cy - r + random.randint(-10, 10),
                       cx + r, cy + r + random.randint(-10, 10)],
                      outline='white', width=3)
        # Eyes
        draw.ellipse([cx - r // 2, cy - r // 2, cx - r // 4, cy - r // 4], fill='white')
        draw.ellipse([cx + r // 4, cy - r // 2, cx + r // 2, cy - r // 4], fill='white')
        # Mouth
        draw.arc([cx - r // 2, cy, cx + r // 2, cy + r // 2], start=0, end=180,
                 fill='white', width=2)

    elif class_name == 'license_plate':
        # Draw a plate-like rectangle
        margin = 20 + random.randint(0, 15)
        draw.rectangle([margin, size[1] // 2 - 30, size[0] - margin, size[1] // 2 + 30],
                        outline='white', width=3)
        # Random text-like marks
        for x in range(margin + 15, size[0] - margin, 30):
            draw.text((x, size[1] // 2 - 15),
                      random.choice(['A', 'B', 'C', '0', '1', '2', '3']),
                      fill='white')
        # Blue top strip
        draw.rectangle([margin, size[1] // 2 - 30, size[0] - margin, size[1] // 2 - 20],
                        fill=(0, 100, 200))

    elif class_name == 'badge':
        # Draw a badge/card shape
        margin = 30
        draw.rounded_rectangle([margin, margin, size[0] - margin, size[1] - margin],
                                radius=15, outline='white', width=3)
        # ID photo area
        draw.rectangle([margin + 10, margin + 10, margin + 60, margin + 70],
                        outline='white', width=2)
        # Text lines
        for i, y in enumerate([margin + 20, margin + 45, margin + 85, margin + 105]):
            w = random.randint(50, 100)
            x0 = margin + 80 + random.randint(0, 10)
            draw.rectangle([x0, y, x0 + w, y + 8], fill='white')

    # Add text label
    try:
        font = ImageFont.load_default()
        text_color = 'white'
    except:
        font = None
        text_color = 'white'

    draw.text((5, size[1] - 20), f"{class_name}_{index}", fill=text_color)

    return img


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic demo data')
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate')
    parser.add_argument('--images_per_class', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    classes = ['face', 'license_plate', 'badge']

    for class_name in classes:
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing synthetic images
        for f in class_dir.glob('synthetic_*.png'):
            f.unlink()

        for i in range(args.images_per_class):
            img = create_synthetic_image(class_name, i)
            img_path = class_dir / f'synthetic_{i:04d}.png'
            img.save(img_path)

        print(f"Created {args.images_per_class} synthetic images in: {class_dir}")

    print(f"\nDone! Created {args.images_per_class * 3} synthetic images.")
    print(f"\nNow run data preparation:")
    print(f"  pythonprepare_data.py --shots 5 --seed 1")
    print(f"\nThen train:")
    print(f"  pythontrain.py --shots 5 --seed 1 --epochs 5")


if __name__ == '__main__':
    main()
