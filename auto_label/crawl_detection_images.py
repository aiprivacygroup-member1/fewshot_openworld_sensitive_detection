"""
Web crawler for collecting raw/scene images that may contain detection targets.

Unlike retrieve_data.py (which downloads clean, centered images for classification),
this script downloads broader scene images where the target (face, license plate,
badge) may appear anywhere, at any scale — suitable for object detection data.

Usage:
  pythoncrawl_detection_images.py --num_per_class 500
  pythoncrawl_detection_images.py --num_per_class 500 --skip_download
      (only rebuild split file from existing images)
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import warnings
import re
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Scene-level queries — broader terms that produce images with targets in context
SCENE_QUERIES = {
    0: [  # face — people in various settings
        "people walking street",
        "crowd of people",
        "office workers",
        "people meeting room",
        "group photo",
        "people standing",
        "person working desk",
        "pedestrians city street",
        "people conference",
        "family portrait",
        "students classroom",
        "people selfie group",
        "team group photo",
        "people outdoor park",
        "commuters subway station",
    ],
    1: [  # license_plate — cars in context
        "cars parked street",
        "traffic road cars",
        "vehicles parking lot",
        "cars highway",
        "city traffic",
        "car front view",
        "car rear view",
        "parked vehicles",
        "street cars",
        "taxi cars city",
        "police car",
        "cars intersection",
        "trucks on road",
        "rush hour traffic",
        "car driving road",
    ],
    2: [  # badge — people wearing badges
        "conference attendees",
        "office workers with badge",
        "employee at work",
        "staff meeting",
        "conference badge lanyard",
        "people wearing name tags",
        "corporate event",
        "business meeting people",
        "security guard",
        "workplace employees",
        "company staff",
        "office team",
        "professional conference",
        "exhibition hall people",
        "trade show attendees",
    ],
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


# ==============================================================================
# Image Downloader
# ==============================================================================

def download_images_icrawler(class_id, query_list, save_dir, num_download):
    """Download scene images using icrawler Bing backend."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  icrawler not installed. Run: pip install icrawler")
        return 0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    per_query = max(10, min(80, num_download // len(query_list) + 10))

    for query in query_list:
        if downloaded >= num_download:
            break

        prefix = hashlib.md5(query.encode()).hexdigest()[:6]
        query_temp = save_dir / f'_tmp_{prefix}'
        query_temp.mkdir(exist_ok=True)

        try:
            crawler = BingImageCrawler(
                downloader_threads=4,
                storage={'root_dir': str(query_temp)}
            )
            crawler.crawl(
                keyword=query,
                max_num=min(per_query, num_download - downloaded),
                filters={'type': 'photo'},
                file_idx_offset='auto'
            )
        except Exception as e:
            print(f"  Query '{query[:40]}...' failed: {e}")
            shutil.rmtree(query_temp, ignore_errors=True)
            continue

        for img_file in query_temp.rglob('*'):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'):
                new_name = f"{prefix}_{downloaded:05d}{img_file.suffix}"
                try:
                    shutil.move(str(img_file), str(save_dir / new_name))
                    downloaded += 1
                except:
                    pass

        shutil.rmtree(query_temp, ignore_errors=True)
        print(f"  [{downloaded}/{num_download}] query='{query[:50]}...'")

    print(f"  icrawler total: {downloaded} images for class {class_id}")
    return downloaded


def download_images_duckduckgo(class_id, query_list, save_dir, num_download):
    """Fallback: DuckDuckGo image search via requests."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for query in query_list:
        if downloaded >= num_download:
            break

        try:
            resp = requests.post(
                'https://lite.duckduckgo.com/lite/',
                data={'q': query, 'iax': 'images', 'ia': 'images'},
                headers=HEADERS, timeout=15
            )
        except:
            continue

        img_urls = list(set(re.findall(
            r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)',
            resp.text, re.I
        )))

        for url in img_urls[:min(len(img_urls), num_download - downloaded)]:
            try:
                r = requests.get(url, headers=HEADERS, timeout=10, stream=True)
                if r.status_code == 200 and 'image' in r.headers.get('content-type', ''):
                    img = Image.open(BytesIO(r.content)).convert('RGB')
                    if img.size[0] >= 224 and img.size[1] >= 224:
                        img.save(save_dir / f"ddg_{downloaded:05d}.jpg", 'JPEG', quality=90)
                        downloaded += 1
            except:
                continue

    print(f"  DuckDuckGo total: {downloaded} images for class {class_id}")
    return downloaded


# ==============================================================================
# Image Validation
# ==============================================================================

def validate_images(image_dir, min_size=224):
    """Remove broken/corrupted/too-small images."""
    image_dir = Path(image_dir)
    removed = 0
    for img_file in image_dir.glob('*'):
        try:
            img = Image.open(img_file)
            img.verify()
            img = Image.open(img_file).convert('RGB')
            if img.size[0] < min_size or img.size[1] < min_size:
                img_file.unlink()
                removed += 1
        except:
            img_file.unlink()
            removed += 1
    if removed:
        print(f"  Removed {removed} invalid images")


# ==============================================================================
# Split File (unlabeled format)
# ==============================================================================

def create_unlabeled_split(raw_root, data_root):
    """Create split file for unlabeled/scene images.

    Format: path class_id -1   (-1 means unlabeled, no ground truth)
    """
    raw_root = Path(raw_root)
    data_root = Path(data_root)
    lines = []

    for class_id in range(3):
        class_dir = raw_root / str(class_id)
        if not class_dir.exists():
            continue
        for img_file in sorted(class_dir.glob('*')):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                rel_path = img_file.relative_to(data_root)
                # class_id is the *expected* dominant class for sorting,
                # but -1 marks it as unlabeled for auto-labeling
                lines.append(f"{rel_path} {class_id} -1")

    split_file = data_root / 'unlabeled_scenes.txt'
    with open(split_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved unlabeled split: {split_file} ({len(lines)} images)")
    return split_file


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download scene images for auto-labeling detection data')
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate')
    parser.add_argument('--num_per_class', type=int, default=500,
                        help='Number of images to download per class')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download, only validate and rebuild split')
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Override download directory')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    download_root = Path(args.download_dir) if args.download_dir else \
                    data_root / 'raw_scenes'

    total_downloaded = 0

    for class_id in range(3):
        class_dir = download_root / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)

        class_name = ['face', 'license_plate', 'badge'][class_id]
        print(f"\n{'='*50}")
        print(f"Class {class_id}: {class_name}")
        print(f"{'='*50}")

        if not args.skip_download:
            print(f"  Target: {args.num_per_class} scene images")
            count = download_images_icrawler(
                class_id, SCENE_QUERIES[class_id], class_dir, args.num_per_class
            )
            if count < args.num_per_class * 0.3:
                print(f"  icrawler had low yield, trying DuckDuckGo...")
                count2 = download_images_duckduckgo(
                    class_id, SCENE_QUERIES[class_id], class_dir,
                    args.num_per_class - count
                )
                count += count2
            total_downloaded += count
        else:
            existing = len(list(class_dir.glob('*')))
            print(f"  Skip download. Existing: {existing} images")
            total_downloaded += existing

        # Validate
        validate_images(class_dir)

    # Create split file
    create_unlabeled_split(download_root, data_root)

    print(f"\n{'='*50}")
    print(f"Done! Total images: {total_downloaded}")
    print(f"Images stored in: {download_root}")
    print(f"\nNext step: run auto-labeling")
    print(f"  pythonauto_label.py \\")
    print(f"    --model_path output/<run>/wsft_model.pth \\")
    print(f"    --input_split unlabeled_scenes.txt \\")
    print(f"    --output_format yolo")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
