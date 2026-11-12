"""
Web image retrieval script for retrieval-augmented few-shot training.

Downloads images from web search engines for each class, filters them by
CLIP similarity to the class text prompt, and organizes them for the
SWAT "finetune-mixed" training pipeline.

Output structure:
  data/face_badge_plate/retrieved/
    0/          # face images from web
    1/          # license_plate images from web
    2/          # badge images from web
    retrieved.txt   # split file (format: "0/img001.jpg 0 0")

Usage:
  # Download 200 images per class, keep top 100 by CLIP similarity
  pythonretrieve_data.py --num_download 200 --num_keep 100

  # Use existing images in retrieved/ folder, just filter and rebuild split
  pythonretrieve_data.py --skip_download --num_keep 100

Dependencies:
  pip install icrawler requests Pillow
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import warnings
import time
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

NUM_CLASSES = 3
CLASS_IDS = [0, 1, 2]

# Search queries per class — diverse terms for broader coverage
SEARCH_QUERIES = {
    0: [  # face
        "human face portrait photo",
        "person face close up",
        "face photograph front view",
        "portrait photo human",
        "person headshot",
        "people face photo",
        "human face selfie",
    ],
    1: [  # license_plate
        "car license plate",
        "vehicle number plate",
        "license plate close up",
        "car plate number",
        "auto license plate photo",
        "truck license plate",
        "vehicle registration plate",
    ],
    2: [  # badge
        "employee ID badge",
        "worker badge photo",
        "staff identification card",
        "company ID card badge",
        "work badge with photo",
        "employee name badge",
        "office ID card",
    ],
}

# Prompt templates for CLIP similarity filtering
PROMPT_TEMPLATES = [
    'a photo of a {}.',
    'a close-up photo of a {}.',
    'a clear photo of a {}.',
    'a photo of the {}.',
]

CLASS_DESCRIPTIONS = {
    0: 'human face',
    1: 'license plate',
    2: 'employee badge',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


# ==============================================================================
# Image Downloader (multiple backends)
# ==============================================================================

def download_images_icrawler(class_id, query_list, save_dir, num_download):
    """Download images using icrawler (Bing backend)."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  icrawler not installed. Try: pip install icrawler")
        return 0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp directory first
    temp_dir = save_dir / '_temp_download'
    temp_dir.mkdir(exist_ok=True)

    downloaded = 0
    for query in query_list:
        if downloaded >= num_download:
            break

        # icrawler will name files as 000001.jpg, 000002.jpg etc.
        # We need to rename them to avoid collisions across queries
        prefix = hashlib.md5(query.encode()).hexdigest()[:6]
        query_temp = temp_dir / prefix
        query_temp.mkdir(exist_ok=True)

        try:
            crawler = BingImageCrawler(
                downloader_threads=4,
                storage={'root_dir': str(query_temp)}
            )
            crawler.crawl(
                keyword=query,
                max_num=min(num_download, 50),  # 50 per query max
                filters={'type': 'photo'},
                file_idx_offset='auto'
            )
        except Exception as e:
            print(f"  Query '{query}' failed: {e}")
            continue

        # Move images from temp to main dir with unique names
        for img_file in query_temp.rglob('*'):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'):
                new_name = f"{prefix}_{downloaded:05d}{img_file.suffix}"
                try:
                    shutil.move(str(img_file), str(save_dir / new_name))
                    downloaded += 1
                except:
                    pass

        # Cleanup query temp
        shutil.rmtree(query_temp, ignore_errors=True)

    # Cleanup overall temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"  icrawler downloaded {downloaded} images for class {class_id}")
    return downloaded


def download_images_requests(class_id, query_list, save_dir, num_download):
    """Fallback: download from DuckDuckGo image search via requests."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for query in query_list:
        if downloaded >= num_download:
            break

        # DuckDuckGo image search (lite version, no API key needed)
        url = "https://lite.duckduckgo.com/lite/"
        try:
            resp = requests.post(url, data={'q': query, 'iax': 'images', 'ia': 'images'},
                                headers=HEADERS, timeout=15)
        except:
            continue

        # Extract image URLs (simple approach)
        import re
        img_urls = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)', resp.text, re.I)

        for img_url in img_urls[:min(len(img_urls), num_download - downloaded)]:
            try:
                img_resp = requests.get(img_url, headers=HEADERS, timeout=10,
                                        stream=True)
                if img_resp.status_code == 200 and 'image' in img_resp.headers.get('content-type', ''):
                    img = Image.open(BytesIO(img_resp.content))
                    img = img.convert('RGB')
                    if img.size[0] >= 100 and img.size[1] >= 100:  # Min size filter
                        img.save(save_dir / f"req_{downloaded:05d}.jpg", 'JPEG', quality=85)
                        downloaded += 1
            except:
                continue

    print(f"  requests downloaded {downloaded} images for class {class_id}")
    return downloaded


def download_images_for_class(class_id, save_dir, num_download):
    """Download images from web for a class, trying multiple backends."""
    queries = SEARCH_QUERIES[class_id]

    # Try icrawler first, fallback to requests-based approach
    count = download_images_icrawler(class_id, queries, save_dir, num_download)

    if count < 10:
        print(f"  icrawler got few images, trying requests fallback...")
        count2 = download_images_requests(class_id, queries, save_dir,
                                          num_download - count)
        count += count2

    return count


# ==============================================================================
# CLIP Similarity Filtering
# ==============================================================================

def load_clip_for_filtering(device='cuda'):
    """Load a lightweight CLIP model for filtering."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32-quickgelu', pretrained='laion400m_e32'
        )
    except:
        try:
            import clip
            model, preprocess = clip.load('ViT-B/32', device)
            return model, preprocess, clip.tokenize
        except:
            raise RuntimeError("Need open_clip or clip installed for filtering.")

    model.float().to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_filter_images(image_dir, class_id, model, preprocess, tokenizer,
                       top_k, device='cuda'):
    """
    Filter images by CLIP similarity to class text prompts.
    Keeps only the top_k most relevant images.
    """
    image_dir = Path(image_dir)
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(list(image_dir.glob(ext)))

    if len(image_files) <= top_k:
        return image_files  # Keep all

    # Build text prompt embeddings
    desc = CLASS_DESCRIPTIONS[class_id]
    prompts = [t.format(desc) for t in PROMPT_TEMPLATES]
    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feat = text_feats.mean(dim=0, keepdim=True)  # Mean prompt embedding

    # Score all images
    scores = []
    for img_file in image_files:
        try:
            img = Image.open(img_file).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item()
            scores.append((img_file, sim))
        except:
            continue

    # Sort by similarity descending, keep top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    keep = [s[0] for s in scores[:top_k]]
    discard = [s[0] for s in scores[top_k:]]

    # Delete discarded images
    for f in discard:
        try:
            f.unlink()
        except:
            pass

    print(f"  CLIP filter: kept {len(keep)}/{len(scores)} images "
          f"(similarity range: {scores[top_k-1][1]:.3f} ~ {scores[0][1]:.3f})")

    return keep


# ==============================================================================
# Validation: Remove near-duplicates and broken images
# ==============================================================================

def validate_images(image_dir):
    """Remove broken/corrupted images."""
    image_dir = Path(image_dir)
    removed = 0
    for img_file in image_dir.glob('*'):
        try:
            img = Image.open(img_file)
            img.verify()
            # Re-open after verify
            img = Image.open(img_file)
            img.convert('RGB')
        except:
            img_file.unlink()
            removed += 1
    if removed:
        print(f"  Removed {removed} broken images")


def deduplicate_images(image_dir, threshold=0.98, device='cuda'):
    """Remove near-duplicate images using CLIP embedding similarity."""
    image_dir = Path(image_dir)
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(list(image_dir.glob(ext)))

    if len(image_files) < 2:
        return

    model, preprocess, _ = load_clip_for_filtering(device)

    # Extract features for all images
    feats = []
    valid_files = []
    for img_file in image_files:
        try:
            img = Image.open(img_file).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat)
            valid_files.append(img_file)
        except:
            img_file.unlink()
            continue

    if len(feats) < 2:
        return

    feats = torch.cat(feats, dim=0)
    sim_matrix = feats @ feats.T

    # Mark duplicates (excluding self-similarity)
    removed = set()
    for i in range(len(valid_files)):
        if i in removed:
            continue
        for j in range(i + 1, len(valid_files)):
            if j in removed:
                continue
            if sim_matrix[i, j] > threshold:
                removed.add(j)  # Remove the later one

    for idx in removed:
        try:
            valid_files[idx].unlink()
        except:
            pass

    if removed:
        print(f"  Removed {len(removed)} near-duplicate images")


# ==============================================================================
# Create split file for retrieval data
# ==============================================================================

def create_retrieval_split(retrieved_root, data_root):
    """Create retrieved.txt split file in the format expected by SWAT."""
    retrieved_root = Path(retrieved_root)
    lines = []

    for class_id in CLASS_IDS:
        class_dir = retrieved_root / str(class_id)
        if not class_dir.exists():
            continue
        for img_file in sorted(class_dir.glob('*')):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                rel_path = img_file.relative_to(data_root)
                # 0 at the end means "retrieved data" (vs 1 = few-shot)
                lines.append(f"{rel_path} {class_id} 0")

    split_file = data_root / 'retrieved.txt'
    with open(split_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved retrieval split: {split_file} ({len(lines)} images)")
    return split_file


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare retrieval-augmented training data')
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate')
    parser.add_argument('--num_download', type=int, default=200,
                        help='Target number of images to download per class')
    parser.add_argument('--num_keep', type=int, default=100,
                        help='Number of best images to keep per class after CLIP filtering')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download, only filter and rebuild split')
    parser.add_argument('--skip_filter', action='store_true',
                        help='Skip CLIP filtering')
    parser.add_argument('--skip_dedup', action='store_true',
                        help='Skip near-duplicate removal')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    retrieved_root = data_root / 'retrieved'

    # Load CLIP model for filtering
    model, preprocess, tokenizer = None, None, None
    if not args.skip_filter or not args.skip_dedup:
        print("\nLoading CLIP model for filtering...")
        model, preprocess, tokenizer = load_clip_for_filtering(device)
        print("CLIP model loaded.\n")

    total_kept = 0

    for class_id in CLASS_IDS:
        class_dir = retrieved_root / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*50}")
        print(f"Class {class_id}: {CLASS_DESCRIPTIONS[class_id]}")
        print(f"{'='*50}")

        # Step 1: Download
        if not args.skip_download:
            print(f"  Downloading (target: {args.num_download})...")
            count = download_images_for_class(class_id, class_dir, args.num_download)
            print(f"  Downloaded: {count} images")
        else:
            existing = len(list(class_dir.glob('*')))
            print(f"  Skip download. Existing: {existing} images")

        # Step 2: Validate (remove broken)
        validate_images(class_dir)

        # Step 3: Deduplicate
        if not args.skip_dedup:
            print("  Checking for near-duplicates...")
            deduplicate_images(class_dir, threshold=0.97, device=device)

        # Step 4: CLIP filter
        if not args.skip_filter:
            print(f"  CLIP filtering (keeping top {args.num_keep})...")
            kept = clip_filter_images(class_dir, class_id, model, preprocess,
                                      tokenizer, args.num_keep, device)
        else:
            kept = list(class_dir.glob('*'))

        kept_count = len(list(class_dir.glob('*')))
        print(f"  Final count for class {class_id}: {kept_count} images\n")
        total_kept += kept_count

    # Step 5: Create split file
    create_retrieval_split(retrieved_root, data_root)

    print(f"\n{'='*50}")
    print(f"Retrieval data preparation complete!")
    print(f"Total images: {total_kept}")
    print(f"Data location: {retrieved_root}")
    print(f"\nNow you can run training with retrieval augmentation:")
    print(f"  pythontrain.py --data_source fewshot+retrieved "
          f"--method finetune-mixed --shots 5")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
