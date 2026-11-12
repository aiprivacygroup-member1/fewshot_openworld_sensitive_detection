"""
Build a visual reference database from unlabeled images of potential
privacy targets.

Takes unlabeled images (no classification labels needed) and extracts
CLIP visual features to build a reference. During open-world discovery,
unknown detections are compared against this reference — high similarity
means the unknown object looks like a known "type of thing we care about",
helping distinguish genuine new targets from noise.

Usage:
  # Add unlabeled images to data/open_world/reference/ first, then:
  python build_reference.py

  # With custom settings
  python build_reference.py --reference_dir data/my_samples/ --num_clusters 10

Requirements:
  pip install open-clip-torch torch
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.models import load_clip_model


def extract_image_features(model, preprocess, image_path, device='cuda'):
    """Extract CLIP feature for a whole image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
    except Exception as e:
        print(f"  Skipping {image_path}: {e}")
        return None


def extract_multi_scale_features(model, preprocess, image_path,
                                 scales=(1.0, 0.7, 0.5, 0.3), device='cuda'):
    """Extract CLIP features at multiple scales/crops for an image.

    This captures both the whole object and its details, useful for
    matching unknown detections that may appear at different scales.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        features = []

        # Full image
        feat = _encode_crop(model, preprocess, img, device)
        if feat is not None:
            features.append(feat)

        # Center crops at different scales
        for scale in scales:
            if scale >= 1.0:
                continue
            cw, ch = int(w * scale), int(h * scale)
            if cw < 32 or ch < 32:
                continue
            left = (w - cw) // 2
            top = (h - ch) // 2
            crop = img.crop((left, top, left + cw, top + ch))
            feat = _encode_crop(model, preprocess, crop, device)
            if feat is not None:
                features.append(feat)

            # Also take 4 corner + center sub-crops at half scale
            if scale < 0.6:
                half = scale / 2
                hw, hh = int(w * half), int(h * half)
                positions = [
                    (0, 0), (w - hw, 0), (0, h - hh), (w - hw, h - hh),
                    ((w - hw) // 2, (h - hh) // 2),
                ]
                for px, py in positions:
                    if hw >= 32 and hh >= 32:
                        subcrop = img.crop((px, py, px + hw, py + hh))
                        feat = _encode_crop(model, preprocess, subcrop, device)
                        if feat is not None:
                            features.append(feat)

        return features if features else None
    except Exception as e:
        print(f"  Skipping {image_path}: {e}")
        return None


def _encode_crop(model, preprocess, image, device):
    """Encode a single image crop through CLIP."""
    try:
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
    except Exception:
        return None


def build_reference(reference_dir, output_path, model_cfg='vitb32_openclip_laion400m',
                    num_clusters=8, device='cuda'):
    """Build reference feature database from unlabeled images.

    Steps:
      1. Scan images in reference_dir
      2. Extract multi-scale CLIP features from each image
      3. Cluster all features into num_clusters groups
      4. Save cluster centroids + stats as reference database

    Args:
        reference_dir: directory of unlabeled images.
        output_path: where to save the reference .npz file.
        model_cfg: CLIP model config string.
        num_clusters: number of visual pattern clusters.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with statistics.
    """
    reference_dir = Path(reference_dir)
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(sorted(reference_dir.glob(ext)))

    if not image_files:
        raise ValueError(f"No images found in {reference_dir}. "
                         f"Place unlabeled images of potential privacy targets there first.")

    print(f"Found {len(image_files)} reference images")

    print("Loading CLIP model...")
    model, preprocess, _ = load_clip_model(model_cfg, device)

    print("Extracting multi-scale features...")
    all_features = []
    image_stats = []
    for i, img_path in enumerate(image_files):
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(image_files)}]")
        feats = extract_multi_scale_features(model, preprocess, img_path, device=device)
        if feats:
            all_features.extend(feats)
            image_stats.append({
                'file': str(img_path),
                'num_features': len(feats),
            })

    if len(all_features) < num_clusters:
        num_clusters = max(1, len(all_features) // 2)
        print(f"  Adjusted clusters to {num_clusters} (limited by {len(all_features)} features)")

    all_features = np.array(all_features)
    print(f"  Total feature vectors: {all_features.shape[0]} (dim={all_features.shape[1]})")

    # ---- K-means clustering ----
    print(f"Clustering into {num_clusters} visual patterns...")
    centroids, labels = _kmeans(all_features, num_clusters, seed=42)

    # ---- Build reference database ----
    cluster_info = []
    for cid in range(num_clusters):
        mask = labels == cid
        cluster_feats = all_features[mask]
        # Find images contributing to this cluster
        offset = 0
        contributing = []
        for stat in image_stats:
            nf = stat['num_features']
            if np.any(labels[offset:offset + nf] == cid):
                contributing.append(Path(stat['file']).name)
            offset += nf

        cluster_info.append({
            'cluster_id': int(cid),
            'size': int(mask.sum()),
            'ratio': round(mask.sum() / len(all_features), 3),
            'centroid_norm': round(float(np.linalg.norm(centroids[cid])), 4),
            'sample_images': contributing[:5],
        })

    # Save
    np.savez(output_path,
             centroids=centroids,
             num_clusters=num_clusters,
             feature_dim=all_features.shape[1],
             model_cfg=model_cfg)
    print(f"  Saved centroids: {output_path}")

    # Save metadata JSON
    meta_path = Path(output_path).with_suffix('.json')
    meta = {
        'model_cfg': model_cfg,
        'num_clusters': num_clusters,
        'num_images': len(image_files),
        'num_features': int(all_features.shape[0]),
        'feature_dim': int(all_features.shape[1]),
        'clusters': cluster_info,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    return {
        'num_images': len(image_files),
        'num_features': int(all_features.shape[0]),
        'num_clusters': num_clusters,
        'output_path': str(output_path),
    }


def _kmeans(features, k, seed=42, max_iters=100):
    """Simple k-means clustering."""
    np.random.seed(seed)
    n = features.shape[0]

    # Init centroids with k-means++
    centroids = [features[np.random.randint(n)]]
    for _ in range(1, k):
        dists = np.min([np.sum((features - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(features[np.random.choice(n, p=probs)])
    centroids = np.array(centroids)

    for _ in range(max_iters):
        dists = np.zeros((n, k))
        for i in range(k):
            dists[:, i] = np.sum((features - centroids[i]) ** 2, axis=1)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = features[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    dists = np.zeros((n, k))
    for i in range(k):
        dists[:, i] = np.sum((features - centroids[i]) ** 2, axis=1)
    labels = np.argmin(dists, axis=1)

    # Normalize centroids
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    return centroids, labels


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build visual reference database from unlabeled images')
    parser.add_argument('--reference_dir', type=str,
                        default='data/open_world/reference',
                        help='Directory of unlabeled reference images')
    parser.add_argument('--output', type=str,
                        default='data/open_world/reference_db.npz',
                        help='Output path for reference database')
    parser.add_argument('--model_cfg', type=str,
                        default='vitb32_openclip_laion400m')
    parser.add_argument('--num_clusters', type=int, default=8,
                        help='Number of visual pattern clusters (default: 8)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BUILDING VISUAL REFERENCE DATABASE")
    print(f"{'='*60}")
    print(f"  Reference dir: {args.reference_dir}")
    print(f"  Output:        {args.output}")
    print(f"  Clusters:      {args.num_clusters}")
    print(f"  Model:         {args.model_cfg}")

    stats = build_reference(
        reference_dir=args.reference_dir,
        output_path=args.output,
        model_cfg=args.model_cfg,
        num_clusters=args.num_clusters,
        device=args.device,
    )

    print(f"\nDone! {stats['num_images']} images → "
          f"{stats['num_features']} features → "
          f"{stats['num_clusters']} visual patterns")
    print(f"\nNow run discovery with:")
    print(f"  python discover.py --reference_db {args.output} ...")


if __name__ == '__main__':
    main()
