"""
Automatic object detection data labeling using a trained classifier.

Uses multi-scale sliding windows + trained few-shot classifier + NMS to
automatically detect and label objects (face, license_plate, badge) in raw images,
producing standard object detection annotation files.

Supported output formats: YOLO (.txt), COCO (.json), Pascal VOC (.xml)

Usage:
  # Basic usage
  python auto_label.py \
    --model_path output/<run>/wsft_model.pth \
    --input_split unlabeled_scenes.txt

  # With custom parameters
  python auto_label.py \
    --model_path output/<run>/wsft_model.pth \
    --input_split unlabeled_scenes.txt \
    --conf_threshold 0.6 \
    --iou_threshold 0.4 \
    --output_format yolo \
    --visualize 20    # Save 20 annotated images for visual inspection
"""

import os
import sys
import json
import copy
import argparse
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import folder as dataset_parser
from concurrent.futures import ThreadPoolExecutor

import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import roi_align

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.models import MyLinear, load_clip_model

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

NUM_CLASSES = 3
CLASS_NAMES = ['face', 'license_plate', 'badge']
CLASS_NAMES_ZH = ['人脸', '车牌', '工牌']

PROMPT_TEMPLATES = [
    'a photo of a {}.',
    'a close-up photo of a {}.',
    'a bright photo of a {}.',
    'a clear photo of a {}.',
    'a photo of the {}.',
    'a close-up photo of the {}.',
    'a good photo of a {}.',
    'a photo of one {}.',
]

# ---- Sliding window config ----
# Scales: window size as fraction of min(image_w, image_h)
WINDOW_SCALES = [0.15, 0.25, 0.40, 0.60, 0.80]
# Aspect ratios (w/h): covers square, wide, tall windows
WINDOW_ASPECT_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]
# Step size as fraction of window size (~25% of window → 75% overlap)
WINDOW_STRIDE_RATIO = 0.25

# Per-class preferred aspect ratios (for NMS re-scoring)
CLASS_ASPECT_RATIOS = {
    0: [0.75, 1.0, 1.25],     # face: roughly square
    1: [0.4, 0.5, 0.6, 0.75], # license_plate: wide rectangle
    2: [1.25, 1.5, 2.0],       # badge: tall rectangle
}

# ---- Feature-map-based fast detection config ----
DEFAULT_FEATURE_MAP_SIZE = 448       # input resolution for full-image feature extraction
DEFAULT_POOL_SIZE = 2                # RoIAlign output size (pool_size × pool_size)
MIN_FEATURE_CELLS = 0.5              # skip windows smaller than this many feature cells


# ==============================================================================
# Feature-Map-Based Fast Detection Utilities
# ==============================================================================

@torch.no_grad()
def _interpolate_pos_embed(visual, new_h, new_w):
    """Interpolate ViT position embeddings to support higher input resolution.

    CLIP ViT is trained on 224×224 → 7×7 patch grid. For larger inputs
    we interpolate the learned position embeddings to the new grid size.
    """
    pos_embed = visual.positional_embedding  # (1, N_patches+1, dim)
    cls_embed = pos_embed[:, :1, :]          # CLS token embedding
    patch_embed = pos_embed[:, 1:, :]        # patch token embeddings

    num_patches = patch_embed.shape[1]
    old_grid = int(num_patches ** 0.5)

    # Reshape patch embeddings to spatial: (1, old_h, old_w, dim) → (1, dim, old_h, old_w)
    patch_embed = patch_embed.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)

    # Interpolate to target grid
    patch_embed = F.interpolate(
        patch_embed, size=(new_h, new_w), mode='bicubic', align_corners=False,
        antialias=True,
    )

    # Flatten back: (1, dim, new_h, new_w) → (1, new_h*new_w, dim)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)

    # Recompose with CLS token
    new_pos_embed = torch.cat([cls_embed, patch_embed], dim=1)
    visual.positional_embedding.data = new_pos_embed.to(visual.positional_embedding.dtype)


def _extract_normalize_transform(preprocess):
    """Extract just the Normalize transform from CLIP's preprocessing pipeline.

    CLIP preprocess = Compose([Resize(224), CenterCrop(224), ToTensor(), Normalize(...)])
    For full-image feature extraction we replace Resize/CenterCrop but keep Normalize.
    """
    if isinstance(preprocess, T.Compose):
        for t in preprocess.transforms:
            if isinstance(t, T.Normalize):
                return t
    # Fallback: CLIP ViT-B-32 standard normalization
    return T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


@torch.no_grad()
def extract_feature_map(model, preprocess, image, device='cuda',
                        input_size=DEFAULT_FEATURE_MAP_SIZE):
    """Run the full image through ViT once and return spatial feature map.

    Instead of encoding each crop independently (thousands of ViT forward passes),
    we encode the entire image once at higher resolution and extract the patch
    token grid as a spatial feature map for downstream RoIAlign pooling.

    Args:
        model: CLIP model
        preprocess: CLIP preprocessing pipeline
        image: PIL Image
        device: 'cuda' or 'cpu'
        input_size: square resolution to resize image to before ViT encoding

    Returns:
        feature_map: (1, C, H_p, W_p) tensor — spatial patch token features
        scale_x: float — original_pixel_x → feature_map_x
        scale_y: float — original_pixel_y → feature_map_y
    """
    visual = model.visual
    orig_w, orig_h = image.size

    # Resize image to target input size
    img_resized = image.resize((input_size, input_size), Image.BICUBIC)

    # Apply only ToTensor + Normalize (skip CLIP's Resize/CenterCrop)
    normalize = _extract_normalize_transform(preprocess)
    img_tensor = T.Compose([T.ToTensor(), normalize])(img_resized)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Interpolate position embeddings for the new resolution
    new_grid = input_size // visual.conv1.stride[0]  # typically 32 for ViT-B-32
    orig_pos_data = visual.positional_embedding.data.clone()
    _interpolate_pos_embed(visual, new_grid, new_grid)

    # Enable token output mode to retrieve patch tokens
    was_output_tokens = visual.output_tokens
    visual.output_tokens = True

    try:
        _, tokens = visual(img_tensor)  # tokens: (1, N_patches, dim)
    finally:
        # Restore original state
        visual.positional_embedding.data = orig_pos_data
        visual.output_tokens = was_output_tokens

    # Reshape patch tokens to spatial feature map: (1, C, H_p, W_p)
    C = tokens.shape[-1]
    feature_map = tokens.reshape(1, new_grid, new_grid, C).permute(0, 3, 1, 2)

    # Scale factors: original image pixel → feature map coordinate
    scale_x = new_grid / orig_w
    scale_y = new_grid / orig_h

    return feature_map, scale_x, scale_y


@torch.no_grad()
def classify_windows_fast(feature_map, scale_x, scale_y, windows,
                          classifier_head, device='cuda', batch_size=1024,
                          pool_size=DEFAULT_POOL_SIZE):
    """Classify sliding windows via RoIAlign on the shared feature map.

    Replaces per-window crop+resize+ViT encoding with lightweight RoIAlign
    pooling on a single pre-computed feature map. Speedup: 100-500×.

    Args:
        feature_map: (1, C, H_p, W_p) from extract_feature_map()
        scale_x, scale_y: pixel → feature map coordinate scales
        windows: list of (x1, y1, x2, y2, scale, ar)
        classifier_head: MyLinear classifier module
        device: 'cuda' or 'cpu'
        batch_size: number of windows per RoIAlign batch
        pool_size: RoIAlign output spatial size

    Returns:
        scores: (N_valid, 3) class probabilities
        valid_indices: indices into the original windows list
    """
    C, H_f, W_f = feature_map.shape[1:]

    if not windows:
        return np.array([]), []

    # Convert windows from pixel coords to feature map coords
    roi_boxes = []
    valid_indices = []
    for i, (x1, y1, x2, y2, _, _) in enumerate(windows):
        fx1 = x1 * scale_x
        fy1 = y1 * scale_y
        fx2 = x2 * scale_x
        fy2 = y2 * scale_y

        # Skip windows smaller than MIN_FEATURE_CELLS
        if (fx2 - fx1) < MIN_FEATURE_CELLS or (fy2 - fy1) < MIN_FEATURE_CELLS:
            continue

        roi_boxes.append([0, fx1, fy1, fx2, fy2])  # batch_idx=0
        valid_indices.append(i)

    if not roi_boxes:
        return np.array([]), valid_indices

    roi_boxes = torch.tensor(roi_boxes, dtype=torch.float32, device=device)

    # RoIAlign + classify in batches
    all_feats = []
    for i in range(0, len(roi_boxes), batch_size):
        batch_boxes = roi_boxes[i:i + batch_size]
        pooled = roi_align(feature_map, batch_boxes,
                          output_size=(pool_size, pool_size),
                          spatial_scale=1.0, aligned=True)
        # pooled: (N, C, pool_s, pool_s) → (N, C*pool_s^2)
        pooled = pooled.flatten(1)
        # Project back to C-dim feature via mean (preserves original dim for classifier)
        if pool_size > 1:
            pooled = pooled.view(pooled.shape[0], C, -1).mean(dim=-1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        all_feats.append(pooled)

    feats = torch.cat(all_feats, dim=0)
    logits = classifier_head(feats)
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs, valid_indices


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(model_cfg, model_path, device='cuda'):
    """Load CLIP model and fine-tuned classifier from model.models."""
    import open_clip

    model, preprocess, tokenizer = load_clip_model(model_cfg, device)

    # Build classifier head from text prompts (fallback / zero-shot)
    class_names_en = ['human face', 'license plate', 'employee badge']
    prompt_weights = []
    for name in class_names_en:
        prompts = [t.format(name) for t in PROMPT_TEMPLATES]
        tokenized = tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokenized)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            prompt_weights.append(feats.mean(dim=0))
    prompt_weights = torch.stack(prompt_weights, dim=0)
    classifier_head = MyLinear(weights=prompt_weights).to(device)

    # Load fine-tuned weights
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        elif 'clip' in ckpt:
            model.load_state_dict(ckpt['clip'])
        if 'head' in ckpt:
            classifier_head.load_state_dict(ckpt['head'])
        print(f"  Loaded. test_acc={ckpt.get('test_acc', 'N/A')}")

    model.eval()
    classifier_head.eval()
    return model, preprocess, classifier_head


# ==============================================================================
# Sliding Window Generator
# ==============================================================================

def generate_windows(img_w, img_h, scales=None, aspect_ratios=None,
                     stride_ratio=0.25):
    """Generate sliding window coordinates.

    Yields: (x1, y1, x2, y2, scale, aspect_ratio)
      - coordinates are in pixel space
      - scale is the scale factor used
      - aspect_ratio is w/h
    """
    if scales is None:
        scales = WINDOW_SCALES
    if aspect_ratios is None:
        aspect_ratios = WINDOW_ASPECT_RATIOS

    min_dim = min(img_w, img_h)

    windows = []
    for scale in scales:
        base_size = int(min_dim * scale)
        if base_size < 20:  # Skip tiny windows
            continue

        for ar in aspect_ratios:
            # Compute window dimensions
            if ar >= 1.0:
                win_w = base_size
                win_h = int(base_size / ar)
            else:
                win_h = base_size
                win_w = int(base_size * ar)

            if win_w < 20 or win_h < 20:
                continue
            if win_w > img_w or win_h > img_h:
                continue

            # Compute stride
            stride_x = max(1, int(win_w * stride_ratio))
            stride_y = max(1, int(win_h * stride_ratio))

            # Slide
            for y1 in range(0, img_h - win_h + 1, stride_y):
                for x1 in range(0, img_w - win_w + 1, stride_x):
                    x2 = x1 + win_w
                    y2 = y1 + win_h
                    windows.append((x1, y1, x2, y2, scale, ar))

    return windows


# ==============================================================================
# Batch Classifier
# ==============================================================================

@torch.no_grad()
def classify_windows_batch(model, preprocess, classifier_head, image, windows,
                           device='cuda', batch_size=256):
    """Classify all windows in batches.

    Args:
        image: PIL Image
        windows: list of (x1, y1, x2, y2, scale, ar)

    Returns:
        scores: numpy array (N, 3) of class probabilities
    """
    if not windows:
        return np.array([])

    # Sort windows by area (larger first) for better batching
    windows_sorted = sorted(windows, key=lambda w: (w[2]-w[0])*(w[3]-w[1]),
                            reverse=True)

    all_probs = []

    for i in range(0, len(windows_sorted), batch_size):
        batch_windows = windows_sorted[i:i + batch_size]
        crops = []

        for (x1, y1, x2, y2, _, _) in batch_windows:
            crop = image.crop((x1, y1, x2, y2))
            # Resize crop to 224x224 for CLIP
            crop = crop.resize((224, 224), Image.BICUBIC)
            crop_tensor = preprocess(crop)
            crops.append(crop_tensor)

        crop_batch = torch.stack(crops).to(device)
        feats = model.encode_image(crop_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = classifier_head(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    if all_probs:
        scores_all = np.concatenate(all_probs, axis=0)
        # Reorder back to original window order
        # (simple approach: store with windows and reorder)
        combined = list(zip(windows_sorted, scores_all))
        combined.sort(key=lambda x: windows.index(x[0]))
        return np.array([c[1] for c in combined])
    else:
        return np.array([])


# ==============================================================================
# Non-Maximum Suppression (NMS)
# ==============================================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def class_specific_nms(detections, iou_threshold=0.5):
    """Apply class-specific NMS.

    Args:
        detections: list of (x1, y1, x2, y2, class_id, confidence)
        iou_threshold: IoU threshold for suppression

    Returns:
        filtered detections
    """
    if not detections:
        return []

    # Group by class
    by_class = defaultdict(list)
    for det in detections:
        by_class[det[4]].append(det)

    kept = []
    for class_id, class_dets in by_class.items():
        # Sort by confidence descending
        class_dets.sort(key=lambda x: x[5], reverse=True)

        while class_dets:
            best = class_dets.pop(0)
            kept.append(best)
            # Suppress overlapping boxes
            class_dets = [
                d for d in class_dets
                if compute_iou(best[:4], d[:4]) < iou_threshold
            ]

    return kept


def refine_detections(detections, image_w, image_h):
    """Refine detections: clamp to image bounds, remove fully contained boxes."""
    refined = []
    for (x1, y1, x2, y2, class_id, conf) in detections:
        # Clamp
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image_w, int(x2)), min(image_h, int(y2))
        w, h = x2 - x1, y2 - y1

        # Remove degenerate boxes
        if w < 10 or h < 10:
            continue
        # Remove boxes that are too large (>90% of image)
        if w > 0.9 * image_w and h > 0.9 * image_h:
            continue

        refined.append((x1, y1, x2, y2, class_id, conf))

    return refined


# ==============================================================================
# Output Formatters
# ==============================================================================

def save_yolo_format(detections, image_w, image_h, output_path):
    """Save detections in YOLO format.

    YOLO format: class_id cx cy w h
      (all normalized to [0, 1], cx/cy = box center)
    """
    lines = []
    for (x1, y1, x2, y2, class_id, conf) in detections:
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        # Normalize
        cx_n = cx / image_w
        cy_n = cy / image_h
        w_n = w / image_w
        h_n = h / image_h
        lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def save_coco_format(all_detections, all_image_paths, output_path,
                     class_names=None):
    """Save all detections in COCO JSON format.

    Args:
        all_detections: list of lists, one per image
        all_image_paths: list of image file paths
        output_path: output JSON path
        class_names: list of class name strings
    """
    if class_names is None:
        class_names = CLASS_NAMES

    coco = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': i, 'name': name} for i, name in enumerate(class_names)
        ]
    }

    annotation_id = 0
    for image_id, (detections, img_path) in enumerate(
            zip(all_detections, all_image_paths), start=1):
        # Get image dimensions
        try:
            img = Image.open(img_path)
            w, h = img.size
        except:
            w, h = 0, 0

        coco['images'].append({
            'id': image_id,
            'file_name': os.path.basename(img_path),
            'width': w,
            'height': h,
        })

        for (x1, y1, x2, y2, class_id, conf) in detections:
            bw, bh = x2 - x1, y2 - y1
            coco['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': class_id,
                'bbox': [int(x1), int(y1), int(bw), int(bh)],
                'area': int(bw * bh),
                'score': float(conf),
                'iscrowd': 0,
            })
            annotation_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations saved: {output_path} "
          f"({len(coco['images'])} images, {len(coco['annotations'])} boxes)")


def save_voc_format(detections, image_w, image_h, image_path, output_path):
    """Save detections in Pascal VOC XML format."""
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder').text = os.path.dirname(image_path)
    ET.SubElement(annotation, 'filename').text = os.path.basename(image_path)

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_w)
    ET.SubElement(size, 'height').text = str(image_h)
    ET.SubElement(size, 'depth').text = '3'

    for (x1, y1, x2, y2, class_id, conf) in detections:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = CLASS_NAMES[class_id]
        ET.SubElement(obj, 'confidence').text = f"{conf:.4f}"

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(int(x1))
        ET.SubElement(bbox, 'ymin').text = str(int(y1))
        ET.SubElement(bbox, 'xmax').text = str(int(x2))
        ET.SubElement(bbox, 'ymax').text = str(int(y2))

    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


# ==============================================================================
# Visualization
# ==============================================================================

def draw_detections(image_path, detections, output_path):
    """Draw bounding boxes on image for visual inspection."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Color per class
    colors = {
        0: (0, 255, 0),    # face: green
        1: (255, 0, 0),    # license_plate: red
        2: (0, 128, 255),  # badge: blue
    }

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2, class_id, conf) in detections:
        color = colors[class_id]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{CLASS_NAMES[class_id]} {conf:.2f}"
        # Background for text
        bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill='white', font=font)

    img.save(output_path)
    return output_path


# ==============================================================================
# Main Detection Pipeline
# ==============================================================================

def detect_single_image(model, preprocess, classifier_head, image_path,
                        device, conf_threshold, iou_threshold,
                        scales=None, aspect_ratios=None,
                        use_fast_mode=True, feature_map_size=DEFAULT_FEATURE_MAP_SIZE):
    """Run detection on a single image.

    Two modes:
      - Fast mode (default): encode full image once → RoIAlign window features.
        Replaces thousands of per-crop ViT forwards with one shared forward pass.
        ~100-500× faster window classification.
      - Slow mode: original per-window crop+encode (kept as fallback).

    Returns:
        detections: list of (x1, y1, x2, y2, class_id, confidence)
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  Skipping {image_path}: {e}")
        return []

    img_w, img_h = image.size

    # Generate sliding windows
    windows = generate_windows(img_w, img_h, scales, aspect_ratios,
                               WINDOW_STRIDE_RATIO)

    if not windows:
        return []

    if use_fast_mode:
        # ---- Fast: full-image feature map + RoIAlign ----
        feature_map, scale_x, scale_y = extract_feature_map(
            model, preprocess, image, device, input_size=feature_map_size)

        scores, valid_indices = classify_windows_fast(
            feature_map, scale_x, scale_y, windows, classifier_head, device)

        if len(scores) == 0:
            return []

        # Build detections only for windows that survived the feature-cell filter
        detections = []
        for local_idx, score_row in enumerate(scores):
            win_idx = valid_indices[local_idx]
            x1, y1, x2, y2, scale, ar = windows[win_idx]

            best_class = int(np.argmax(score_row))
            best_conf = float(score_row[best_class])

            if best_conf < conf_threshold:
                continue

            ar_bonus = 0.0
            if best_class in CLASS_ASPECT_RATIOS:
                for pref_ar in CLASS_ASPECT_RATIOS[best_class]:
                    if abs(ar - pref_ar) < 0.2:
                        ar_bonus = 0.05
                        break

            detections.append((x1, y1, x2, y2, best_class, best_conf + ar_bonus))
    else:
        # ---- Slow: original per-window crop + encode ----
        scores = classify_windows_batch(model, preprocess, classifier_head,
                                        image, windows, device)

        if len(scores) == 0:
            return []

        detections = []
        for (x1, y1, x2, y2, scale, ar), score in zip(windows, scores):
            best_class = int(np.argmax(score))
            best_conf = float(score[best_class])

            if best_conf < conf_threshold:
                continue

            ar_bonus = 0.0
            if best_class in CLASS_ASPECT_RATIOS:
                preferred_ars = CLASS_ASPECT_RATIOS[best_class]
                for pref_ar in preferred_ars:
                    if abs(ar - pref_ar) < 0.2:
                        ar_bonus = 0.05
                        break

            detections.append((x1, y1, x2, y2, best_class, best_conf + ar_bonus))

    # NMS
    kept = class_specific_nms(detections, iou_threshold)

    # Refine
    kept = refine_detections(kept, img_w, img_h)

    return kept


# ==============================================================================
# Batch Processing
# ==============================================================================

def read_split_file(data_root, split_file):
    """Read a SWAT-format split file, return list of image paths."""
    data_root = Path(data_root)
    images = []
    filepath = data_root / split_file
    if not filepath.exists():
        print(f"Split file not found: {filepath}")
        return images

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
            img_path = data_root / parts[0]
            if img_path.exists():
                images.append(str(img_path))

    return images


def main():
    parser = argparse.ArgumentParser(
        description='Auto-label detection data using trained classifier')
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m')
    # Input
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate')
    parser.add_argument('--input_split', type=str, default='raw_scenes/unlabeled_scenes.txt',
                        help='Split file with images to label')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Or, directly specify a directory of images')
    # Detection params
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Minimum confidence for a detection')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--scales', type=str, default=None,
                        help='Comma-separated window scales, e.g. "0.2,0.4,0.6"')
    parser.add_argument('--aspect_ratios', type=str, default=None,
                        help='Comma-separated aspect ratios, e.g. "0.5,1.0,2.0"')
    # Output
    parser.add_argument('--output_format', type=str, default='yolo',
                        choices=['yolo', 'coco', 'voc', 'all'])
    parser.add_argument('--output_dir', type=str, default='output/labels')
    parser.add_argument('--visualize', type=int, default=20,
                        help='Number of images to save with drawn boxes (0=none)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images to process (for testing)')
    # Feature-map fast mode (default on)
    parser.add_argument('--slow_mode', action='store_true', default=False,
                        help='Use original per-window crop+encode (slow) instead of '
                             'feature-map-based fast detection')
    parser.add_argument('--feature_map_size', type=int,
                        default=DEFAULT_FEATURE_MAP_SIZE,
                        help=f'Input resolution for full-image feature extraction '
                             f'(default: {DEFAULT_FEATURE_MAP_SIZE})')
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Window classification batch size')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Parse scales and aspect ratios
    scales = None
    if args.scales:
        scales = [float(s) for s in args.scales.split(',')]
    aspect_ratios = None
    if args.aspect_ratios:
        aspect_ratios = [float(a) for a in args.aspect_ratios.split(',')]

    # ---- Load model ----
    print("\nLoading model...")
    model, preprocess, classifier_head = load_model(
        args.model_cfg, args.model_path, device
    )

    # ---- Gather images ----
    if args.input_dir:
        img_dir = Path(args.input_dir)
        image_paths = sorted([str(p) for p in img_dir.glob('*')
                              if p.suffix.lower() in
                              ('.jpg', '.jpeg', '.png', '.bmp')])
    else:
        image_paths = read_split_file(args.data_root, args.input_split)

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    use_fast_mode = not args.slow_mode
    print(f"Processing {len(image_paths)} images...")
    if use_fast_mode:
        print(f"  Mode: FAST (feature-map RoIAlign, input_size={args.feature_map_size})")
    else:
        print(f"  Mode: SLOW (per-window crop+encode)")
    print(f"  Window scales: {scales or WINDOW_SCALES}")
    print(f"  Aspect ratios: {aspect_ratios or WINDOW_ASPECT_RATIOS}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")

    # ---- Output dirs ----
    output_dir = Path(args.output_dir)
    label_dir = output_dir / 'labels'
    vis_dir = output_dir / 'visualizations'
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process images ----
    all_detections = []   # Per-image detection lists
    processed_paths = []
    total_boxes = 0
    images_with_detections = 0

    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(image_paths)}] "
                  f"Processing: {os.path.basename(img_path)}")

        detections = detect_single_image(
            model, preprocess, classifier_head, img_path, device,
            args.conf_threshold, args.iou_threshold, scales, aspect_ratios,
            use_fast_mode=use_fast_mode,
            feature_map_size=args.feature_map_size,
        )

        all_detections.append(detections)
        processed_paths.append(img_path)
        total_boxes += len(detections)
        if detections:
            images_with_detections += 1

        # ---- Save per-image labels ----
        base_name = Path(img_path).stem

        if args.output_format in ('yolo', 'all'):
            try:
                img = Image.open(img_path)
                iw, ih = img.size
            except:
                iw, ih = 0, 0
            save_yolo_format(detections, iw, ih,
                             label_dir / f"{base_name}.txt")

        if args.output_format in ('voc', 'all'):
            try:
                img = Image.open(img_path)
                iw, ih = img.size
            except:
                iw, ih = 0, 0
            save_voc_format(detections, iw, ih, img_path,
                            label_dir / f"{base_name}.xml")

        # ---- Visualize ----
        if args.visualize > 0 and idx < args.visualize:
            vis_path = vis_dir / f"{base_name}_labeled.jpg"
            draw_detections(img_path, detections, str(vis_path))

    # ---- Save COCO JSON ----
    if args.output_format in ('coco', 'all'):
        save_coco_format(all_detections, processed_paths,
                         output_dir / 'annotations.json')

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("AUTO-LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Images processed:       {len(processed_paths)}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Total bounding boxes:   {total_boxes}")
    print(f"  Avg boxes per image:    {total_boxes / max(1, len(processed_paths)):.1f}")
    print(f"  Labels saved to:        {label_dir}")
    if args.visualize > 0:
        print(f"  Visualizations saved:   {vis_dir}")

    # Class distribution
    class_counts = defaultdict(int)
    for dets in all_detections:
        for d in dets:
            class_counts[d[4]] += 1
    print("  Class distribution:")
    for cid in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[cid]} ({CLASS_NAMES_ZH[cid]}): "
              f"{class_counts[cid]} boxes")


if __name__ == '__main__':
    main()
