"""
Open-World Privacy Target Discovery

Uses a baseline detection model (trained on known privacy classes) and an
open-world detection model (COCO pretrained) to discover new privacy-sensitive
targets in videos that the baseline model does not recognize.

Pipeline:
  1. Run baseline model -> known objects (face, license_plate, badge)
  2. Run open-world model (YOLOv8n COCO) -> all detectable objects
  3. Detect text-like regions (MSER + edge density) -> small text-carrying targets
  4. IoU filter: open-world + text regions NOT matching any known -> unknown candidates
  5. Track unknowns and knowns across frames
  6. Cluster similar unknowns by geometric features
  7. Compute spatio-temporal correlation between unknown clusters and known classes
  8. Output video clips of highly-correlated unknown objects for human review

Usage:
  python discover.py --baseline_model output/models/detection/best.pt \\
      --video data/video/discovery/

  python discover.py --baseline_model best.pt --video scene.mp4 \\
      --conf_unknown 0.2 --corr_threshold 0.3 --frame_skip 5

Requirements:
  pip install ultralytics opencv-python numpy
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# ==============================================================================
# Constants
# ==============================================================================

KNOWN_CLASSES = {0: 'face', 1: 'license_plate', 2: 'badge'}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush',
}


# ==============================================================================
# IoU Utility
# ==============================================================================

def _compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ==============================================================================
# Simple IoU-based Multi-Object Tracker
# ==============================================================================

class SimpleMultiObjectTracker:
    """IoU-based tracker, no external dependencies.

    Each track stores bbox history, frame indices, class info, and appearance
    count. Matching is greedy by IoU. Tracks not seen for max_age consecutive
    frames are pruned.
    """

    def __init__(self, max_age=60, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections, frame_idx):
        """Update tracker with new detections.

        Args:
            detections: list of dicts with 'bbox', 'class_id', 'class_name',
                        and optionally 'confidence'.
            frame_idx: current video frame index.

        Returns:
            active_tracks: tracks seen >= min_hits times.
        """
        self.frame_count += 1

        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        for track_id, det_idx in matches:
            det = detections[det_idx]
            t = self.tracks[track_id]
            t['bbox_history'].append(det['bbox'])
            t['frame_indices'].append(frame_idx)
            t['total_frames_seen'] += 1
            t['last_seen_frame'] = self.frame_count
            t['class_id'] = det['class_id']
            t['class_name'] = det['class_name']

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks[self.next_id] = {
                'id': self.next_id,
                'bbox_history': [det['bbox']],
                'frame_indices': [frame_idx],
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'total_frames_seen': 1,
                'last_seen_frame': self.frame_count,
            }
            self.next_id += 1

        stale = [
            tid for tid, t in self.tracks.items()
            if self.frame_count - t['last_seen_frame'] > self.max_age
        ]
        for tid in stale:
            del self.tracks[tid]

        return self.get_active()

    def _match(self, detections):
        """Greedy IoU matching between track last bboxes and new detections."""
        if not self.tracks or not detections:
            return [], list(self.tracks.keys()), list(range(len(detections)))

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox_history'][-1] for tid in track_ids]
        det_bboxes = [d['bbox'] for d in detections]

        iou_mat = np.zeros((len(track_ids), len(detections)))
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_mat[i, j] = _compute_iou(tb, db)

        pairs = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if iou_mat[i, j] >= 0.3:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])

        matches = []
        used_t = set()
        used_d = set()
        for _, i, j in pairs:
            if i not in used_t and j not in used_d:
                matches.append((track_ids[i], j))
                used_t.add(i)
                used_d.add(j)

        unmatched_t = [track_ids[i] for i in range(len(track_ids)) if i not in used_t]
        unmatched_d = [j for j in range(len(detections)) if j not in used_d]
        return matches, unmatched_t, unmatched_d

    def get_active(self):
        """Tracks that have been seen at least min_hits times."""
        return [t for t in self.tracks.values()
                if t['total_frames_seen'] >= self.min_hits]

    def get_all(self):
        """All tracks including short-lived ones."""
        return list(self.tracks.values())


# ==============================================================================
# Model Loading
# ==============================================================================

def load_baseline_model(path, device='cuda'):
    """Load the trained 3-class YOLO model."""
    model = YOLO(str(path))
    return model


def load_openworld_model(device='cuda'):
    """Load YOLOv8n pretrained on COCO (80 classes)."""
    model = YOLO('yolov8n.pt')
    return model


# ==============================================================================
# Video Discovery
# ==============================================================================

def find_video_files(input_path):
    """Find all video files in a directory or return a single file."""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return [input_path]
        return []
    if input_path.is_dir():
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_path.rglob(f'*{ext}'))
            videos.extend(input_path.rglob(f'*{ext.upper()}'))
        return sorted(videos)
    return []


def _extract_detections(results, class_map):
    """Extract detection dicts from ultralytics results.

    Args:
        results: ultralytics predict() output.
        class_map: dict of class_id -> class_name.

    Returns:
        list of {'bbox': [x1,y1,x2,y2], 'class_id': int, 'class_name': str,
                 'confidence': float}
    """
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for box, cls_id, conf in zip(boxes, cls_ids, confs):
            detections.append({
                'bbox': box.tolist(),
                'class_id': int(cls_id),
                'class_name': class_map.get(int(cls_id), f'class_{cls_id}'),
                'confidence': float(conf),
            })
    return detections


def _filter_unknowns(ow_dets, known_dets, iou_match):
    """Filter open-world detections: keep only those NOT matching any known."""
    unknowns = []
    for ow in ow_dets:
        max_iou = 0.0
        for kd in known_dets:
            iou_val = _compute_iou(ow['bbox'], kd['bbox'])
            if iou_val > max_iou:
                max_iou = iou_val
        if max_iou < iou_match:
            unknowns.append(ow)
    return unknowns


# ==============================================================================
# Text Region Detection (for small text-carrying privacy targets)
# ==============================================================================

def _detect_text_regions(frame, edge_threshold=15, min_area=200, max_area=20000):
    """Detect likely text regions using MSER + edge density analysis.

    Privacy targets (badges, ID cards, phone screens, documents) are small
    and carry text — exactly what COCO YOLO misses. This detector finds
    text-like regions using two complementary signals:
      1. MSER (Maximally Stable Extremal Regions) — finds stable intensity
         blobs which often correspond to individual characters.
      2. Edge density — text regions have high local gradient density.

    The combination finds clusters of text without needing OCR models.

    Args:
        frame: BGR numpy array (H, W, 3).
        edge_threshold: Canny lower threshold for edge detection.
        min_area: minimum bounding box area in pixels.
        max_area: maximum bounding box area in pixels.

    Returns:
        list of {'bbox': [x1,y1,x2,y2], 'class_id': -1,
                 'class_name': 'text_region', 'confidence': float}
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- MSER: find stable intensity regions (characters) ----
    mser = cv2.MSER_create(
        _delta=5, _min_area=30, _max_area=max_area // 3)
    regions, _ = mser.detectRegions(gray)

    # Convert MSER regions to bounding boxes
    mser_boxes = []
    for region in regions:
        rx, ry, rw, rh = cv2.boundingRect(region)
        if rw * rh >= min_area and rw * rh <= max_area:
            mser_boxes.append([rx, ry, rx + rw, ry + rh])

    if not mser_boxes:
        return []

    # Merge overlapping MSER boxes (characters → text lines)
    merged = _merge_overlapping_boxes(mser_boxes, iou_threshold=0.1)

    # ---- Edge density filter ----
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 3)

    text_regions = []
    for (x1, y1, x2, y2) in merged:
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < min_area or bw * bh > max_area:
            continue
        if bw < 5 or bh < 5:
            continue
        # Edge density in this region
        roi_edges = edges[y1:y2, x1:x2]
        edge_density = np.count_nonzero(roi_edges) / (bw * bh)
        # Aspect ratio filter: text is wider than tall, or roughly square
        ar = bw / bh if bh > 0 else 0
        # Score: edge density weighted by reasonable aspect ratio
        is_text_like = (
            edge_density > 0.03 and
            (0.1 < ar < 15 or 0.5 < ar < 2.0)
        )
        if is_text_like:
            confidence = min(0.9, edge_density * 3.0)
            text_regions.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': -1,
                'class_name': 'text_region',
                'confidence': round(confidence, 4),
            })

    return text_regions


def _merge_overlapping_boxes(boxes, iou_threshold=0.1):
    """Merge overlapping bounding boxes iteratively."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # sort by y, then x
    merged = []
    used = [False] * len(boxes)

    for i, b1 in enumerate(boxes):
        if used[i]:
            continue
        mx1, my1, mx2, my2 = b1
        changed = True
        while changed:
            changed = False
            for j, b2 in enumerate(boxes):
                if used[j] or i == j:
                    continue
                # Check overlap on both axes
                ox1 = max(mx1, b2[0])
                oy1 = max(my1, b2[1])
                ox2 = min(mx2, b2[2])
                oy2 = min(my2, b2[3])
                if ox1 < ox2 and oy1 < oy2:
                    inter = (ox2 - ox1) * (oy2 - oy1)
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    if area2 > 0 and inter / area2 > iou_threshold:
                        mx1 = min(mx1, b2[0])
                        my1 = min(my1, b2[1])
                        mx2 = max(mx2, b2[2])
                        my2 = max(my2, b2[3])
                        used[j] = True
                        changed = True
        used[i] = True
        merged.append((mx1, my1, mx2, my2))

    return merged


def discover_video(baseline_model, openworld_model, video_path, output_dir,
                   conf_known=0.5, conf_unknown=0.3, iou_match=0.3,
                   corr_threshold=0.3, frame_skip=1, device='cuda',
                   detect_text=False, text_edge_threshold=15,
                   reference_db=None, ref_similarity_threshold=0.3,
                   generate_clips=True, show_progress=True):
    """Run open-world discovery on a single video.

    Returns dict with stats, correlations, clip paths, and report path.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if show_progress:
        print(f"\n  Video: {Path(video_path).name}")
        print(f"  Resolution: {orig_w}x{orig_h}, FPS: {fps:.1f}, "
              f"Frames: {total_frames}")

    known_tracker = SimpleMultiObjectTracker(max_age=int(fps * 2), min_hits=3)
    unknown_tracker = SimpleMultiObjectTracker(max_age=int(fps * 2), min_hits=3)

    frame_known_classes = defaultdict(set)
    frame_unknown_tracks = defaultdict(set)
    all_unknown_track_centroids = defaultdict(list)
    all_known_detections = []

    stats = {
        'total_frames': 0,
        'processed_frames': 0,
        'total_known_detections': 0,
        'total_unknown_detections': 0,
        'total_text_regions': 0,
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats['total_frames'] += 1

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        stats['processed_frames'] += 1

        # Step a: baseline model -> known detections
        results_known = baseline_model.predict(
            frame, device=device, conf=conf_known, iou=0.5,
            verbose=False, imgsz=640
        )
        known_dets = _extract_detections(results_known, KNOWN_CLASSES)
        stats['total_known_detections'] += len(known_dets)

        known_class_ids = {d['class_id'] for d in known_dets}
        frame_known_classes[frame_idx] = known_class_ids
        for d in known_dets:
            all_known_detections.append((frame_idx, d['class_id'], d['bbox']))

        # Step b: open-world model -> all COCO detections
        results_ow = openworld_model.predict(
            frame, device=device, conf=conf_unknown, iou=0.5,
            verbose=False, imgsz=640
        )
        ow_dets = _extract_detections(results_ow, COCO_CLASSES)

        # Step b2: detect text-like regions (small text-carrying objects)
        text_dets = []
        if detect_text:
            text_dets = _detect_text_regions(
                frame, edge_threshold=text_edge_threshold,
                min_area=200, max_area=20000)
            stats['total_text_regions'] += len(text_dets)
            # Merge text regions into open-world detections (avoid duplicates)
            for td in text_dets:
                is_dup = False
                for ow in ow_dets:
                    if _compute_iou(td['bbox'], ow['bbox']) > 0.5:
                        is_dup = True
                        break
                if not is_dup:
                    ow_dets.append(td)

        # Step c: filter unknowns
        unknown_dets = _filter_unknowns(ow_dets, known_dets, iou_match)
        stats['total_unknown_detections'] += len(unknown_dets)

        # Step d+e: update trackers
        known_tracker.update(known_dets, frame_idx)
        active_unknown = unknown_tracker.update(unknown_dets, frame_idx)

        # Step f: bookkeeping
        for track in active_unknown:
            frame_unknown_tracks[frame_idx].add(track['id'])
            bbox = track['bbox_history'][-1]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            all_unknown_track_centroids[track['id']].append((frame_idx, cx, cy))

        if show_progress and stats['processed_frames'] % 100 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  [{frame_idx}/{total_frames}] {pct:.0f}% "
                  f"| known: {len(known_dets)} | unknown: {len(unknown_dets)} "
                  f"| tracks: {len(unknown_tracker.tracks)}")

        frame_idx += 1

    cap.release()

    # ---- Post-processing ----
    all_unknown = unknown_tracker.get_all()
    all_known = known_tracker.get_all()

    clusters = _cluster_tracks(all_unknown)
    correlations = _compute_correlations(
        clusters, all_unknown,
        frame_known_classes, frame_unknown_tracks,
        all_unknown_track_centroids, all_known_detections,
        corr_threshold
    )

    # ---- Reference DB scoring (optional) ----
    ref_scored = False
    if reference_db and os.path.exists(reference_db):
        ref_scored = _score_by_reference(
            correlations, all_unknown, video_path,
            reference_db, ref_similarity_threshold, device
        )

    correlated = [c for c in correlations if c['is_correlated']]

    # ---- Generate clips ----
    clip_paths = []
    if generate_clips and correlated:
        clips_dir = output_dir / 'clips'
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_paths = _generate_clips(
            video_path, correlated, all_unknown, all_known,
            clips_dir, fps, orig_w, orig_h
        )

    # ---- Generate report ----
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = _generate_report(
        correlations, clip_paths, video_path, stats,
        reports_dir, corr_threshold
    )

    return {
        'video': str(video_path),
        'stats': stats,
        'num_unknown_tracks': len(all_unknown),
        'num_clusters': len(clusters),
        'num_correlated_clusters': len(correlated),
        'correlations': correlations,
        'clip_paths': clip_paths,
        'report_path': report_path,
    }


# ==============================================================================
# Clustering
# ==============================================================================

def _get_aspect_ratio_bin(ar):
    if ar < 0.5:
        return 'tall_narrow'
    if ar < 1.0:
        return 'portrait'
    if ar < 2.0:
        return 'landscape'
    return 'wide'


def _get_area_bin(rel_area):
    if rel_area < 0.01:
        return 'tiny'
    if rel_area < 0.05:
        return 'small'
    if rel_area < 0.15:
        return 'medium'
    return 'large'


def _cluster_tracks(tracks):
    """Cluster unknown tracks by (class_name, aspect_ratio, relative_size)."""
    if not tracks:
        return []

    track_feats = []
    for t in tracks:
        bboxes = t['bbox_history']
        if len(bboxes) < 2:
            continue
        ars = []
        for b in bboxes:
            w = b[2] - b[0]
            h = b[3] - b[1]
            if h > 0:
                ars.append(w / h)
        mean_ar = np.mean(ars) if ars else 1.0
        b = bboxes[0]
        area = (b[2] - b[0]) * (b[3] - b[1])
        rel_area = area / (640.0 * 640.0)

        track_feats.append({
            'track': t,
            'ar_bin': _get_aspect_ratio_bin(mean_ar),
            'area_bin': _get_area_bin(rel_area),
            'class_name': t['class_name'],
        })

    groups = defaultdict(list)
    for tf in track_feats:
        key = (tf['class_name'], tf['ar_bin'], tf['area_bin'])
        groups[key].append(tf['track'])

    clusters = []
    for cid, ((cname, ar_bin, area_bin), group) in enumerate(groups.items()):
        clusters.append({
            'cluster_id': cid,
            'track_ids': [t['id'] for t in group],
            'class_name': cname,
            'aspect_ratio_bin': ar_bin,
            'area_bin': area_bin,
            'num_tracks': len(group),
            'total_frames': sum(t['total_frames_seen'] for t in group),
        })
    return clusters


# ==============================================================================
# Correlation Analysis
# ==============================================================================

def _compute_correlations(clusters, all_unknown, frame_known_classes,
                          frame_unknown_tracks, unknown_centroids,
                          all_known_detections, corr_threshold):
    """Compute co-occurrence and spatial proximity for each cluster vs known."""
    known_by_frame = defaultdict(lambda: defaultdict(list))
    for frame_idx, cls_id, bbox in all_known_detections:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        known_by_frame[frame_idx][cls_id].append((cx, cy))

    track_frames = {}
    for t in all_unknown:
        track_frames[t['id']] = set(t['frame_indices'])

    results = []
    for cluster in clusters:
        cluster_frames = set()
        for tid in cluster['track_ids']:
            cluster_frames.update(track_frames.get(tid, set()))

        if not cluster_frames:
            continue

        per_class = {}
        correlated = False

        for known_cls_id, known_cls_name in KNOWN_CLASSES.items():
            co_frames = set()
            distances = []
            for f in cluster_frames:
                if known_cls_id in frame_known_classes.get(f, set()):
                    co_frames.add(f)
                    for tid in cluster['track_ids']:
                        for (fidx, ucx, ucy) in unknown_centroids.get(tid, []):
                            if fidx == f:
                                for (kcx, kcy) in known_by_frame.get(f, {}).get(known_cls_id, []):
                                    distances.append(
                                        np.sqrt((ucx - kcx) ** 2 + (ucy - kcy) ** 2))

            co_rate = len(co_frames) / len(cluster_frames)
            avg_dist = float(np.mean(distances)) if distances else None

            per_class[known_cls_name] = {
                'co_occurrence_rate': round(co_rate, 4),
                'co_occurring_frames': len(co_frames),
                'total_cluster_frames': len(cluster_frames),
                'avg_centroid_distance_px': round(avg_dist, 1) if avg_dist else None,
            }

        correlated_with = [
            name for name, c in per_class.items()
            if c['co_occurrence_rate'] >= corr_threshold
        ]
        if correlated_with:
            correlated = True

        results.append({
            'cluster_id': cluster['cluster_id'],
            'class_name': cluster['class_name'],
            'aspect_ratio_bin': cluster['aspect_ratio_bin'],
            'area_bin': cluster['area_bin'],
            'num_tracks': cluster['num_tracks'],
            'total_frames': cluster['total_frames'],
            'total_unique_frames': len(cluster_frames),
            'track_ids': cluster['track_ids'],
            'per_class_correlation': per_class,
            'is_correlated': correlated,
            'correlated_with': correlated_with,
        })

    return results


# ==============================================================================
# Reference Database Scoring
# ==============================================================================

def _score_by_reference(correlations, all_unknown, video_path,
                        reference_db_path, similarity_threshold, device):
    """Score unknown clusters against the reference database.

    For each cluster, extracts CLIP features from representative detection
    crops and computes cosine similarity to reference centroids. Clusters
    with high similarity are boosted: their is_correlated flag is set if
    they weren't already, and a reference_similarity score is added.
    """
    try:
        data = np.load(reference_db_path)
        centroids = data['centroids']
        model_cfg = str(data['model_cfg'])
    except Exception as e:
        print(f"  WARNING: Failed to load reference DB: {e}")
        return False

    # Load CLIP model
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from model.models import load_clip_model
        clip_model, preprocess, _ = load_clip_model(model_cfg, device)
    except Exception as e:
        print(f"  WARNING: Failed to load CLIP for reference scoring: {e}")
        return False

    track_lookup = {t['id']: t for t in all_unknown}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    for cluster in correlations:
        # Sample up to 3 tracks, each contributing their middle-frame crop
        sample_features = []
        for tid in cluster['track_ids'][:3]:
            t = track_lookup.get(tid)
            if not t or len(t['frame_indices']) < 3:
                continue
            mid_idx = len(t['frame_indices']) // 2
            frame_idx = t['frame_indices'][mid_idx]
            bbox = t['bbox_history'][mid_idx]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                crop_pil = Image.fromarray(crop_rgb)
                img_tensor = preprocess(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = clip_model.encode_image(img_tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                sample_features.append(feat.cpu().numpy()[0])
            except Exception:
                continue

        if not sample_features:
            cluster['reference_similarity'] = None
            continue

        sample_feats = np.array(sample_features)
        # Cosine similarity to each centroid
        sims = sample_feats @ centroids.T  # (n_samples, n_centroids)
        max_sims = sims.max(axis=1)        # best match per sample
        avg_sim = float(max_sims.mean())
        best_centroid = int(sims.mean(axis=0).argmax())

        cluster['reference_similarity'] = round(avg_sim, 4)
        cluster['best_reference_centroid'] = best_centroid

        # Boost: if reference similarity is high but correlation was borderline
        if avg_sim >= similarity_threshold and not cluster['is_correlated']:
            # Check if any known class correlation is close to threshold
            for name, c in cluster['per_class_correlation'].items():
                if c['co_occurrence_rate'] >= cluster.get('_corr_threshold', 0.3) * 0.6:
                    cluster['is_correlated'] = True
                    if name not in cluster['correlated_with']:
                        cluster['correlated_with'].append(name)
                    cluster['boosted_by_reference'] = True
                    break

    cap.release()
    return True


# ==============================================================================
# Clip Generation
# ==============================================================================

def _generate_clips(video_path, correlated, all_unknown, all_known,
                    clips_dir, fps, orig_w, orig_h, max_clip_sec=10.0):
    """Extract annotated video clips for correlated unknown clusters."""
    clip_paths = []
    max_frames = int(max_clip_sec * fps)
    merge_gap = int(2.0 * fps)

    track_lookup = {t['id']: t for t in all_unknown}
    known_lookup = {}
    for t in all_known:
        for fidx, bbox in zip(t['frame_indices'], t['bbox_history']):
            known_lookup.setdefault(fidx, []).append((t['class_id'], bbox))

    for cluster in correlated:
        cluster_frames = set()
        for tid in cluster['track_ids']:
            t = track_lookup.get(tid)
            if t:
                cluster_frames.update(t['frame_indices'])
        if not cluster_frames:
            continue

        sorted_frames = sorted(cluster_frames)
        ranges = []
        rs, re = sorted_frames[0], sorted_frames[0]
        for f in sorted_frames[1:]:
            if f - re <= merge_gap:
                re = f
            else:
                ranges.append((rs, re))
                rs, re = f, f
        ranges.append((rs, re))
        ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
        ranges = ranges[:5]

        for ri, (sf, ef) in enumerate(ranges):
            if (ef - sf) / fps < 0.5:
                continue

            segs = []
            cs = sf
            while cs <= ef:
                ce = min(cs + max_frames, ef)
                segs.append((cs, ce))
                cs = ce + 1

            for si, (ss, se) in enumerate(segs):
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, ss)

                fname = (f"c{cluster['cluster_id']:03d}_"
                         f"{cluster['class_name']}_r{ri}_s{si}.mp4")
                clip_path = str(clips_dir / fname)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(clip_path, fourcc, fps, (orig_w, orig_h))

                for fnum in range(ss, se + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Draw known boxes (green)
                    for cls_id, bbox in known_lookup.get(fnum, []):
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = KNOWN_CLASSES.get(cls_id, str(cls_id))
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw unknown boxes (red)
                    for tid in cluster['track_ids']:
                        t = track_lookup.get(tid)
                        if not t:
                            continue
                        for tfidx, tbbox in zip(t['frame_indices'], t['bbox_history']):
                            if tfidx == fnum:
                                x1, y1, x2, y2 = [int(v) for v in tbbox]
                                cv2.rectangle(frame, (x1, y1), (x2, y2),
                                              (0, 0, 255), 2)
                                label = f"{t['class_name']} (new?)"
                                cv2.putText(frame, label, (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2)

                    writer.write(frame)

                writer.release()
                cap.release()
                clip_paths.append(clip_path)

    return clip_paths


# ==============================================================================
# Report
# ==============================================================================

def _generate_report(correlations, clip_paths, video_path, stats,
                     reports_dir, corr_threshold):
    """Generate JSON discovery report."""
    correlated = [c for c in correlations if c['is_correlated']]
    report = {
        'video': str(video_path),
        'video_frames': stats['total_frames'],
        'processed_frames': stats['processed_frames'],
        'timestamp': datetime.now().isoformat(),
        'corr_threshold': corr_threshold,
        'statistics': {
            'total_known_detections': stats['total_known_detections'],
            'total_unknown_detections': stats['total_unknown_detections'],
            'total_clusters': len(correlations),
            'correlated_clusters': len(correlated),
            'clips_generated': len(clip_paths),
        },
        'correlated_discoveries': [
            {
                'cluster_id': c['cluster_id'],
                'coco_class': c['class_name'],
                'aspect_ratio': c['aspect_ratio_bin'],
                'size_category': c['area_bin'],
                'num_tracks': c['num_tracks'],
                'total_frames_seen': c['total_frames'],
                'correlated_with': c['correlated_with'],
                'per_class_correlation': c['per_class_correlation'],
                'reference_similarity': c.get('reference_similarity'),
                'boosted_by_reference': c.get('boosted_by_reference', False),
                'track_ids': c['track_ids'],
            }
            for c in correlated
        ],
        'all_clusters': [
            {
                'cluster_id': c['cluster_id'],
                'coco_class': c['class_name'],
                'aspect_ratio': c['aspect_ratio_bin'],
                'size_category': c['area_bin'],
                'num_tracks': c['num_tracks'],
                'total_frames': c['total_frames'],
                'is_correlated': c['is_correlated'],
                'correlated_with': c['correlated_with'],
                'per_class_correlation': c['per_class_correlation'],
                'reference_similarity': c.get('reference_similarity'),
            }
            for c in correlations
        ],
    }

    stem = Path(video_path).stem
    report_path = str(reports_dir / f'{stem}_discovery_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Open-World Privacy Target Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discover.py --baseline_model output/models/detection/best.pt \\
      --video data/video/discovery/

  python discover.py --baseline_model best.pt --video scene.mp4 \\
      --conf_unknown 0.2 --corr_threshold 0.2

  python discover.py --baseline_model best.pt --video video.mp4 \\
      --frame_skip 5 --no_clips
        """)

    parser.add_argument('--baseline_model', type=str, required=True,
                        help='Path to trained 3-class YOLO model (.pt)')
    parser.add_argument('--video', type=str, required=True,
                        help='Video file or directory of videos')
    parser.add_argument('--conf_known', type=float, default=0.5,
                        help='Confidence for baseline model (default: 0.5)')
    parser.add_argument('--conf_unknown', type=float, default=0.3,
                        help='Confidence for open-world model (default: 0.3)')
    parser.add_argument('--iou_match', type=float, default=0.3,
                        help='IoU threshold for known/unknown matching (default: 0.3)')
    parser.add_argument('--corr_threshold', type=float, default=0.3,
                        help='Co-occurrence rate to flag as correlated (default: 0.3)')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Process every Nth frame (default: 1)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='data/open_world')
    parser.add_argument('--detect_text', action='store_true',
                        help='Enable MSER text region detection for small '
                             'text-carrying targets (badges, screens, docs)')
    parser.add_argument('--text_edge_threshold', type=int, default=15,
                        help='Canny edge threshold for text detection (default: 15)')
    parser.add_argument('--reference_db', type=str, default=None,
                        help='Path to reference .npz file from build_reference.py. '
                             'If provided, unknown clusters are scored against '
                             'unlabeled reference patterns.')
    parser.add_argument('--ref_similarity', type=float, default=0.3,
                        help='Reference similarity threshold for boosting '
                             '(default: 0.3)')
    parser.add_argument('--no_clips', action='store_true',
                        help='Skip clip generation (report only)')

    args = parser.parse_args()

    if not OPENCV_AVAILABLE:
        print("ERROR: opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)
    if not ULTRALYTICS_AVAILABLE:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Input not found: {args.video}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    for d in [output_dir, output_dir / 'clips', output_dir / 'reports']:
        d.mkdir(parents=True, exist_ok=True)

    videos = find_video_files(args.video)
    if not videos:
        print(f"No video files found in: {args.video}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("OPEN-WORLD PRIVACY TARGET DISCOVERY")
    print(f"{'='*60}")
    print(f"  Baseline model: {args.baseline_model}")
    print(f"  Videos to analyze: {len(videos)}")
    print(f"  conf_known: {args.conf_known}")
    print(f"  conf_unknown: {args.conf_unknown}")
    print(f"  iou_match: {args.iou_match}")
    print(f"  corr_threshold: {args.corr_threshold}")
    print(f"  frame_skip: {args.frame_skip}")

    print(f"\nLoading baseline model: {args.baseline_model}")
    baseline_model = load_baseline_model(args.baseline_model, args.device)
    print("Loading open-world model: yolov8n.pt (COCO pretrained)")
    openworld_model = load_openworld_model(args.device)

    start_time = datetime.now()
    all_results = []

    for i, vid in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {vid.name}")
        try:
            result = discover_video(
                baseline_model=baseline_model,
                openworld_model=openworld_model,
                video_path=vid,
                output_dir=output_dir,
                conf_known=args.conf_known,
                conf_unknown=args.conf_unknown,
                iou_match=args.iou_match,
                corr_threshold=args.corr_threshold,
                frame_skip=args.frame_skip,
                device=args.device,
                detect_text=args.detect_text,
                text_edge_threshold=args.text_edge_threshold,
                reference_db=args.reference_db,
                ref_similarity_threshold=args.ref_similarity,
                generate_clips=not args.no_clips,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {vid.name}: {e}")
            continue

    elapsed = datetime.now() - start_time
    total_corr = sum(r['num_correlated_clusters'] for r in all_results)
    total_clips = sum(len(r['clip_paths']) for r in all_results)

    print(f"\n{'='*60}")
    print("DISCOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"  Videos analyzed:        {len(all_results)}")
    print(f"  Correlated clusters:    {total_corr}")
    print(f"  Clips generated:        {total_clips}")
    print(f"  Processing time:        {elapsed}")
    print(f"  Reports:                {output_dir / 'reports'}")
    print(f"  Clips:                  {output_dir / 'clips'}")
    print(f"\n  Next: Review clips in {output_dir / 'clips'}")
    print(f"  If new targets confirmed, add to training data and retrain.")


if __name__ == '__main__':
    main()
