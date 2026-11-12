"""
Video privacy protection using trained object detection model.

Detects license plates (class 1) and employee badges (class 2) in video frames,
then applies blur / pixelation to protect sensitive information.
Faces (class 0) are intentionally kept unblurred.

Supports:
  - Multiple video formats: mp4, avi, mov, mkv, webm
  - Three blur modes: gaussian, pixelate (mosaic), solid_fill
  - Adjustable blur strength and detection thresholds
  - Frame skipping for speed
  - Batch processing of all videos in a folder

Usage:
  # Process a single video
  python privacy_protect.py \
    --model output/detection/best_ensemble.pt \
    --input data/video/protect/camera1.mp4

  # Process all videos in a folder
  python privacy_protect.py \
    --model output/detection/best_ensemble.pt \
    --input data/video/protect/

  # Pixelate (mosaic blur) with strong effect
  python privacy_protect.py \
    --model model.pt \
    --input video.mp4 \
    --blur_mode pixelate \
    --blur_strength 8

  # Fast processing (skip every other frame)
  python privacy_protect.py \
    --model model.pt \
    --input video.mp4 \
    --frame_skip 2

Requirements:
  pip install ultralytics opencv-python numpy Pillow
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
# Configuration
# ==============================================================================

CLASS_NAMES = {0: 'face', 1: 'license_plate', 2: 'badge'}

# Classes to BLUR (privacy-sensitive)
BLUR_CLASSES = {1, 2}  # license_plate and badge
# Class 0 (face) is kept unblurred by default

# Video extensions to scan
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


# ==============================================================================
# Blur Functions
# ==============================================================================

def gaussian_blur(roi, strength=21):
    """Apply Gaussian blur to a region.

    Args:
        roi: image region (H, W, 3)
        strength: kernel size, must be odd. Larger = more blur.
    """
    if strength % 2 == 0:
        strength += 1
    return cv2.GaussianBlur(roi, (strength, strength), 0)


def pixelate_blur(roi, block_size=8):
    """Apply pixelation (mosaic) effect.

    Args:
        roi: image region (H, W, 3)
        block_size: pixel block size. Larger = coarser pixelation.
    """
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return roi

    # Downsample
    small_h = max(1, h // block_size)
    small_w = max(1, w // block_size)
    temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    # Upsample back
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def solid_fill(roi, color=None):
    """Fill region with a solid color.

    Args:
        roi: image region
        color: BGR color tuple. Default: gray.
    """
    if color is None:
        color = (128, 128, 128)
    roi[:] = color
    return roi


def apply_blur(frame, bbox, blur_mode='gaussian', blur_strength=21):
    """Apply blur to a bounding box region in a frame.

    Args:
        frame: numpy array (H, W, 3) in BGR
        bbox: (x1, y1, x2, y2) in pixel coordinates
        blur_mode: 'gaussian', 'pixelate', or 'solid_fill'
        blur_strength: kernel size or block size

    Returns:
        frame with blurred region
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clamp to frame boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]

    if blur_mode == 'gaussian':
        roi = gaussian_blur(roi, blur_strength)
    elif blur_mode == 'pixelate':
        roi = pixelate_blur(roi, blur_strength)
    elif blur_mode == 'solid_fill':
        roi = solid_fill(roi)

    frame[y1:y2, x1:x2] = roi
    return frame


# ==============================================================================
# Video Processing
# ==============================================================================

def process_video(model, video_path, output_path, conf_threshold=0.5,
                  blur_mode='gaussian', blur_strength=21, frame_skip=1,
                  device='cuda', show_progress=True, resize=None):
    """Process a single video: detect sensitive regions and blur them.

    Args:
        model: YOLO model
        video_path: input video path
        output_path: output video path
        conf_threshold: minimum detection confidence
        blur_mode: 'gaussian' | 'pixelate' | 'solid_fill'
        blur_strength: kernel size / block size
        frame_skip: process every Nth frame (1 = every frame, 2 = every other)
        device: 'cuda' or 'cpu'
        show_progress: print progress
        resize: (w, h) optional output resolution

    Returns:
        dict with processing statistics
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_name = ''.join(chr(c) for c in [
        (codec >> 0) & 0xFF,
        (codec >> 8) & 0xFF,
        (codec >> 16) & 0xFF,
        (codec >> 24) & 0xFF,
    ])

    # Output resolution
    out_w = resize[0] if resize else orig_w
    out_h = resize[1] if resize else orig_h

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps / frame_skip, (out_w, out_h))

    if show_progress:
        print(f"\nProcessing: {Path(video_path).name}")
        print(f"  Resolution: {orig_w}×{orig_h}, FPS: {fps:.1f}, "
              f"Frames: {total_frames}")
        print(f"  Output: {out_w}×{out_h}, frame_skip={frame_skip}")

    stats = {
        'video': str(video_path),
        'total_frames': 0,
        'processed_frames': 0,
        'total_detections': 0,
        'per_class': defaultdict(int),
        'avg_detections_per_frame': 0.0,
        'blur_mode': blur_mode,
        'blur_strength': blur_strength,
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats['total_frames'] += 1

        # Frame skipping
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        stats['processed_frames'] += 1

        # Resize if needed
        if resize:
            frame = cv2.resize(frame, (out_w, out_h))

        # Run detection on current frame
        results = model.predict(
            frame, device=device, conf=conf_threshold,
            iou=0.5, verbose=False, imgsz=640
        )

        # Process detections
        frame_detections = 0
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                # Only blur sensitive classes (license_plate, badge)
                if cls_id in BLUR_CLASSES:
                    frame = apply_blur(frame, box, blur_mode, blur_strength)
                    stats['total_detections'] += 1
                    stats['per_class'][int(cls_id)] += 1
                    frame_detections += 1

        # Write processed frame
        out.write(frame)

        if show_progress and stats['processed_frames'] % 100 == 0:
            progress = stats['total_frames'] / total_frames * 100 if total_frames > 0 else 0
            print(f"  [{stats['total_frames']}/{total_frames}] "
                  f"{progress:.0f}% | detections: {stats['total_detections']}")

        frame_idx += 1

    cap.release()
    out.release()

    if stats['processed_frames'] > 0:
        stats['avg_detections_per_frame'] = \
            stats['total_detections'] / stats['processed_frames']

    return stats


# ==============================================================================
# Batch Processing
# ==============================================================================

def find_video_files(input_path):
    """Find all video files in a directory or return a single file.

    Returns: list of Path objects
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return [input_path]
        else:
            print(f"Unsupported file format: {input_path.suffix}")
            return []

    if input_path.is_dir():
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_path.rglob(f'*{ext}'))
            videos.extend(input_path.rglob(f'*{ext.upper()}'))
        return sorted(videos)

    return []


def process_all_videos(model, input_path, output_dir, **kwargs):
    """Process all videos in input_path.

    Args:
        model: YOLO model
        input_path: video file or directory of videos
        output_dir: directory for processed videos
        **kwargs: forwarded to process_video()

    Returns:
        all_stats: list of per-video statistics
    """
    videos = find_video_files(input_path)

    if not videos:
        print(f"No video files found in: {input_path}")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(videos)} video(s) to process")
    all_stats = []

    for i, video_path in enumerate(videos):
        # Build output filename
        stem = video_path.stem
        if kwargs.get('frame_skip', 1) > 1:
            stem += f"_skip{kwargs['frame_skip']}"
        stem += f"_{kwargs.get('blur_mode', 'gaussian')}"
        output_path = output_dir / f"{stem}_protected.mp4"

        print(f"\n[{i+1}/{len(videos)}] {video_path.name}")

        try:
            stats = process_video(model, video_path, output_path, **kwargs)
            all_stats.append(stats)
            print(f"  Done! {stats['total_detections']} regions blurred "
                  f"over {stats['processed_frames']} frames")
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return all_stats


# ==============================================================================
# Pre-Processing Frame Viewer (preview detection before processing)
# ==============================================================================

def preview_detections(model, video_path, num_frames=5, device='cuda',
                       conf_threshold=0.5):
    """Preview detections on sample frames (without processing video).

    Useful for tuning conf_threshold before full processing.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames),
                                 dtype=int)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, device=device, conf=conf_threshold,
                                iou=0.5, verbose=False)

        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                for box, cls_id, conf in zip(boxes, cls_ids, confs):
                    detections.append({
                        'class': CLASS_NAMES.get(cls_id, str(cls_id)),
                        'class_id': int(cls_id),
                        'bbox': [float(x) for x in box],
                        'confidence': float(conf),
                        'will_blur': cls_id in BLUR_CLASSES,
                    })

        print(f"\nFrame {idx}: {len(detections)} detections")
        for d in detections:
            action = "BLUR" if d['will_blur'] else "keep"
            print(f"  [{action}] {d['class']}: conf={d['confidence']:.3f} "
                  f"bbox={[int(x) for x in d['bbox']]}")

    cap.release()


# ==============================================================================
# Reporting
# ==============================================================================

def print_summary(all_stats, output_dir):
    """Print processing summary."""
    print(f"\n{'='*60}")
    print("PRIVACY PROTECTION SUMMARY")
    print(f"{'='*60}")

    total_detections = sum(s['total_detections'] for s in all_stats)
    total_frames = sum(s['processed_frames'] for s in all_stats)
    total_videos = len(all_stats)

    print(f"  Videos processed:     {total_videos}")
    print(f"  Total frames:         {total_frames}")
    print(f"  Total blurs applied:  {total_detections}")
    print(f"  Avg blurs/frame:      {total_detections/max(1,total_frames):.2f}")

    print(f"\n  Per-class counts:")
    class_totals = defaultdict(int)
    for s in all_stats:
        for cls_id, count in s['per_class'].items():
            class_totals[cls_id] += count
    for cls_id in sorted(class_totals.keys()):
        name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        print(f"    {name}: {class_totals[cls_id]}")

    print(f"\n  Per-video:")
    for s in all_stats:
        name = Path(s['video']).name
        print(f"    {name}: {s['total_detections']} blurs / "
              f"{s['processed_frames']} frames")

    print(f"\n  Output directory: {output_dir}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Video Privacy Protection - Blur license plates & badges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video, gaussian blur
  python privacy_protect.py --model best.pt --input data/video/protect/cam1.mp4

  # All videos in folder, pixelate blur
  python privacy_protect.py --model best.pt --input data/video/protect/ --blur_mode pixelate

  # Preview detections first (no processing)
  python privacy_protect.py --model best.pt --input video.mp4 --preview

  # Fast mode: skip frames, lower resolution
  python privacy_protect.py --model best.pt --input video.mp4 --frame_skip 3 --resize 960 540
        """)

    # ---- Model ----
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO detection model (.pt)')

    # ---- Input ----
    parser.add_argument('--input', type=str, default='data/video/protect',
                        help='Video file or directory (default: data/video/protect/)')

    # ---- Blur Settings ----
    parser.add_argument('--blur_mode', type=str, default='gaussian',
                        choices=['gaussian', 'pixelate', 'solid_fill'],
                        help='gaussian=Gaussian blur, pixelate=mosaic, solid_fill=gray box')
    parser.add_argument('--blur_strength', type=int, default=21,
                        help='Blur kernel size (gaussian: odd number, pixelate: block size). '
                             'Higher = stronger blur.')
    parser.add_argument('--blur_classes', type=str, default='1,2',
                        help='Class IDs to blur (default: 1,2 = license_plate,badge; '
                             'class 0=face is never blurred)')

    # ---- Detection ----
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Minimum detection confidence')

    # ---- Performance ----
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Process every Nth frame (1=all, 2=half, 3=third). '
                             'Higher = faster but jerkier.')
    parser.add_argument('--resize', type=int, nargs=2, default=None,
                        metavar=('W', 'H'),
                        help='Resize output (faster processing). e.g.: --resize 960 540')
    parser.add_argument('--device', type=str, default='cuda')

    # ---- Output ----
    parser.add_argument('--output_dir', type=str, default='output/protected_videos')
    parser.add_argument('--save_stats', action='store_true',
                        help='Save processing statistics as JSON')

    # ---- Preview Mode ----
    parser.add_argument('--preview', action='store_true',
                        help='Preview detections only (no video processing)')

    args = parser.parse_args()

    # ---- Validate dependencies ----
    if not OPENCV_AVAILABLE:
        print("ERROR: opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)
    if not ULTRALYTICS_AVAILABLE:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # ---- Validate input ----
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)

    # ---- Parse blur classes ----
    global BLUR_CLASSES
    BLUR_CLASSES = {int(x.strip()) for x in args.blur_classes.split(',')}
    print(f"Blur targets: {[CLASS_NAMES.get(c, str(c)) for c in BLUR_CLASSES]}")

    # ---- Load model ----
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)

    # ---- Preview mode ----
    if args.preview:
        if input_path.is_dir():
            videos = find_video_files(input_path)
            if videos:
                input_path = videos[0]  # Preview first video
                print(f"Previewing first video: {input_path.name}")

        preview_detections(model, input_path, num_frames=5,
                           device=args.device, conf_threshold=args.conf_threshold)
        return

    # ---- Process ----
    start_time = datetime.now()

    all_stats = process_all_videos(
        model=model,
        input_path=input_path,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        blur_mode=args.blur_mode,
        blur_strength=args.blur_strength,
        frame_skip=args.frame_skip,
        device=args.device,
        show_progress=True,
        resize=tuple(args.resize) if args.resize else None,
    )

    # ---- Summary ----
    elapsed = datetime.now() - start_time
    print_summary(all_stats, args.output_dir)
    print(f"\n  Processing time: {elapsed}")

    # ---- Save stats ----
    if args.save_stats and all_stats:
        stats_path = os.path.join(args.output_dir, 'processing_stats.json')
        # Convert defaultdict to regular dict for JSON serialization
        for s in all_stats:
            s['per_class'] = dict(s['per_class'])
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f"  Stats saved: {stats_path}")


if __name__ == '__main__':
    main()
