"""
Detection model evaluation and inference.

Supports:
  1. Dataset evaluation — mAP, per-class metrics, confusion matrix
  2. Single image inference — draw boxes, save output
  3. Folder batch inference — process all images
  4. Export to ONNX / TensorRT

Usage:
  # Evaluate on validation set
  pythondet_test.py --model best.pt --data data.yaml

  # Single image inference
  pythondet_test.py --model best.pt --source image.jpg --save

  # Folder batch inference
  pythondet_test.py --model best.pt --source /path/to/images/ --save

  # Compare two models (Stage 1 vs Stage 2 vs Ensemble)
  pythondet_test.py --compare stage1.pt stage2.pt ensemble.pt --data data.yaml
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
import yaml

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def load_class_names(data_yaml_path):
    """Load class names from data.yaml."""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', ['face', 'license_plate', 'badge'])


def evaluate_model(model_path, data_yaml, device='cuda', imgsz=640,
                   conf=0.5, iou=0.5):
    """Evaluate a model on a YOLO dataset.

    Returns:
        dict with mAP@50, mAP@50-95, per-class metrics, confusion matrix
    """
    print(f"\nEvaluating: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, imgsz=imgsz,
                        conf=conf, iou=iou, verbose=False)

    results = {}

    # mAP metrics
    if hasattr(metrics, 'box'):
        results['mAP50'] = float(metrics.box.map50)
        results['mAP50_95'] = float(metrics.box.map)
        results['mAP75'] = float(metrics.box.map75)

        # Per-class mAP
        if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap is not None:
            class_names = load_class_names(data_yaml)
            ap_per_class = metrics.box.ap.tolist() if hasattr(metrics.box.ap, 'tolist') else list(metrics.box.ap)
            for i, ap in enumerate(ap_per_class):
                if i < len(class_names):
                    results[f'ap_{class_names[i]}'] = float(ap)

        # Precision / Recall
        results['precision'] = float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0
        results['recall'] = float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0

    return results


def compare_models(model_paths, data_yaml, device='cuda', imgsz=640):
    """Compare multiple models side by side."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")

    all_results = {}
    for model_path in model_paths:
        name = Path(model_path).stem
        results = evaluate_model(model_path, data_yaml, device, imgsz)
        all_results[name] = results

    # Print comparison table
    print(f"\n{'Model':40s} {'mAP@50':>8s} {'mAP@50-95':>10s} {'Precision':>10s} {'Recall':>10s}")
    print('-' * 80)
    for name, r in all_results.items():
        print(f"{name:40s} {r.get('mAP50', 0):8.4f} {r.get('mAP50_95', 0):10.4f} "
              f"{r.get('precision', 0):10.4f} {r.get('recall', 0):10.4f}")

    # Per-class comparison
    class_names = load_class_names(data_yaml)
    print(f"\n{'Per-Class AP':40s}", end='')
    for name in class_names:
        print(f" {name:>10s}", end='')
    print()
    print('-' * (40 + 11 * len(class_names)))
    for model_name, r in all_results.items():
        print(f"{model_name:40s}", end='')
        for cls_name in class_names:
            ap_key = f'ap_{cls_name}'
            print(f" {r.get(ap_key, 0):10.4f}", end='')
        print()

    return all_results


def run_inference(model_path, source, output_dir, device='cuda', imgsz=640,
                  conf=0.5, iou=0.5, save_txt=True, save_conf=True):
    """Run inference on images/video/webcam.

    Args:
        model_path: path to .pt model
        source: image path, directory, or video path
        output_dir: where to save results
    """
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(
        source=source,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        project=output_dir,
        name='inference',
        exist_ok=True,
    )

    # Count detections per class
    class_names = load_class_names(
        os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'data.yaml')
    )
    # Fallback names
    if not class_names:
        class_names = ['face', 'license_plate', 'badge']

    class_counts = defaultdict(int)
    total_boxes = 0
    for r in results:
        if r.boxes is not None:
            cls_ids = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls
            for cid in cls_ids:
                class_counts[int(cid)] += 1
                total_boxes += 1

    print(f"\nInference Results:")
    print(f"  Total detections: {total_boxes}")
    for cid in range(3):
        name = class_names[cid] if cid < len(class_names) else f'class_{cid}'
        print(f"  {name}: {class_counts[cid]}")

    return results


def export_model(model_path, output_dir, formats=('onnx',)):
    """Export model to ONNX / TensorRT for deployment."""
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        print(f"Exporting to {fmt}...")
        export_path = model.export(format=fmt, imgsz=640)
        print(f"  Exported: {export_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Detection Model Evaluation and Inference')

    # ---- Mode ----
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['eval', 'compare', 'infer', 'export'],
                        help='Operation mode')

    # ---- Model ----
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (.pt)')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='Multiple model paths for comparison')

    # ---- Data ----
    parser.add_argument('--data', type=str,
                        default='output/detection/dataset/data.yaml',
                        help='data.yaml path')
    parser.add_argument('--source', type=str, default=None,
                        help='Image/folder/video for inference')

    # ---- Params ----
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')

    # ---- Output ----
    parser.add_argument('--output_dir', type=str,
                        default='output/detection/results')
    parser.add_argument('--save_json', action='store_true',
                        help='Save metrics as JSON')

    # ---- System ----
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if not ULTRALYTICS_AVAILABLE:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Mode: Compare ----
    if args.mode == 'compare':
        model_paths = args.compare or [args.model]
        if not model_paths or len(model_paths) < 2:
            print("ERROR: --compare needs at least 2 model paths")
            sys.exit(1)
        all_results = compare_models(model_paths, args.data, args.device, args.imgsz)

        if args.save_json:
            path = os.path.join(args.output_dir, 'comparison.json')
            with open(path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nSaved: {path}")

    # ---- Mode: Evaluate ----
    elif args.mode == 'eval':
        if not args.model:
            print("ERROR: --model required for eval mode")
            sys.exit(1)

        results = evaluate_model(args.model, args.data, args.device,
                                 args.imgsz, args.conf, args.iou)

        class_names = load_class_names(args.data)
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"  mAP@50:     {results.get('mAP50', 0):.4f}")
        print(f"  mAP@50-95:  {results.get('mAP50_95', 0):.4f}")
        print(f"  mAP@75:     {results.get('mAP75', 0):.4f}")
        print(f"  Precision:  {results.get('precision', 0):.4f}")
        print(f"  Recall:     {results.get('recall', 0):.4f}")
        print(f"  Per-class AP:")
        for name in class_names:
            ap_key = f'ap_{name}'
            if ap_key in results:
                print(f"    {name}: {results[ap_key]:.4f}")

        if args.save_json:
            path = os.path.join(args.output_dir, 'evaluation.json')
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nSaved: {path}")

    # ---- Mode: Inference ----
    elif args.mode == 'infer':
        if not args.model or not args.source:
            print("ERROR: --model and --source required for inference")
            sys.exit(1)

        run_inference(args.model, args.source, args.output_dir,
                      args.device, args.imgsz, args.conf, args.iou)

    # ---- Mode: Export ----
    elif args.mode == 'export':
        if not args.model:
            print("ERROR: --model required for export")
            sys.exit(1)
        export_model(args.model, args.output_dir, formats=('onnx',))


if __name__ == '__main__':
    main()
