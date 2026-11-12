# Exploration Beyond Boundaries: Few-Shot Open-World Sensitive Object Detection via Dual-Model Differential Detection

This repository implements a complete pipeline for **few-shot open-world sensitive object detection**. The core idea: train a detector with only a handful of labeled samples per class (few-shot), automatically discover new, previously-unseen sensitive objects in surveillance video via **dual-model differential detection** coupled with spatiotemporal association, and finally apply differentiated protection (blur / mosaic / SimSwap face replacement) to the recognized faces, license plates, badges, and other sensitive targets.

The system is composed of five functional modules, with data flowing along the chain "classification → auto-labeling → detection → discovery → protection".

## Overall Framework

```
                ┌──────────────────────────────────────────────────────────┐
                │              Few-Shot Classification (SWAT)               │
                │   classify/  SWAT two-stage retrieval-augmented           │
                │              fine-tuning (CLIP ViT-B/32 backbone)         │
                └───────────────────────────┬──────────────────────────────┘
                                            │ trained classifier
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │              Weakly-Supervised Auto-Labeling              │
                │   auto_label/  shared feature map + multi-scale           │
                │                sliding window + RoIAlign                  │
                └───────────────────────────┬──────────────────────────────┘
                                            │ YOLO / COCO / VOC annotations
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │              Few-Shot Object Detection                    │
                │   object_detection/  two-stage YOLOv8 training            │
                │                      + model fusion                      │
                └──────────────┬───────────────────────┬────────────────────┘
                               │                       │
                               ▼                       ▼
                ┌──────────────────────┐   ┌──────────────────────────────┐
                │ Open-World Sensitive │   │      Privacy Protection       │
                │   Object Discovery   │   │       privacy_protect/        │
                │ open_world_discovery/│   │  license plate / badge        │
                │  dual-model          │   │  blur · mosaic                │
                │  differencing +      │   │  face SimSwap replacement     │
                │  tracking +          │   │                               │
                │  clustering +        │   │                               │
                │  spatiotemporal      │   │                               │
                │  association         │   │                               │
                └──────────────────────┘   └──────────────────────────────┘
```

- **model/**: Shared model definitions (CLIP classification head, SimSwap generator, etc.).
- **classify/**: Few-shot classification implementing SWAT (Stage-wise Retrieval-Augmented Finetuning).
- **auto_label/**: Weakly-supervised auto-labeling based on a shared feature map, turning "encode-every-window-independently" into "encode-once, query-many".
- **object_detection/**: Two-stage YOLOv8 training that adapts the SWAT idea to object detection.
- **open_world_discovery/**: Open-world discovery of sensitive objects.
- **privacy_protect/**: Differentiated privacy protection for detected sensitive objects.

## Directory Structure

```
├── README.md
├── requirements.txt
├── model/                          # Shared model definitions
│   ├── __init__.py
│   └── models.py                   # CLIP loader, classification head, SimSwap generator
├── classify/                       # Few-shot classification (SWAT)
│   ├── config.yml                  # Classification training config
│   ├── demo_data.py                # Generate synthetic data for end-to-end validation
│   ├── prepare_data.py             # Split few-shot train/test sets
│   ├── retrieve_data.py            # Download retrieval-augmented images
│   ├── train.py                    # Two-stage training (WiSE-FT, Tau Norm)
│   └── test.py                     # Evaluation + single-image inference
├── auto_label/                     # Weakly-supervised auto-labeling
│   ├── auto_label.py               # Sliding window + RoIAlign auto-labeling
│   └── crawl_detection_images.py   # Crawl scene images
├── object_detection/               # Few-shot object detection
│   ├── det_config.yml              # Detection training config
│   ├── det_dataset.py              # Merge hand-labeled + auto-labeled data
│   ├── det_train.py                # Two-stage training + model fusion
│   └── det_test.py                 # Evaluation / comparison / inference / ONNX export
├── open_world_discovery/           # Open-world sensitive-object discovery
│   ├── build_reference.py          # Build CLIP visual reference library
│   └── discover.py                 # Dual-model differential + spatiotemporal discovery
├── privacy_protect/                # Privacy protection
│   ├── badge_plate_privacy_protect.py  # License plate / badge blur · mosaic
│   └── face_privacy_protect.py         # Face SimSwap synthetic replacement
├── data/                           # Data directory (see "Data Preparation" below)
└── output/                         # Output directory (models / labels / results / protected videos)
```

## Environment

Python 3.9+ recommended, CUDA environment recommended.

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `open-clip-torch`, `ultralytics` (YOLOv8), `opencv-python`, `numpy`, `Pillow`, `PyYAML`, `requests`, `scikit-image`.

## Data Preparation

Images are stored by class in separate folders, supporting `.jpg / .jpeg / .png / .bmp`:

```
data/face_badge_plate/
├── face/              # all face images
├── license_plate/     # all license plate images
└── badge/             # all employee badge images
```

Class mapping: `0 = face`, `1 = license_plate`, `2 = badge`. Each class needs at least `shots + 2` images (e.g., 5-shot needs at least 7 images per class).

> Without real data, run `python classify/demo_data.py` to generate synthetic data and validate the whole pipeline.

## Usage

### 1. Few-Shot Classification (SWAT)

```bash
# 1) Split train/test sets
python classify/prepare_data.py --shots 5 --seed 1

# 2) Download retrieval-augmented images (200 per class, keep 100 after CLIP filtering)
python classify/retrieve_data.py --num_download 200 --num_keep 100

# 3) SWAT training (default mixup-fs method, see classify/config.yml)
python classify/train.py --data_source fewshot+retrieved --method mixup-fs --shots 5

# 4) Evaluation
python classify/test.py --model_path output/<run>/wsft_model.pth --mode evaluate

# 5) Single-image inference
python classify/test.py --model_path output/<run>/wsft_model.pth --mode single --image xxx.jpg
```

### 2. Auto-Labeling

```bash
# Crawl scene images
python auto_label/crawl_detection_images.py --num_per_class 500

# Auto-label with the trained classifier (YOLO format + 30 visualizations)
python auto_label/auto_label.py \
  --model_path output/<run>/wsft_model.pth \
  --input_split raw_scenes/unlabeled_scenes.txt \
  --conf_threshold 0.6 --iou_threshold 0.4 \
  --output_format yolo --visualize 30
```

### 3. Few-Shot Object Detection

```bash
# 1) Merge hand-labeled + auto-labeled dataset
python object_detection/det_dataset.py \
  --hand_dir data/det_hand_labeled \
  --auto_label_dir output/labels \
  --auto_image_dir data/face_badge_plate/raw_scenes

# 2) Two-stage training
python object_detection/det_train.py \
  --data output/detection/dataset/data.yaml \
  --model yolov8n --oversample_hand 3 \
  --stage1_epochs 50 --stage2_epochs 30

# 3) Evaluation / inference / ONNX export
python object_detection/det_test.py --mode eval \
  --model output/detection/ensemble/best_ensemble.pt \
  --data output/detection/dataset/data.yaml
```

### 4. Open-World Sensitive-Object Discovery

```bash
# 1) Build the visual reference library (first place unlabeled reference images in data/open_world/reference/)
python open_world_discovery/build_reference.py --num_clusters 8

# 2) Run the discovery pipeline
python open_world_discovery/discover.py \
  --baseline_model output/models/detection/best.pt \
  --video data/video/discovery/ \
  --conf_unknown 0.2 --corr_threshold 0.3 --frame_skip 5
```

Output: video clips of the discovered unknown objects (`data/open_world/clips/`, green box = known, red box = unknown) and JSON reports (`data/open_world/reports/`).

### 5. Privacy Protection

```bash
# Option A: blur license plates + badges (faces preserved)
python privacy_protect/badge_plate_privacy_protect.py \
  --model output/models/detection/best_ensemble.pt \
  --input data/video/protect/video.mp4 \
  --blur_mode gaussian --blur_strength 31

# Option B: SimSwap synthetic face replacement (irreversible)
python privacy_protect/face_privacy_protect.py \
  --detection_model output/models/detection/best_ensemble.pt \
  --simswap_model model/weights/90000_net_G.pth \
  --input data/video/protect/
```

Protected videos are output to `output/protected_videos/`.

## Model Weights

- **SimSwap generator weights** (`model/weights/90000_net_G.pth`, ~220MB): required by `face_privacy_protect.py`, but **not included in this code package**. Download the SimSwap pretrained weights yourself and place them under `model/weights/`, or specify the path via `--simswap_model`.
- Classification / detection models are training artifacts output to `output/models/` after training, and reused by the auto-labeling, discovery, and protection modules.

## Notes

Scripts locate the project-root `model` package via `sys.path.insert(...)`, so run commands from the project root (or keep `model/` at the same level as the other modules); otherwise adjust `PYTHONPATH` yourself. Legacy paths such as `custom/output` in the config files may not match your local layout — adjust them to your directory structure as needed.
