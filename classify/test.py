"""
Testing and inference script for the 3-class few-shot model.
Supports:
  1. Batch evaluation on a test set (with metrics)
  2. Single image inference
  3. Folder-level batch inference
"""

import os
import sys
import json
import copy
import argparse
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import pathlib
from torchvision.datasets import folder as dataset_parser

import open_clip

warnings.filterwarnings('ignore')

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



from train import FewShotDataset, validate, compute_metrics, ClassifierHead


def load_model(model_cfg='vitb32_openclip_laion400m', model_path=None, device='cuda'):
    """Load CLIP model and optionally fine-tuned weights."""
    cfgs = model_cfg.split('_')
    arch, model_name, pretrain = cfgs[0], cfgs[1], cfgs[2]

    model_map = {
        'laion400m': {
            'vitb32': ('ViT-B-32-quickgelu', 'laion400m_e32'),
            'vitb16': ('ViT-B-16', 'laion400m_e32'),
        },
        'laion2b': {
            'vitb32': ('ViT-B-32', 'laion2b_e16'),
        }
    }

    model_arch, pretrained = model_map[pretrain][arch]
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_arch, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_arch)
    model.float()
    model.to(device)

    # Build classifier head from text prompts (zero-shot)
    class_names_en = ['human face', 'license plate', 'employee badge']
    prompt_tensors = build_prompt_tensors(model, tokenizer, class_names_en, device)
    weights = torch.stack([prompt_tensors[str(i)]['mean'] for i in range(NUM_CLASSES)], dim=0)
    classifier_head = ClassifierHead(weights).to(device)

    # Load fine-tuned weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        ckpt = torch.load(model_path, map_location=device)

        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if 'head' in ckpt:
                classifier_head.load_state_dict(ckpt['head'])
            print(f"Loaded fine-tuned model (test_acc={ckpt.get('test_acc', 'N/A')})")
        elif 'clip' in ckpt:
            model.load_state_dict(ckpt['clip'])
            if 'head' in ckpt:
                classifier_head.load_state_dict(ckpt['head'])
            print(f"Loaded SWAT checkpoint (test_acc={ckpt.get('test_acc', 'N/A')})")
        else:
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
            print("Loaded model weights (partial match)")

    model.eval()
    classifier_head.eval()

    return model, preprocess, tokenizer, classifier_head


def build_prompt_tensors(model, tokenizer, class_names, device='cuda'):
    """Build text prompt embeddings for each class."""
    prompt_tensors = {}
    for class_id, name in enumerate(class_names):
        prompts = [t.format(name) for t in PROMPT_TEMPLATES]
        tokenized = tokenizer(prompts).to(device)
        with torch.no_grad():
            features = model.encode_text(tokenized)
            features = features / features.norm(dim=-1, keepdim=True)
            mean_feat = features.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
        prompt_tensors[str(class_id)] = {
            'all': features.cpu(),
            'mean': mean_feat.cpu()
        }
    return prompt_tensors


@torch.no_grad()
def predict_image(model, preprocess, classifier_head, image_path, device='cuda'):
    """Predict class for a single image."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    feats = model.encode_image(img_tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    logits = classifier_head(feats)

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(torch.argmax(logits, dim=1).cpu().item())

    return {
        'class_id': pred_class,
        'class_name': CLASS_NAMES[pred_class],
        'class_name_zh': CLASS_NAMES_ZH[pred_class],
        'confidence': float(probs[pred_class]),
        'probabilities': {
            CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
        }
    }


@torch.no_grad()
def predict_folder(model, preprocess, classifier_head, folder_path, device='cuda',
                   extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Predict classes for all images in a folder."""
    folder = Path(folder_path)
    results = []

    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))

    image_files = sorted(image_files)

    for img_path in image_files:
        result = predict_image(model, preprocess, classifier_head, str(img_path), device)
        result['file'] = img_path.name
        results.append(result)

    return results


@torch.no_grad()
def evaluate_test_set(model, preprocess, classifier_head, data_root, test_file, device='cuda'):
    """Evaluate model on test set and compute full metrics."""
    # Build tokenized prompts for dataset
    class_names_en = ['human face', 'license plate', 'employee badge']
    _, _, tokenizer = None, None, open_clip.get_tokenizer('ViT-B-32-quickgelu')
    # We need tokenized prompts for the dataset
    tokenized_dict = {}
    for class_id, name in enumerate(class_names_en):
        prompts = [t.format(name) for t in PROMPT_TEMPLATES]
        tokenized = tokenizer(prompts)
        tokenized_dict[str(class_id)] = {'all': tokenized}

    dataset = FewShotDataset(data_root, test_file, preprocess, tokenized_dict)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    logit_scale = torch.tensor([4.60517]).to(device)  # ln(1/0.01)
    acc, cm = validate(model, dataloader, classifier_head, logit_scale, device)

    metrics = compute_metrics(cm)
    metrics['accuracy'] = acc

    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description='Test 3-class few-shot model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'single', 'folder', 'evaluate'])
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--folder', type=str, help='Folder of images')
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model, preprocess, tokenizer, classifier_head = load_model(
        args.model_cfg, args.model_path, device
    )

    if args.mode == 'evaluate':
        print(f"Evaluating on test set: {args.data_root}/{args.test_file}")

        try:
            class_names_en = ['human face', 'license plate', 'employee badge']
            tokenized_dict = {}
            for class_id, name in enumerate(class_names_en):
                prompts = [t.format(name) for t in PROMPT_TEMPLATES]
                tokenized = tokenizer(prompts)
                tokenized_dict[str(class_id)] = {'all': tokenized}

            dataset = FewShotDataset(args.data_root, args.test_file, preprocess, tokenized_dict)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

            logit_scale = torch.tensor([4.60517]).to(device)
            acc, cm = validate(model, dataloader, classifier_head, logit_scale, device)
            metrics = compute_metrics(cm)

            print(f"\n{'='*50}")
            print(f"Test Accuracy: {acc:.2f}%")
            print(f"{'='*50}")
            for i, (en, zh) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_ZH)):
                print(f"  {en} ({zh}):")
                print(f"    Recall:    {metrics[f'class_{i}_recall']:.3f}")
                print(f"    Precision: {metrics[f'class_{i}_precision']:.3f}")
                print(f"    F1:        {metrics[f'class_{i}_f1']:.3f}")
            print(f"\nConfusion Matrix (rows=true, cols=pred):")
            print(f"            face  lic.plate  badge")
            for i, name in enumerate(CLASS_NAMES):
                print(f"  {name:10s}  {cm[i,0]:4d}  {cm[i,1]:9d}  {cm[i,2]:5d}")

            if args.output:
                result = {'accuracy': acc, 'cm': cm.tolist(),
                          'metrics': {k: v for k, v in metrics.items()}}
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {args.output}")

        except ImportError as e:
            print(f"Error importing train module: {e}")
            print("Make sure train.py is accessible.")

    elif args.mode == 'single':
        if not args.image:
            print("Please provide --image path")
            return
        result = predict_image(model, preprocess, classifier_head, args.image, device)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {result['class_name']} ({result['class_name_zh']})")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Probabilities:")
        for cls_name, prob in result['probabilities'].items():
            print(f"  {cls_name}: {prob:.3f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

    elif args.mode == 'folder':
        if not args.folder:
            print("Please provide --folder path")
            return
        results = predict_folder(model, preprocess, classifier_head, args.folder, device)

        print(f"\nResults for {len(results)} images in: {args.folder}")
        print(f"{'Filename':30s} {'Predicted':15s} {'Confidence':>10s}")
        print("-" * 57)
        for r in results:
            print(f"{r['file']:30s} {r['class_name']:15s} {r['confidence']:10.3f}")

        # Summary
        counts = {name: 0 for name in CLASS_NAMES}
        for r in results:
            counts[r['class_name']] += 1
        print(f"\nSummary:")
        for name in CLASS_NAMES:
            print(f"  {name}: {counts[name]} images")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.mode == 'interactive':
        print("\n" + "=" * 50)
        print("Interactive Inference Mode")
        print("Enter image path to classify, or 'q' to quit")
        print("=" * 50)

        while True:
            img_path = input("\nImage path: ").strip()
            if img_path.lower() == 'q':
                break
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue

            result = predict_image(model, preprocess, classifier_head, img_path, device)
            print(f"  -> {result['class_name']} ({result['class_name_zh']}) "
                  f"[confidence: {result['confidence']:.3f}]")
            for cls_name, prob in result['probabilities'].items():
                bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
                print(f"     {cls_name:15s} |{bar}| {prob:.3f}")


if __name__ == '__main__':
    main()
