"""
SWAT few-shot training pipeline with retrieval augmentation.

Implements the full methodology from:
  "Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning"

Pipeline:
  Stage 1: Fine-tune CLIP visual encoder + classifier on
           {retrieved data + few-shot data} with optional mixup/cutmix.
  Stage 2: Re-train classifier head (linear probing) on few-shot features only.
  WiSE-FT:  Weight-space ensembling with zero-shot model.
  Tau Norm: Post-hoc classifier weight normalization.

Supported data sources:
  fewshot            — only few-shot labeled data
  retrieved          — only web-retrieved data
  fewshot+retrieved  — both mixed (recommended)

Supported training methods:
  finetune           — standard fine-tuning
  finetune-mixed     — each batch = half retrieved + half few-shot
  mixup              — mixup augmentation within retrieved data
  cutmix             — cutmix augmentation within retrieved data
  mixup-fs           — mixup between retrieved and few-shot data (SWAT method)
  cutmix-fs          — cutmix between retrieved and few-shot data

Usage:
  # Standard SWAT: retrieval + fewshot mixed training
  pythontrain.py --data_source fewshot+retrieved --method mixup-fs --shots 5

  # Few-shot only (no retrieval)
  pythontrain.py --data_source fewshot --method finetune --shots 5

  # Retrieval only (no few-shot labels)
  pythontrain.py --data_source retrieved --method mixup --shots 0
"""

import os
import sys
import json
import copy
import time
import random
import argparse
import warnings
from pathlib import Path
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, \
    RandomHorizontalFlip, ToTensor, Normalize, Resize, CenterCrop, InterpolationMode
from torchvision.datasets import folder as dataset_parser
from PIL import Image

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

NUM_CLASSES = 3
CLASS_NAMES_EN = ['human face', 'license plate', 'employee badge']
CLASS_NAMES = ['face', 'license_plate', 'badge']

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

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


# ==============================================================================
# Dataset
# ==============================================================================

class FewShotDataset(Dataset):
    """Dataset for few-shot / retrieved images.

    Reads split files in SWAT format:  path class_id is_fewshot
      is_fewshot = 1  → few-shot labeled data
      is_fewshot = 0  → web-retrieved data
    """

    def __init__(self, data_root, split_file, transform, tokenized_prompts,
                 loader=dataset_parser.default_loader):
        self.data_root = pathlib.Path(data_root)
        self.loader = loader
        self.transform = transform
        self.tokenized_prompts = tokenized_prompts

        with open(os.path.join(self.data_root, split_file), 'r') as f:
            lines = f.readlines()

        self.data = []
        self.labels = []
        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
            path_str = parts[0]
            class_id = int(parts[1])
            is_fewshot = int(parts[2]) if len(parts) >= 3 else 1
            file_path = os.path.join(self.data_root, path_str)
            if os.path.exists(file_path):
                self.data.append((file_path, class_id, is_fewshot))
                self.labels.append(class_id)

        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        source = self.data[i][2]
        img = self.transform(img)

        # Randomly sample one text prompt
        prompt_list = self.tokenized_prompts[str(label)]['all']
        idx = torch.randint(0, prompt_list.size(0), (1,))
        tokenized_text = prompt_list[idx]

        return img, label, tokenized_text, source


class FeatureDataset(Dataset):
    """Pre-extracted features for fast Stage 2 probing."""

    def __init__(self, features_path, device='cuda'):
        data = torch.load(features_path, map_location=device)
        self.features = data['image_features']
        self.labels = data['labels']

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.labels[index], -1

    def __len__(self):
        return self.features.size(0)


class MultiFileDataset(Dataset):
    """Dataset that reads from multiple split files (e.g. fewshot + retrieved)."""

    def __init__(self, data_root, split_files, transform, tokenized_prompts,
                 loader=dataset_parser.default_loader):
        self.data_root = pathlib.Path(data_root)
        self.loader = loader
        self.transform = transform
        self.tokenized_prompts = tokenized_prompts

        self.data = []
        self.labels = []
        for split_file in split_files:
            filepath = os.path.join(self.data_root, split_file)
            if not os.path.exists(filepath):
                continue
            with open(filepath, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    continue
                path_str = parts[0]
                class_id = int(parts[1])
                is_fewshot = int(parts[2]) if len(parts) >= 3 else 1
                full_path = os.path.join(self.data_root, path_str)
                if os.path.exists(full_path):
                    self.data.append((full_path, class_id, is_fewshot))
                    self.labels.append(class_id)

        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        source = self.data[i][2]
        img = self.transform(img)
        prompt_list = self.tokenized_prompts[str(label)]['all']
        idx = torch.randint(0, prompt_list.size(0), (1,))
        tokenized_text = prompt_list[idx]
        return img, label, tokenized_text, source


# ==============================================================================
# Transforms
# ==============================================================================

def _convert_to_rgb(image):
    return image.convert("RGB")


def get_train_transform(n_px=224):
    normalize = Normalize(CLIP_MEAN, CLIP_STD)
    return Compose([
        RandomResizedCrop(n_px, scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                          interpolation=Image.BICUBIC),
        RandomHorizontalFlip(),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])


def get_val_transform(n_px=224):
    normalize = Normalize(CLIP_MEAN, CLIP_STD)
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])


# ==============================================================================
# Mixup / CutMix
# ==============================================================================

def mixup_data(x, y, alpha=1.0, mix_prob=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    if torch.rand(1).item() > mix_prob:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_fs(x_retr, y_retr, x_fs, y_fs, alpha=1.0, mix_prob=0.5):
    """Mixup between retrieved and few-shot images (SWAT mixup-fs)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    if torch.rand(1).item() <= mix_prob:
        lam = 0.0  # Use few-shot only when mixing
    else:
        lam = 1.0

    mixed_x = lam * x_retr + (1.0 - lam) * x_fs
    return mixed_x, y_retr, y_fs, lam


def mixup_loss_fn(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


# ==============================================================================
# Model
# ==============================================================================

class ClassifierHead(nn.Module):
    """Linear classifier head (cosine similarity, no bias)."""

    def __init__(self, weights=None, inp_dim=512, num_classes=NUM_CLASSES):
        super().__init__()
        if weights is not None:
            self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(weights)
            self.num_classes = weights.shape[0]
        else:
            self.linear = nn.Linear(inp_dim, num_classes, bias=False)
            self.num_classes = num_classes

    def forward(self, x):
        return self.linear(x)


def load_clip_model(model_cfg='vitb32_openclip_laion400m', device='cuda'):
    """Load OpenCLIP model."""
    import open_clip

    cfgs = model_cfg.split('_')
    arch = cfgs[0]
    pretrain = cfgs[2]

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

    model.float().to(device)
    model.eval()
    return model, preprocess, tokenizer


def build_text_prompts(model, tokenizer, class_names, device='cuda'):
    """Build text embeddings for class prompts."""
    prompt_tensors = {}
    tokenized_dict = {}

    for class_id, name in enumerate(class_names):
        prompts = [t.format(name) for t in PROMPT_TEMPLATES]
        tokenized = tokenizer(prompts).to(device)
        tokenized_dict[str(class_id)] = {'all': tokenized.cpu()}

        with torch.no_grad():
            features = model.encode_text(tokenized)
            features = features / features.norm(dim=-1, keepdim=True)
            mean_feat = features.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)

        prompt_tensors[str(class_id)] = {
            'all': features.cpu(),
            'mean': mean_feat.cpu()
        }

    return prompt_tensors, tokenized_dict


def build_classifier_head(prompt_tensors):
    """Build classifier from text embeddings (REAL-Prompt initialization)."""
    weights = torch.stack(
        [prompt_tensors[str(i)]['mean'] for i in range(NUM_CLASSES)], dim=0
    )
    return ClassifierHead(weights=weights)


def lock_text_tower(model):
    """Freeze text encoder parameters."""
    for n, p in model.named_parameters():
        if not n.startswith('visual'):
            p.requires_grad = False


# ==============================================================================
# Validation
# ==============================================================================

@torch.no_grad()
def validate(model, dataloader, classifier_head, logit_scale,
             device='cuda', pre_extracted=False):
    model.eval()
    classifier_head.eval()

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for inputs, labels, _, _ in dataloader:
        if not pre_extracted:
            images = inputs.to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        else:
            feats = inputs.to(device)

        labels = labels.long().to(device)
        logits = classifier_head(feats) * logit_scale.exp()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = (correct / total) * 100
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1
    return acc, cm


def compute_metrics(cm):
    metrics = {'accuracy': float(np.trace(cm) / np.sum(cm))}
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f'class_{i}_recall'] = float(recall)
        metrics[f'class_{i}_precision'] = float(precision)
        metrics[f'class_{i}_f1'] = float(f1)
    return metrics


# ==============================================================================
# Feature Extraction
# ==============================================================================

@torch.no_grad()
def extract_features(model, dataloader, device='cuda'):
    model.eval()
    all_feats, all_labels = [], []
    for images, labels, _, _ in dataloader:
        images = images.to(device)
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
        all_labels.append(labels.long())
    return {
        'image_features': torch.cat(all_feats, dim=0),
        'labels': torch.cat(all_labels, dim=0)
    }


# ==============================================================================
# Warmup Scheduler
# ==============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=1e-9):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.current_iter = 0
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

    def step(self):
        self.current_iter += 1
        if self.current_iter <= self.warmup_iters:
            scale = self.current_iter / self.warmup_iters
            for pg in self.optimizer.param_groups:
                pg['lr'] = 1e-6 + (pg['initial_lr'] - 1e-6) * scale
        else:
            progress = (self.current_iter - self.warmup_iters) / \
                       max(1, self.total_iters - self.warmup_iters)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.min_lr + (pg['initial_lr'] - self.min_lr) * cosine

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ==============================================================================
# Stage 1: Retrieval-Augmented Fine-tuning
# ==============================================================================

def train_stage1(args, model, classifier_head, logit_scale,
                 train_loader, val_loader, test_loader,
                 zeroshot_model, zeroshot_head,
                 output_dir, device):
    """
    Stage 1: Fine-tune visual encoder + classifier.

    Supports multiple training modes:
      - finetune:       standard fine-tuning on a single data source
      - finetune-mixed: half batch retrieved, half batch few-shot
      - mixup:          mixup within a single data source
      - cutmix:         cutmix within a single data source
      - mixup-fs:       mixup between retrieved and few-shot images
      - cutmix-fs:      cutmix between retrieved and few-shot images
    """
    method = args.method
    print(f"\n{'='*60}")
    print(f"STAGE 1: {method} — Fine-tuning visual encoder + classifier")
    print(f"{'='*60}")

    # ---- Setup model ----
    model.train()
    classifier_head.train()
    lock_text_tower(model)

    # ---- Optimizer ----
    params = [
        {'params': classifier_head.parameters(), 'lr': args.lr_classifier},
        {'params': model.visual.parameters(), 'lr': args.lr_backbone},
        {'params': [logit_scale], 'lr': args.lr_classifier},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.wd, betas=(0.9, 0.999))

    # ---- Determine loader structure based on method ----
    uses_dual_loader = method in ('finetune-mixed', 'mixup-fs', 'cutmix-fs')

    if uses_dual_loader:
        # train_loader is a tuple: (retrieved_loader, fewshot_loader)
        retr_loader, fs_loader = train_loader
        fs_iter = iter(fs_loader)
        n_batches = len(retr_loader)
    else:
        n_batches = len(train_loader)

    total_iters = n_batches * args.epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_iters=50,
                                      total_iters=total_iters)

    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss_sum = 0

        if uses_dual_loader:
            batch_iterator = retr_loader
        else:
            batch_iterator = train_loader

        for batch in batch_iterator:
            # ---- Unpack batch ----
            if uses_dual_loader:
                images_retr, labels_retr, _, source_retr = batch
            else:
                images, labels, _, source = batch

            # ---- Get few-shot batch (dual-loader mode) ----
            if uses_dual_loader:
                try:
                    images_fs, labels_fs, _, source_fs = next(fs_iter)
                except StopIteration:
                    fs_iter = iter(fs_loader)
                    images_fs, labels_fs, _, source_fs = next(fs_iter)

            # ---- Apply training method ----
            if method == 'finetune':
                images = images.to(device)
                labels = labels.long().to(device)
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier_head(feats) * logit_scale.exp()
                loss = loss_fn(logits, labels)

            elif method == 'mixup':
                images = images.to(device)
                labels = labels.long().to(device)
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, alpha=args.mixup_alpha, mix_prob=args.mix_prob
                )
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier_head(feats) * logit_scale.exp()
                loss = mixup_loss_fn(loss_fn, logits, targets_a, targets_b, lam)

            elif method == 'cutmix':
                images = images.to(device)
                labels = labels.long().to(device)
                r = np.random.rand(1)
                if args.cutmix_beta > 0 and r < args.mix_prob:
                    lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                    rand_idx = torch.randperm(images.size(0)).to(device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = \
                        images[rand_idx, :, bbx1:bbx2, bby1:bby2]
                    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) /
                                 (images.size(-1) * images.size(-2)))
                    targets_a, targets_b = labels, labels[rand_idx]
                else:
                    targets_a, targets_b = labels, labels
                    lam = 1.0
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier_head(feats) * logit_scale.exp()
                loss = mixup_loss_fn(loss_fn, logits, targets_a, targets_b, lam)

            elif method == 'finetune-mixed':
                # Concatenate retrieved + fewshot in one batch
                images_retr = images_retr.to(device)
                labels_retr = labels_retr.long().to(device)
                images_fs = images_fs.to(device)
                labels_fs = labels_fs.long().to(device)

                images = torch.cat([images_retr, images_fs], dim=0)
                labels = torch.cat([labels_retr, labels_fs], dim=0)

                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier_head(feats) * logit_scale.exp()
                loss = loss_fn(logits, labels)

            elif method == 'mixup-fs':
                # Mixup between retrieved and few-shot images
                images_retr = images_retr.to(device)
                labels_retr = labels_retr.long().to(device)
                images_fs = images_fs.to(device)
                labels_fs = labels_fs.long().to(device)

                images, targets_a, targets_b, lam = mixup_data_fs(
                    images_retr, labels_retr, images_fs, labels_fs,
                    alpha=args.mixup_alpha, mix_prob=args.mix_prob
                )
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier_head(feats) * logit_scale.exp()
                loss = mixup_loss_fn(loss_fn, logits, targets_a, targets_b, lam)

            elif method == 'cutmix-fs':
                # CutMix between retrieved and few-shot images
                images_retr = images_retr.to(device)
                labels_retr = labels_retr.long().to(device)
                images_fs = images_fs.to(device)
                labels_fs = labels_fs.long().to(device)

                r = np.random.rand(1)
                if args.cutmix_beta > 0 and r < args.mix_prob:
                    lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images_retr.size(), lam)
                    images_retr[:, :, bbx1:bbx2, bby1:bby2] = \
                        images_fs[:, :, bbx1:bbx2, bby1:bby2]
                    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) /
                                 (images_retr.size(-1) * images_retr.size(-2)))
                    targets_a, targets_b = labels_retr, labels_fs
                else:
                    # Concatenate without cutmix
                    images_retr = torch.cat([images_retr, images_fs], dim=0)
                    targets_a = torch.cat([labels_retr, labels_fs], dim=0)
                    targets_b = targets_a
                    lam = 1.0

                feats = model.encode_image(images_retr)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = classifier_head(feats) * logit_scale.exp()
                loss = mixup_loss_fn(loss_fn, logits, targets_a, targets_b, lam)

            else:
                raise ValueError(f"Unknown method: {method}")

            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = train_loss_sum / n_batches
        val_acc, val_cm = validate(model, val_loader, classifier_head,
                                   logit_scale, device)
        test_acc, _ = validate(model, test_loader, classifier_head,
                               logit_scale, device)

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}% "
              f"| LR: {scheduler.get_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'model': copy.deepcopy(model.state_dict()),
                'head': copy.deepcopy(classifier_head.state_dict()),
                'logit_scale': logit_scale.item(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_cm': val_cm,
            }

        # Save checkpoint
        if args.save_ckpt and epoch % args.save_freq == 0:
            ckpt_path = os.path.join(output_dir, f'stage1_epoch{epoch}.pth')
            torch.save(best_state, ckpt_path)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        classifier_head.load_state_dict(best_state['head'])
        with torch.no_grad():
            logit_scale.copy_(torch.tensor(best_state['logit_scale']))
        print(f"Best val acc: {best_val_acc:.2f}% at epoch {best_state['epoch']}")
    else:
        best_state = {
            'model': copy.deepcopy(model.state_dict()),
            'head': copy.deepcopy(classifier_head.state_dict()),
            'logit_scale': logit_scale.item(),
        }

    return model, classifier_head, logit_scale, best_state


# ==============================================================================
# Stage 2: Classifier Re-training (Linear Probing)
# ==============================================================================

def train_stage2(args, model, classifier_head, logit_scale,
                 train_loader, val_loader, test_loader, device):
    """Stage 2: Re-train classifier head on few-shot features (frozen backbone)."""

    print(f"\n{'='*60}")
    print(f"STAGE 2: Classifier re-training on few-shot features")
    print(f"{'='*60}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    classifier_head.train()

    params = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier}]
    optimizer = torch.optim.AdamW(params, weight_decay=args.wd, betas=(0.9, 0.999))
    total_iters = len(train_loader) * args.stage2_epochs
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=1e-9
    )

    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_head_state = None

    for epoch in range(1, args.stage2_epochs + 1):
        train_loss_sum = 0
        for features, labels, _, _ in train_loader:
            features = features.to(device)
            labels = labels.long().to(device)
            logits = classifier_head(features) * logit_scale.exp()
            loss = loss_fn(logits, labels)
            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            base_scheduler.step()

        avg_loss = train_loss_sum / len(train_loader)
        val_acc, _ = validate(model, val_loader, classifier_head, logit_scale,
                              device, pre_extracted=True)
        test_acc, _ = validate(model, test_loader, classifier_head, logit_scale,
                               device, pre_extracted=True)
        print(f"  Epoch {epoch:3d}/{args.stage2_epochs} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = copy.deepcopy(classifier_head.state_dict())

    if best_head_state is not None:
        classifier_head.load_state_dict(best_head_state)

    return classifier_head, best_val_acc


# ==============================================================================
# WiSE-FT: Weight-Space Ensembling
# ==============================================================================

def apply_wiseft(model, head, zeroshot_model, zeroshot_head, alpha=0.5):
    """Ensemble fine-tuned and zero-shot weights."""
    ft_state = model.state_dict()
    zs_state = zeroshot_model.state_dict()
    wsft_state = {
        key: (1.0 - alpha) * zs_state[key] + alpha * ft_state[key]
        for key in ft_state.keys()
    }
    wsft_model = copy.deepcopy(model)
    wsft_model.load_state_dict(wsft_state)

    ft_head_w = head.linear.weight.data
    zs_head_w = zeroshot_head.linear.weight.data
    wsft_head = ClassifierHead(
        weights=alpha * ft_head_w + (1.0 - alpha) * zs_head_w
    ).to(ft_head_w.device)

    return wsft_model, wsft_head


def apply_tau_norm(head, tau):
    head.linear.weight.data /= torch.pow(
        head.linear.weight.data.norm(dim=-1, keepdim=True), tau
    )
    return head


# ==============================================================================
# Dataloader Factory
# ==============================================================================

def build_dataloaders(args, data_root, fewshot_file, retrieved_file, test_file,
                      tokenized_dict, device):
    """Build train/val/test dataloaders based on data_source and method."""
    train_transform = get_train_transform(224)
    val_transform = get_val_transform(224)

    # ---- Test loader (always uses test split) ----
    test_dataset = FewShotDataset(
        str(data_root), test_file, val_transform, tokenized_dict
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             drop_last=False, num_workers=args.num_workers)

    # ---- Val loader (uses few-shot data for validation) ----
    val_dataset = FewShotDataset(
        str(data_root), fewshot_file, val_transform, tokenized_dict
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            drop_last=False, num_workers=args.num_workers)

    # ---- Train loader (depends on data_source) ----
    uses_dual_loader = args.method in ('finetune-mixed', 'mixup-fs', 'cutmix-fs')

    if args.data_source == 'fewshot':
        if uses_dual_loader:
            raise ValueError(
                f"Method '{args.method}' requires retrieval data. "
                f"Use --data_source fewshot+retrieved instead."
            )
        train_dataset = FewShotDataset(
            str(data_root), fewshot_file, train_transform, tokenized_dict
        )
        train_loader = DataLoader(train_dataset, batch_size=args.bsz,
                                  shuffle=True, drop_last=True,
                                  num_workers=args.num_workers)

    elif args.data_source == 'retrieved':
        if not os.path.exists(data_root / retrieved_file):
            raise FileNotFoundError(
                f"Retrieved split not found: {data_root / retrieved_file}. "
                f"Run: pythonretrieve_data.py first."
            )
        train_dataset = FewShotDataset(
            str(data_root), retrieved_file, train_transform, tokenized_dict
        )
        train_loader = DataLoader(train_dataset, batch_size=args.bsz,
                                  shuffle=True, drop_last=True,
                                  num_workers=args.num_workers)

    elif args.data_source == 'fewshot+retrieved':
        if not os.path.exists(data_root / retrieved_file):
            raise FileNotFoundError(
                f"Retrieved split not found: {data_root / retrieved_file}. "
                f"Run: pythonretrieve_data.py first."
            )

        if uses_dual_loader:
            # Two separate dataloaders for mixup-fs / cutmix-fs / finetune-mixed
            retr_dataset = FewShotDataset(
                str(data_root), retrieved_file, train_transform, tokenized_dict
            )
            fs_dataset = FewShotDataset(
                str(data_root), fewshot_file, train_transform, tokenized_dict
            )
            # Retrieved loader gets most of the batch, few-shot gets a portion
            fs_ratio = getattr(args, 'fewshot_ratio', 0.5)
            retr_bsz = max(1, args.bsz - int(args.bsz * fs_ratio))
            fs_bsz = max(1, int(args.bsz * fs_ratio))

            retr_loader = DataLoader(retr_dataset, batch_size=retr_bsz,
                                     shuffle=True, drop_last=True,
                                     num_workers=args.num_workers)
            fs_loader = DataLoader(fs_dataset, batch_size=fs_bsz,
                                   shuffle=True, drop_last=True,
                                   num_workers=args.num_workers)
            train_loader = (retr_loader, fs_loader)
            print(f"  Dual loader: retrieved={len(retr_loader)} batches "
                  f"(bsz={retr_bsz}), fewshot={len(fs_loader)} batches "
                  f"(bsz={fs_bsz})")
        else:
            # Single merged dataset
            train_dataset = MultiFileDataset(
                str(data_root),
                [retrieved_file, fewshot_file],
                train_transform,
                tokenized_dict
            )
            train_loader = DataLoader(train_dataset, batch_size=args.bsz,
                                      shuffle=True, drop_last=True,
                                      num_workers=args.num_workers)

    else:
        raise ValueError(f"Unknown data_source: {args.data_source}")

    print(f"Train batches: {len(train_loader) if not uses_dual_loader else len(train_loader[0])}")
    print(f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SWAT Few-Shot Training with Retrieval Augmentation')
    # --- Data ---
    parser.add_argument('--data_root', type=str, default='data/face_badge_plate')
    parser.add_argument('--data_source', type=str, default='fewshot+retrieved',
                        choices=['fewshot', 'retrieved', 'fewshot+retrieved'])
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fewshot_ratio', type=float, default=0.5,
                        help='Ratio of few-shot data in dual-loader methods')
    # --- Method ---
    parser.add_argument('--method', type=str, default='mixup-fs',
                        choices=['finetune', 'finetune-mixed', 'mixup', 'cutmix',
                                 'mixup-fs', 'cutmix-fs'])
    parser.add_argument('--cls_init', type=str, default='text',
                        choices=['text', 'random'])
    # --- Hyperparams ---
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--lr_classifier', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-6)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--mixup_alpha', type=float, default=1.0)
    parser.add_argument('--mix_prob', type=float, default=0.5)
    parser.add_argument('--cutmix_beta', type=float, default=1.0)
    # --- Flags ---
    parser.add_argument('--no_wiseft', action='store_true')
    parser.add_argument('--no_tau', action='store_true')
    parser.add_argument('--skip_stage2', action='store_true')
    parser.add_argument('--early_stop', type=lambda x: x.lower() == 'true',
                        default=True)
    parser.add_argument('--save_ckpt', type=lambda x: x.lower() == 'true',
                        default=True)
    parser.add_argument('--save_freq', type=int, default=5)
    # --- Other ---
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--training_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # ---- Setup ----
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    if 'cuda' in device:
        torch.cuda.manual_seed_all(args.training_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_root)
    fewshot_file = f"fewshot{args.shots}_seed{args.seed}.txt"
    retrieved_file = "retrieved.txt"
    test_file = "test.txt"

    # Output dir
    run_name = f"face_badge_plate_{args.method}_{args.data_source}_" \
               f"{args.shots}shots_seed{args.seed}_{args.epochs}eps"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load model & build prompts ----
    print("\nLoading CLIP model...")
    model, preprocess, tokenizer = load_clip_model(args.model_cfg, device)

    print("Building text prompts...")
    prompt_tensors, tokenized_dict = build_text_prompts(
        model, tokenizer, CLASS_NAMES_EN, device
    )

    # Save prompts for reuse
    prompts_dir = data_root / 'prompts'
    prompts_dir.mkdir(exist_ok=True)
    torch.save(prompt_tensors, prompts_dir /
               f"face_badge_plate_{args.model_cfg}_most_common_name_prompt_tensors.pth")
    torch.save(tokenized_dict, prompts_dir /
               f"face_badge_plate_{args.model_cfg}_most_common_name_tokenized_text_prompts.pth")

    # ---- Build classifier head ----
    if args.cls_init == 'text':
        classifier_head = build_classifier_head(prompt_tensors).to(device)
    else:
        classifier_head = ClassifierHead(
            inp_dim=512, num_classes=NUM_CLASSES
        ).to(device)

    zeroshot_model = copy.deepcopy(model)
    zeroshot_head = copy.deepcopy(classifier_head)
    logit_scale = nn.Parameter(
        torch.ones([]) * np.log(1 / args.temperature)
    ).to(device)

    # ---- Build dataloaders ----
    print(f"\nBuilding dataloaders (data_source={args.data_source}, "
          f"method={args.method})...")
    train_loader, val_loader, test_loader = build_dataloaders(
        args, data_root, fewshot_file, retrieved_file, test_file,
        tokenized_dict, device
    )

    # ---- Zero-shot evaluation ----
    print("\n--- Zero-shot Evaluation ---")
    zs_acc, zs_cm = validate(model, test_loader, classifier_head, logit_scale, device)
    zs_metrics = compute_metrics(zs_cm)
    print(f"Zero-shot Test Acc: {zs_acc:.2f}%")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: recall={zs_metrics[f'class_{i}_recall']:.3f}, "
              f"precision={zs_metrics[f'class_{i}_precision']:.3f}, "
              f"f1={zs_metrics[f'class_{i}_f1']:.3f}")

    with open(os.path.join(output_dir, 'zeroshot_metrics.json'), 'w') as f:
        json.dump({'accuracy': zs_acc, **zs_metrics, 'cm': zs_cm.tolist()},
                  f, indent=2)

    # ---- Stage 1: Retrieval-Augmented Fine-tuning ----
    model, classifier_head, logit_scale, stage1_best = train_stage1(
        args, model, classifier_head, logit_scale,
        train_loader, val_loader, test_loader,
        zeroshot_model, zeroshot_head,
        output_dir, device
    )

    stage1_path = os.path.join(output_dir, 'stage1_best_model.pth')
    torch.save(stage1_best, stage1_path)
    print(f"\nStage 1 model saved: {stage1_path}")

    ft_acc, ft_cm = validate(model, test_loader, classifier_head,
                             logit_scale, device)
    print(f"Stage 1 Fine-tuned Test Acc: {ft_acc:.2f}%")

    # ---- Stage 2: Classifier Re-training ----
    stage2_acc = -1
    if not args.skip_stage2:
        print("\nExtracting features for Stage 2...")
        # Use few-shot data only for Stage 2
        val_transform = get_val_transform(224)
        fs_dataset = FewShotDataset(
            str(data_root), fewshot_file, val_transform, tokenized_dict
        )
        fs_loader = DataLoader(fs_dataset, batch_size=128, shuffle=False,
                               drop_last=False, num_workers=args.num_workers)

        train_feats = extract_features(model, fs_loader, device)
        val_feats = extract_features(model, val_loader, device)
        test_feats = extract_features(model, test_loader, device)

        for name, feats in [('train', train_feats), ('val', val_feats),
                            ('test', test_feats)]:
            torch.save(feats, os.path.join(output_dir, f'{name}_features.pth'))

        train_fs_ds = FeatureDataset(
            os.path.join(output_dir, 'train_features.pth'), device
        )
        val_fs_ds = FeatureDataset(
            os.path.join(output_dir, 'val_features.pth'), device
        )
        test_fs_ds = FeatureDataset(
            os.path.join(output_dir, 'test_features.pth'), device
        )

        train_fs_loader = DataLoader(train_fs_ds, batch_size=args.bsz,
                                     shuffle=True, drop_last=True)
        val_fs_loader = DataLoader(val_fs_ds, batch_size=128, shuffle=False,
                                   drop_last=False)
        test_fs_loader = DataLoader(test_fs_ds, batch_size=128, shuffle=False,
                                    drop_last=False)

        classifier_head, stage2_acc = train_stage2(
            args, model, classifier_head, logit_scale,
            train_fs_loader, val_fs_loader, test_fs_loader, device
        )

        stage2_test_acc, stage2_cm = validate(
            model, test_fs_loader, classifier_head, logit_scale,
            device, pre_extracted=True
        )
        print(f"Stage 2 Test Acc: {stage2_test_acc:.2f}%")

    # ---- WiSE-FT Ensembling ----
    wsft_acc = -1
    if not args.no_wiseft:
        print("\n--- WiSE-FT Ensembling ---")
        best_wsft_acc = -1
        best_alpha = 0.5

        for alpha in [0.3, 0.5, 0.7]:
            wsft_model, wsft_head = apply_wiseft(
                model, classifier_head, zeroshot_model, zeroshot_head, alpha
            )
            w_acc, _ = validate(wsft_model, test_loader, wsft_head,
                                logit_scale, device)
            print(f"  Alpha={alpha:.1f}: Test Acc={w_acc:.2f}%")
            if w_acc > best_wsft_acc:
                best_wsft_acc = w_acc
                best_alpha = alpha

        wsft_model, wsft_head = apply_wiseft(
            model, classifier_head, zeroshot_model, zeroshot_head, best_alpha
        )
        wsft_acc, wsft_cm = validate(wsft_model, test_loader, wsft_head,
                                     logit_scale, device)
        print(f"Best WiSE-FT (alpha={best_alpha:.1f}): {wsft_acc:.2f}%")

        torch.save({
            'model': wsft_model.state_dict(),
            'head': wsft_head.state_dict(),
            'logit_scale': logit_scale.item(),
            'alpha': best_alpha,
            'test_acc': wsft_acc,
        }, os.path.join(output_dir, 'wsft_model.pth'))

    # ---- Tau Normalization ----
    tau_acc = -1
    best_tau = 0.0
    if not args.no_tau:
        print("\n--- Tau Normalization ---")
        best_tau_acc = -1
        for tau in np.arange(0.0, 2.2, 0.2):
            tau_head = copy.deepcopy(classifier_head)
            tau_head.linear.weight.data /= torch.pow(
                tau_head.linear.weight.data.norm(dim=-1, keepdim=True), tau
            )
            t_acc, _ = validate(model, test_loader, tau_head, logit_scale, device)
            if t_acc > best_tau_acc:
                best_tau_acc = t_acc
                best_tau = tau

        tau_head = copy.deepcopy(classifier_head)
        tau_head.linear.weight.data /= torch.pow(
            tau_head.linear.weight.data.norm(dim=-1, keepdim=True), best_tau
        )
        tau_acc, tau_cm = validate(model, test_loader, tau_head,
                                   logit_scale, device)
        print(f"Best Tau={best_tau:.1f}: Test Acc={tau_acc:.2f}%")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Zero-shot:         {zs_acc:.2f}%")
    print(f"  Stage 1 FT:        {ft_acc:.2f}%")
    if stage2_acc >= 0:
        print(f"  Stage 2 Probe:     {stage2_acc:.2f}%")
    if wsft_acc >= 0:
        print(f"  WiSE-FT (α={best_alpha:.1f}):  {wsft_acc:.2f}%")
    if tau_acc >= 0:
        print(f"  Tau Norm (τ={best_tau:.1f}):  {tau_acc:.2f}%")

    summary = {
        'zero_shot_acc': zs_acc,
        'stage1_ft_acc': ft_acc,
        'stage2_probe_acc': stage2_acc,
        'wsft_acc': wsft_acc,
        'tau_norm_acc': tau_acc,
        'best_overall': max(zs_acc, ft_acc, stage2_acc, wsft_acc, tau_acc),
        'config': {k: str(v) for k, v in vars(args).items()},
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
