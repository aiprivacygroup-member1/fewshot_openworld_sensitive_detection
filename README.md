# 基于开放世界隐私发现的少样本隐私目标自适应遮蔽

本项目实现了一套面向**开放世界隐私发现**与**少样本敏感目标保护**的完整流水线。核心思路是：在仅用极少标注样本（few-shot）训练模型的前提下，自动发现监控视频中模型"不认识"、但可能同样属于隐私敏感的新目标，并对已识别的人脸、车牌、工牌等敏感目标做差异化的自适应遮蔽（模糊 / 马赛克 / SimSwap 换脸）。

整个系统由五个功能模块组成，数据沿"分类 → 自动标注 → 检测 → 发现 → 保护"的链路流转。

## 整体框架

```
                ┌──────────────────────────────────────────────────────────┐
                │                    少样本知识迁移（分类）                   │
                │  classify/  SWAT 两阶段检索增强微调（CLIP ViT-B/32 骨干）  │
                └───────────────────────────┬──────────────────────────────┘
                                            │ 训练好的分类器
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │                    弱监督自动标注                          │
                │  auto_label/  共享特征图 + 多尺度滑动窗口 + RoIAlign       │
                └───────────────────────────┬──────────────────────────────┘
                                            │ YOLO/COCO/VOC 标注
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │                    少样本目标检测                          │
                │  object_detection/  两阶段 YOLOv8 训练 + 模型融合           │
                └──────────────┬───────────────────────┬────────────────────┘
                               │                       │
                               ▼                       ▼
                ┌──────────────────────┐   ┌──────────────────────────────┐
                │   开放世界隐私发现     │   │        隐私保护                │
                │ open_world_discovery/ │   │     privacy_protect/          │
                │ 双模型差分 + 追踪 +    │   │  车牌/工牌 模糊·马赛克          │
                │ 聚类 + 时空关联        │   │  人脸 SimSwap 合成替换          │
                └──────────────────────┘   └──────────────────────────────┘
```

- **model/**：共享的模型定义（CLIP 分类头、SimSwap 生成器等）。
- **classify/**：少样本分类，实现 SWAT（Stage-wise Retrieval-Augmented Finetuning）方法。
- **auto_label/**：基于共享特征图的弱监督自动标注，把"逐窗口独立编码"转化为"一次编码、多次查询"。
- **object_detection/**：SWAT 思想迁移到目标检测的两阶段 YOLOv8 训练。
- **open_world_discovery/**：开放世界隐私目标自动发现。
- **privacy_protect/**：对检测到的敏感目标做差异化隐私保护。

## 目录结构

```
├── README.md
├── requirements.txt
├── model/                          # 共享模型定义
│   ├── __init__.py
│   └── models.py                   # CLIP 加载器、分类头、SimSwap 生成器
├── classify/                       # 少样本分类（SWAT）
│   ├── config.yml                  # 分类训练配置
│   ├── demo_data.py                # 生成合成数据做端到端验证
│   ├── prepare_data.py             # 划分 few-shot 训练/测试集
│   ├── retrieve_data.py            # 网络检索增强数据下载
│   ├── train.py                    # 两阶段训练（含 WiSE-FT、Tau Norm）
│   └── test.py                     # 评估 + 单张推理
├── auto_label/                     # 弱监督自动标注
│   ├── auto_label.py               # 滑动窗口 + RoIAlign 自动标注
│   └── crawl_detection_images.py   # 爬取场景图片
├── object_detection/               # 少样本目标检测
│   ├── det_config.yml              # 检测训练配置
│   ├── det_dataset.py              # 合并手工标注 + 自动标注
│   ├── det_train.py                # 两阶段训练 + 模型融合
│   └── det_test.py                 # 评估 / 对比 / 推理 / 导出 ONNX
├── open_world_discovery/           # 开放世界隐私发现
│   ├── build_reference.py          # 构建 CLIP 视觉参考库
│   └── discover.py                 # 双模型差分 + 时空关联发现
├── privacy_protect/                # 隐私保护
│   ├── badge_plate_privacy_protect.py  # 车牌/工牌 模糊·马赛克
│   └── face_privacy_protect.py         # 人脸 SimSwap 合成替换
├── data/                           # 数据目录（见下方"数据准备"）
└── output/                         # 输出目录（模型 / 标注 / 评估结果 / 保护后视频）
```

## 环境依赖

建议使用 Python 3.9+，推荐 CUDA 环境。

```bash
pip install -r requirements.txt
```

核心依赖：`torch`、`torchvision`、`open-clip-torch`、`ultralytics`（YOLOv8）、`opencv-python`、`numpy`、`Pillow`、`PyYAML`、`requests`、`scikit-image`。

## 数据准备

图片按类别分文件夹存放，支持 `.jpg / .jpeg / .png / .bmp`：

```
data/face_badge_plate/
├── face/              # 所有人脸图片
├── license_plate/     # 所有车牌图片
└── badge/             # 所有工牌图片
```

类别映射：`0 = face`，`1 = license_plate`，`2 = badge`。每类至少 `shots + 2` 张（如 5-shot 至少 7 张/类）。

> 没有真实数据时，可运行 `python classify/demo_data.py` 生成合成数据验证整条流水线。

## 运行流程

### 1. 少样本分类（SWAT）

```bash
# 1) 划分训练/测试集
python classify/prepare_data.py --shots 5 --seed 1

# 2) 网络检索下载增强数据（每类下载 200 张，CLIP 过滤保留 100 张）
python classify/retrieve_data.py --num_download 200 --num_keep 100

# 3) SWAT 训练（默认 mixup-fs 方法，详见 classify/config.yml）
python classify/train.py --data_source fewshot+retrieved --method mixup-fs --shots 5

# 4) 评估
python classify/test.py --model_path output/<run>/wsft_model.pth --mode evaluate

# 5) 单张推理
python classify/test.py --model_path output/<run>/wsft_model.pth --mode single --image xxx.jpg
```

### 2. 自动标注

```bash
# 爬取场景图片
python auto_label/crawl_detection_images.py --num_per_class 500

# 用训练好的分类器做自动标注（输出 YOLO 格式 + 可视化 30 张）
python auto_label/auto_label.py \
  --model_path output/<run>/wsft_model.pth \
  --input_split raw_scenes/unlabeled_scenes.txt \
  --conf_threshold 0.6 --iou_threshold 0.4 \
  --output_format yolo --visualize 30
```

### 3. 少样本目标检测

```bash
# 1) 合并手工标注 + 自动标注数据集
python object_detection/det_dataset.py \
  --hand_dir data/det_hand_labeled \
  --auto_label_dir output/labels \
  --auto_image_dir data/face_badge_plate/raw_scenes

# 2) 两阶段训练
python object_detection/det_train.py \
  --data output/detection/dataset/data.yaml \
  --model yolov8n --oversample_hand 3 \
  --stage1_epochs 50 --stage2_epochs 30

# 3) 评估 / 推理 / 导出 ONNX
python object_detection/det_test.py --mode eval \
  --model output/detection/ensemble/best_ensemble.pt \
  --data output/detection/dataset/data.yaml
```

### 4. 开放世界隐私发现

```bash
# 1) 构建视觉参考库（先放无标注参考图到 data/open_world/reference/）
python open_world_discovery/build_reference.py --num_clusters 8

# 2) 运行发现流水线
python open_world_discovery/discover.py \
  --baseline_model output/models/detection/best.pt \
  --video data/video/discovery/ \
  --conf_unknown 0.2 --corr_threshold 0.3 --frame_skip 5
```

输出：相关的未知目标视频片段（`data/open_world/clips/`，绿框=已知、红框=未知）与 JSON 报告（`data/open_world/reports/`）。

### 5. 隐私保护

```bash
# 方案 A：模糊车牌 + 工牌（人脸保留）
python privacy_protect/badge_plate_privacy_protect.py \
  --model output/models/detection/best_ensemble.pt \
  --input data/video/protect/video.mp4 \
  --blur_mode gaussian --blur_strength 31

# 方案 B：SimSwap 合成替换人脸（不可逆）
python privacy_protect/face_privacy_protect.py \
  --detection_model output/models/detection/best_ensemble.pt \
  --simswap_model model/weights/90000_net_G.pth \
  --input data/video/protect/
```

保护后的视频输出到 `output/protected_videos/`。

## 模型权重说明

- **SimSwap 生成器权重**（`model/weights/90000_net_G.pth`，约 220MB）：`face_privacy_protect.py` 依赖它，但**未包含在本代码包中**。请自行下载 SimSwap 预训练权重并放置到 `model/weights/` 目录，或通过 `--simswap_model` 参数指定其路径。
- 分类 / 检测模型为训练产物，训练后输出到 `output/models/`，供后续自动标注、发现与保护模块复用。

## 注意

代码中各脚本默认通过 `sys.path.insert(...)` 定位到项目根目录的 `model` 包，请保证从项目根目录运行命令（或保持 `model/` 与各模块在同一级目录），否则需自行调整 `PYTHONPATH`。配置文件中出现的 `custom/output` 等历史路径，如与实际布局不一致，请按本地目录结构微调。
