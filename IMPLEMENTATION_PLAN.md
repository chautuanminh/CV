# Tennis Video Analysis — Implementation Plan

## Goal

Build a system that processes tennis match video to **track the ball, players, and court** — then derives **ball speed**, **trajectories**, and **serve counts per player**.

---

## System Architecture

```
Input Video
    ├── Model 1: Player Detection (YOLOv8x + BotSORT tracking)
    ├── Model 2: Ball Detection (YOLOv8 or TrackNet heatmap)
    └── Model 3: Court Keypoint Detection (ResNet50 → 14 keypoints)
            │
            ▼
      Homography Matrix (pixel → real-world meters)
            │
            ▼
      Analytics Engine
        ├── Ball speed (m/s → km/h)
        ├── Serve detection (toss + speed spike + direction)
        └── Per-player serve counts
            │
            ▼
      Outputs: Annotated Video + JSON Stats
```

---

## Phase 1 — Data Collection & Labeling

| Component | Labels Needed | Recommended Tool | Min. Size |
|-----------|--------------|------------------|-----------|
| Players | Bounding boxes (`player`) | Roboflow / CVAT | ~500 images |
| Ball | Bounding boxes (`ball`) or center points | Roboflow | ~5,000 frames |
| Court | 14 keypoint (x,y) coordinates per image | Label Studio | ~1,000 images |

### Steps
1. Collect broadcast or fixed-camera tennis footage (720p+)
2. Extract frames: `python data/extract_frames.py video.mp4 -f 30`
3. Label using Roboflow → export as YOLO format for players/ball
4. Label court keypoints → export as JSON
5. Convert annotations if needed: `python data/convert_annotations.py coco2yolo ...`
6. Split: 70% train / 20% val / 10% test

---

## Phase 2 — Model Training

### 2.1 Player Detection (YOLOv8)

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8x.pt` (COCO pre-trained) |
| Input size | 640×640 |
| Epochs | 100 |
| Batch size | 16 |

```bash
python train_player_detector.py train --data data/players.yaml --epochs 100 --device 0
```

### 2.2 Ball Detection

**Option A — YOLOv8** (simpler, start here):

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8m.pt` |
| Input size | 1280×1280 (high-res for small objects) |
| Epochs | 150 |

```bash
python train_ball_detector.py train --data data/ball.yaml --epochs 150 --imgsz 1280 --device 0
```

**Option B — TrackNet** (higher accuracy for fast/blurry balls):

| Parameter | Value |
|-----------|-------|
| Input | 3 consecutive frames (640×360) |
| Output | Heatmap with Gaussian at ball center |
| Epochs | 500 |

```bash
python train_tracknet.py train --frames-dir data/frames --annotations data/ball_annotations.json --device cuda:0
```

### 2.3 Court Keypoint Detection (ResNet50)

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet50 (ImageNet pre-trained) |
| Output | 28 values (14 keypoints × 2 coords) |
| Epochs | 200 |

```bash
python train_court_detector.py train --images-dir data/court_images --annotations data/court_keypoints.json --device cuda:0
```

---

## Phase 3 — Analytics Pipeline

### Ball Speed
1. Court keypoints → homography matrix (pixel → real-world meters)
2. Transform ball positions per frame
3. Speed = frame-to-frame displacement × FPS → km/h
4. Smoothed via rolling average (window=5)

### Serve Detection
1. **Ball toss** — ball moves upward significantly near a player
2. **Contact** — speed spike above threshold (80+ km/h)
3. **Assignment** — nearest tracked player = server
4. Cooldown prevents double-counting

---

## Phase 4 — Running the Full Pipeline

```bash
# Process a video end-to-end
python pipeline.py match.mp4 --ball-model yolo --output output/result.mp4

# With TrackNet instead
python pipeline.py match.mp4 --ball-model tracknet --tracknet-weights weights/tracknet/tracknet_best.pth
```

**Outputs:**
- `output/result.mp4` — annotated video with player boxes, ball trail, speed overlay
- `output/result.json` — ball speeds, serve counts per player, detection rates

---

## Phase 5 — Training Schedule

| Week | Task | Deliverable |
|------|------|-------------|
| 1–2 | Data collection & labeling | Labeled datasets |
| 3 | Train player detector | `best_player.pt`, mAP > 0.90 |
| 4–5 | Train ball detector | `best_ball.pt`, F1 > 0.85 |
| 5–6 | Train court model | `court_best.pth`, error < 5px |
| 7 | Integrate pipeline | End-to-end processing |
| 8 | Build analytics | Speed + serve counting |
| 9 | Evaluation & tuning | Accuracy report |
| 10 | Final output | Demo with overlays + stats |

---

## Evaluation Targets

| Component | Metric | Target |
|-----------|--------|--------|
| Player detection | mAP@0.5 | > 0.90 |
| Ball detection | F1-score | > 0.85 |
| Ball tracking | Detection rate | > 90% |
| Court keypoints | Mean error | < 5 px |
| Ball speed | vs. ground truth | ±10% |
| Serve detection | Precision / Recall | > 0.90 / 0.85 |

---

## Project Structure

```
pickleball/
├── config.py                      # Central configuration
├── pipeline.py                    # End-to-end video processor
├── requirements.txt               # Python dependencies
├── IMPLEMENTATION_PLAN.md         # This file
├── train_player_detector.py       # YOLOv8 player training
├── train_ball_detector.py         # YOLOv8 ball training
├── train_tracknet.py              # TrackNet ball training
├── train_court_detector.py        # Court keypoint training
├── models/
│   ├── tracknet.py                # TrackNet architecture
│   └── court_keypoint.py          # ResNet50 court model
├── analytics/
│   ├── speed.py                   # Ball speed estimation
│   ├── serve_detection.py         # Serve counting
│   └── player_assignment.py       # Player side assignment
├── data/
│   ├── extract_frames.py          # Frame extraction utility
│   ├── dataset.py                 # PyTorch dataset classes
│   └── convert_annotations.py     # COCO/VOC → YOLO converters
├── weights/                       # Trained model weights (gitignored)
└── output/                        # Pipeline output videos + JSON
```

---

## Google Colab Setup

Since training requires a GPU, use Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/pickleball')

!pip install -q ultralytics torch torchvision opencv-python-headless numpy pandas supervision scipy tqdm
```

Then run any training script with `!python train_*.py train ...` using `--device 0` for GPU.
