# Pickleball Video Analysis — Implementation Plan

## Goal

Build a system that processes pickleball match video to **track the ball, players, and court** — then derives analytics and overlays for a comprehensive video output.

---

## System Architecture

```
Input Video
    ├── Model 1: Player Detection (YOLOv8n + ByteTrack)
    ├── Model 2: Ball Detection (YOLOv8n optimized for small objects)
    └── Model 3: Court Segmentation (YOLOv8n Segmentation)
            │
            ▼
      Dynamic Homography Matrix (computed per-frame on mask corners)
            │
            ▼
      Analytics Engine & Overlay Renderer
        ├── Trajectory rendering
        ├── Player tracking trails
        └── Court boundary projection
            │
            ▼
      Outputs: Annotated Video `*_full_pipeline.mp4`
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
| Base model | `yolo26n.pt` (or standard `yolov8n.pt`) |
| Output | Bounding boxes |

### 2.2 Ball Detection (YOLOv8)

Used YOLOv8 optimized for small objects, replacing TrackNet due to faster inference and better coverage/stability (less jitter).

| Parameter | Value |
|-----------|-------|
| Input size | High-res evaluation |
| Tracking | Custom buffer logic mapping trajectory |

### 2.3 Court Segmentation (YOLOv8 Segmentation)

Replaced ResNet50 keypoint regression with robust YOLO instance segmentation.

| Parameter | Value |
|-----------|-------|
| Output | Binary mask of the court surface |
| Processing | Contours -> 4 corners -> Homography Matrix |

---

## Phase 3 — Full Pipeline Execution

Orchestrated through `test/full_pipeline.py`, which:
1. Translates video frames
2. Computes the dynamic homography matrix frame-by-frame
3. Runs ByteTrack-assisted YOLO for players
4. Tracks the ball and maintains trail history
5. Outputs to a combined MP4 video

```bash
# Process a video end-to-end
python test/full_pipeline.py --video res/src/Final.mp4
```

**Outputs:**
- `output/Final_full_pipeline.mp4` — annotated video with players boxes, ball trail, and court projection based on real-time homography.

---

## Evaluation Targets & Metrics

1.  **Coverage:** yolo26n detects the ball in significantly more frames than traditional baselines.
2.  **Stability:** Maintains extremely few jitter spikes compared to TrackNet.
3.  **Performance:** Capable of processing videos purely on CPU at reasonable speeds.

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
├── requirements.txt               # Python dependencies
├── IMPLEMENTATION_PLAN.md         # This file
├── agents.md                      # Pipeline summary
├── models/
│   ├── court_homography.py        # Segmentation-to-homography logic
│   └── ...                        # ONNX/PT model weights
├── test/
│   ├── full_pipeline.py           # End-to-end video processor
│   ├── homography_precheck.py     # Static homography sampling/debug script
│   └── apply_homography_video.py  # Court tracking validation script
├── output/                        # Pipeline output videos
└── resy/                          # README and demo artifacts
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
