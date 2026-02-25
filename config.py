"""
Central configuration for the tennis video analysis pipeline.
All paths, model parameters, and thresholds are defined here.
"""

from pathlib import Path

# ──────────────────────────── Project Paths ────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
WEIGHTS_DIR = ROOT_DIR / "weights"
OUTPUT_DIR = ROOT_DIR / "output"

# Create directories on import
for d in [DATA_DIR, MODELS_DIR, WEIGHTS_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)


# ──────────────────────────── Player Detection ─────────────────────────
PLAYER_MODEL = {
    "base_weights": "yolov8x.pt",
    "dataset_yaml": str(DATA_DIR / "players.yaml"),
    "imgsz": 640,
    "epochs": 100,
    "batch_size": 16,
    "optimizer": "AdamW",
    "lr": 0.001,
    "classes": ["player"],
    "tracker": "botsort.yaml",       # BotSORT for persistent tracking
    "confidence_threshold": 0.5,
}


# ──────────────────────────── Ball Detection (YOLO) ────────────────────
BALL_MODEL_YOLO = {
    "base_weights": "yolov8m.pt",
    "dataset_yaml": str(DATA_DIR / "ball.yaml"),
    "imgsz": 1280,                   # High resolution for small object
    "epochs": 150,
    "batch_size": 8,
    "optimizer": "AdamW",
    "lr": 0.001,
    "classes": ["ball"],
    "confidence_threshold": 0.15,    # Lower threshold — better recall
    "interpolation_order": 2,        # Polynomial order for gap filling
}


# ──────────────────────────── Ball Detection (TrackNet) ────────────────
BALL_MODEL_TRACKNET = {
    "input_height": 360,
    "input_width": 640,
    "num_input_frames": 3,           # Consecutive frames as input
    "gaussian_sigma": 2.5,           # Heatmap ground-truth σ
    "epochs": 500,
    "steps_per_epoch": 200,
    "batch_size": 2,
    "lr": 0.001,
    "heatmap_threshold": 0.5,        # Detection confidence threshold
}


# ──────────────────────────── Court Segmentation (Homography) ──────────
COURT_SEGMENT_MODEL = {
    "weights": str(MODELS_DIR / "court_segment.pt"),
    "confidence_threshold": 0.25,
}

# Standard pickleball court dimensions (feet)
PICKLEBALL_COURT = {
    "width_ft": 20,                  # Sideline to sideline
    "length_ft": 44,                 # Baseline to baseline
    "nvz_depth_ft": 7,               # Non-volley zone (kitchen) depth from net
    "net_position_ft": 22,           # Net at mid-court
}

# Scaled reference coordinate system: 1 ft = 10 units
import numpy as np

_SCALE = 10
_W   = PICKLEBALL_COURT["width_ft"]  * _SCALE   # 200
_H   = PICKLEBALL_COURT["length_ft"] * _SCALE   # 440
_NET = _H // 2                                    # 220
_NVZ = PICKLEBALL_COURT["nvz_depth_ft"] * _SCALE  # 70
_NVZ_TOP = _NET - _NVZ                            # 150
_NVZ_BOT = _NET + _NVZ                            # 290
_MID = _W // 2                                     # 100

# Reference corners for homography: [TL, TR, BR, BL]
STANDARD_CORNERS = np.array([
    [0,   0  ],   # Top-left
    [_W,  0  ],   # Top-right
    [_W,  _H ],   # Bottom-right
    [0,   _H ],   # Bottom-left
], dtype="float32")

# All court line segments in the reference coordinate system
COURT_LINES_REF = [
    # Outer boundary
    ([0, 0],        [_W, 0      ]),   # Top baseline
    ([_W, 0],       [_W, _H     ]),   # Right sideline
    ([_W, _H],      [0,  _H     ]),   # Bottom baseline
    ([0,  _H],      [0,  0      ]),   # Left sideline
    # Net
    ([0, _NET],     [_W, _NET   ]),
    # Non-Volley Zone (kitchen) lines
    ([0, _NVZ_TOP], [_W, _NVZ_TOP]),
    ([0, _NVZ_BOT], [_W, _NVZ_BOT]),
    # Centre service lines (baseline → NVZ only, NOT through kitchen)
    ([_MID, 0      ], [_MID, _NVZ_TOP]),   # Top half
    ([_MID, _NVZ_BOT], [_MID, _H      ]),  # Bottom half
]

# Real-world court reference points in feet (for speed estimation)
# 4 corners: TL, TR, BR, BL — same order as STANDARD_CORNERS
COURT_REFERENCE_POINTS_FT = [
    (0.0, 0.0),                       # TL
    (20.0, 0.0),                      # TR
    (20.0, 44.0),                     # BR
    (0.0, 44.0),                      # BL
]


# ──────────────────────────── Analytics ────────────────────────────────
ANALYTICS = {
    "serve_speed_threshold_kmh": 80,     # Minimum speed to register a serve
    "ball_toss_frames": 15,              # Lookback window for toss detection
    "ball_toss_min_rise_px": 50,         # Min upward pixel displacement for toss
    "speed_smoothing_window": 5,         # Rolling avg window for speed
}


# ──────────────────────────── Video / General ──────────────────────────
VIDEO = {
    "default_fps": 30,
    "frame_extraction_fps": 30,
    "output_codec": "mp4v",
    "annotation_color_player": (0, 255, 0),   # Green
    "annotation_color_ball": (0, 255, 255),    # Yellow
    "annotation_color_court": (255, 0, 0),     # Blue
}
