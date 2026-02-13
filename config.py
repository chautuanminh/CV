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


# ──────────────────────────── Court Keypoint Detection ─────────────────
COURT_MODEL = {
    "backbone": "resnet50",
    "num_keypoints": 14,             # 14 landmarks on a tennis court
    "input_size": (224, 224),
    "epochs": 200,
    "batch_size": 16,
    "lr": 0.0001,
    "weight_decay": 1e-5,
}

# ITF standard tennis court dimensions in meters
COURT_DIMENSIONS = {
    "length": 23.77,                 # Baseline to baseline
    "width_singles": 8.23,
    "width_doubles": 10.97,
    "service_line_dist": 6.40,       # From net to service line
    "net_height": 0.914,             # Center net height
}

# 14 reference keypoints in real-world coordinates (meters)
# Origin = bottom-left corner of the doubles court
COURT_REFERENCE_POINTS = [
    # Near baseline (bottom)
    (0.0, 0.0),                      # 0: bottom-left corner
    (10.97, 0.0),                    # 1: bottom-right corner
    # Far baseline (top)
    (0.0, 23.77),                    # 2: top-left corner
    (10.97, 23.77),                  # 3: top-right corner
    # Near service line
    (1.37, 6.40),                    # 4: service-left (near)
    (9.60, 6.40),                    # 5: service-right (near)
    # Far service line
    (1.37, 17.37),                   # 6: service-left (far)
    (9.60, 17.37),                   # 7: service-right (far)
    # Net posts
    (0.0, 11.885),                   # 8: net-left
    (10.97, 11.885),                 # 9: net-right
    # Center service line endpoints
    (5.485, 6.40),                   # 10: center-service (near)
    (5.485, 17.37),                  # 11: center-service (far)
    # Center marks
    (5.485, 0.0),                    # 12: center-mark (near baseline)
    (5.485, 23.77),                  # 13: center-mark (far baseline)
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
