"""
Custom dataset classes for court keypoint detection and TrackNet ball tracking.
YOLO datasets use Ultralytics' built-in data loading via YAML configs.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────── Court Keypoint Dataset ────────────────────────

class CourtKeypointDataset(Dataset):
    """
    Dataset for court keypoint detection.

    Expected annotation format (JSON per image):
    {
        "image": "frame_000100.jpg",
        "keypoints": [[x1, y1], [x2, y2], ..., [x14, y14]]
    }

    Keypoints are normalized to [0, 1] relative to image dimensions.
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        input_size: tuple[int, int] = (224, 224),
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        self.augment = augment

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        img_path = self.images_dir / ann["image"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Keypoints as flat tensor [x1, y1, x2, y2, ..., x14, y14]
        keypoints = np.array(ann["keypoints"], dtype=np.float32).flatten()

        if self.augment:
            image = self.augment_transform(image)
        else:
            image = self.transform(image)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return image, keypoints


# ─────────────────────── TrackNet Ball Dataset ─────────────────────────

class TrackNetDataset(Dataset):
    """
    Dataset for TrackNet ball detection.

    Each sample is a sequence of `num_frames` consecutive frames.
    The target is a 2D Gaussian heatmap centered on the ball position
    in the last frame of the sequence.

    Expected annotation format (JSON):
    [
        {"frame": "frame_000001.jpg", "x": 320, "y": 180, "visible": true},
        {"frame": "frame_000002.jpg", "x": null, "y": null, "visible": false},
        ...
    ]
    """

    def __init__(
        self,
        frames_dir: str,
        annotations_file: str,
        num_frames: int = 3,
        input_height: int = 360,
        input_width: int = 640,
        sigma: float = 2.5,
    ):
        self.frames_dir = Path(frames_dir)
        self.num_frames = num_frames
        self.input_height = input_height
        self.input_width = input_width
        self.sigma = sigma

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        # Filter valid sequences (need num_frames consecutive frames)
        self.valid_indices = list(range(num_frames - 1, len(self.annotations)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _generate_heatmap(self, x: float | None, y: float | None) -> np.ndarray:
        """Create a 2D Gaussian heatmap centered at (x, y)."""
        heatmap = np.zeros((self.input_height, self.input_width), dtype=np.float32)
        if x is None or y is None:
            return heatmap

        x, y = int(round(x)), int(round(y))
        if x < 0 or x >= self.input_width or y < 0 or y >= self.input_height:
            return heatmap

        # Build Gaussian
        yy, xx = np.mgrid[0:self.input_height, 0:self.input_width]
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
        return heatmap.astype(np.float32)

    def __getitem__(self, idx: int):
        actual_idx = self.valid_indices[idx]

        # Load consecutive frames
        frames = []
        for i in range(self.num_frames):
            ann = self.annotations[actual_idx - (self.num_frames - 1) + i]
            img_path = self.frames_dir / ann["frame"]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_width, self.input_height))
            img = img.astype(np.float32) / 255.0
            frames.append(img)

        # Stack frames → (num_frames * 3, H, W)
        frames = np.concatenate(frames, axis=2)  # (H, W, num_frames*3)
        frames = np.transpose(frames, (2, 0, 1))  # (C, H, W)

        # Heatmap for the LAST frame
        last_ann = self.annotations[actual_idx]
        heatmap = self._generate_heatmap(last_ann.get("x"), last_ann.get("y"))

        return torch.tensor(frames), torch.tensor(heatmap).unsqueeze(0)  # (1, H, W)
