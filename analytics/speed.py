"""
Ball speed estimation using court homography.

Transforms ball pixel positions to real-world court coordinates via a
homography matrix, then computes speed from frame-to-frame displacement.
"""

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COURT_REFERENCE_POINTS, ANALYTICS, VIDEO


def compute_homography(
    detected_keypoints: np.ndarray,
    reference_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Compute the homography matrix from detected court keypoints
    to real-world court coordinates.

    Args:
        detected_keypoints: Pixel coordinates of court keypoints, shape (N, 2).
        reference_points:   Real-world coordinates, shape (N, 2). Defaults to
                            ITF standard court from config.

    Returns:
        3×3 homography matrix.
    """
    if reference_points is None:
        reference_points = COURT_REFERENCE_POINTS

    src = np.array(detected_keypoints, dtype=np.float32)
    dst = np.array(reference_points, dtype=np.float32)

    # Use only keypoints that are valid (non-zero / non-NaN)
    valid = ~np.isnan(src).any(axis=1) & ~np.isnan(dst).any(axis=1)
    src = src[valid]
    dst = dst[valid]

    if len(src) < 4:
        raise ValueError(f"Need at least 4 valid keypoints for homography, got {len(src)}")

    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed")

    return H


def pixel_to_real(
    pixel_pos: tuple[float, float],
    homography: np.ndarray,
) -> tuple[float, float]:
    """
    Transform a pixel position to real-world court coordinates.

    Args:
        pixel_pos:   (x, y) in pixels.
        homography:  3×3 homography matrix.

    Returns:
        (x, y) in meters on the court plane.
    """
    pt = np.array([pixel_pos[0], pixel_pos[1], 1.0], dtype=np.float64)
    real = homography @ pt
    real /= real[2]
    return float(real[0]), float(real[1])


def compute_ball_speed(
    positions: list[dict],
    homography: np.ndarray,
    fps: float | None = None,
    smoothing_window: int | None = None,
) -> list[dict]:
    """
    Compute ball speed for each frame using homography-transformed positions.

    Args:
        positions:         List of {"frame": int, "x": float|None, "y": float|None}.
        homography:        3×3 homography matrix.
        fps:               Frames per second (default from config).
        smoothing_window:  Rolling average window size (default from config).

    Returns:
        List of {"frame": int, "speed_mps": float, "speed_kmh": float}.
    """
    fps = fps or VIDEO["default_fps"]
    window = smoothing_window or ANALYTICS["speed_smoothing_window"]

    # Convert all valid positions to real-world
    real_positions = []
    for p in positions:
        if p["x"] is not None and p["y"] is not None:
            rx, ry = pixel_to_real((p["x"], p["y"]), homography)
            real_positions.append({"frame": p["frame"], "rx": rx, "ry": ry})
        else:
            real_positions.append({"frame": p["frame"], "rx": None, "ry": None})

    # Compute frame-to-frame speed
    speeds = []
    for i in range(1, len(real_positions)):
        curr = real_positions[i]
        prev = real_positions[i - 1]

        if curr["rx"] is not None and prev["rx"] is not None:
            dx = curr["rx"] - prev["rx"]
            dy = curr["ry"] - prev["ry"]
            dist = np.sqrt(dx**2 + dy**2)
            speed_mps = dist * fps
            speed_kmh = speed_mps * 3.6
        else:
            speed_mps = 0.0
            speed_kmh = 0.0

        speeds.append({
            "frame": curr["frame"],
            "speed_mps": speed_mps,
            "speed_kmh": speed_kmh,
        })

    # Apply rolling average smoothing
    if len(speeds) >= window:
        speed_vals = np.array([s["speed_kmh"] for s in speeds])
        kernel = np.ones(window) / window
        smoothed = np.convolve(speed_vals, kernel, mode="same")
        for i, s in enumerate(speeds):
            s["speed_kmh_smooth"] = float(smoothed[i])
            s["speed_mps_smooth"] = float(smoothed[i] / 3.6)

    return speeds


def get_max_speed(speeds: list[dict]) -> dict:
    """Get the frame with the highest ball speed."""
    if not speeds:
        return {"frame": -1, "speed_kmh": 0}
    key = "speed_kmh_smooth" if "speed_kmh_smooth" in speeds[0] else "speed_kmh"
    return max(speeds, key=lambda s: s[key])
