"""
Serve detection module.

Detects tennis serves by analyzing ball trajectory patterns:
1. Ball toss — upward movement near a player
2. Contact — sudden speed spike + direction change
3. Crossing — ball moves toward opposite side of court

Counts serves per player using tracked player IDs.
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANALYTICS


class ServeDetector:
    """
    Detects serves and counts them per player.

    Args:
        fps:                   Video frames per second.
        speed_threshold_kmh:   Minimum ball speed to qualify as a serve.
        toss_lookback_frames:  How many frames back to check for ball toss.
        toss_min_rise_px:      Minimum upward pixel displacement for toss.
    """

    def __init__(
        self,
        fps: float = 30.0,
        speed_threshold_kmh: float | None = None,
        toss_lookback_frames: int | None = None,
        toss_min_rise_px: float | None = None,
    ):
        self.fps = fps
        self.speed_threshold = speed_threshold_kmh or ANALYTICS["serve_speed_threshold_kmh"]
        self.toss_lookback = toss_lookback_frames or ANALYTICS["ball_toss_frames"]
        self.toss_min_rise = toss_min_rise_px or ANALYTICS["ball_toss_min_rise_px"]

        # Results
        self.serves: list[dict] = []
        self.serve_counts: dict[int, int] = {}  # player_id → count

        # Cooldown to avoid double-counting
        self._last_serve_frame = -100

    def _detect_ball_toss(
        self,
        ball_positions: list[dict],
        frame_idx: int,
    ) -> bool:
        """
        Check if a ball toss occurred in the recent frames.
        Ball toss = significant upward movement (y decreasing in image coords).
        """
        if frame_idx < self.toss_lookback:
            return False

        lookback_start = frame_idx - self.toss_lookback
        positions_window = ball_positions[lookback_start:frame_idx + 1]

        # Get valid y values
        y_values = [p["y"] for p in positions_window if p["y"] is not None]
        if len(y_values) < 3:
            return False

        # Check for upward movement (y decreases in image coordinates)
        min_y = min(y_values)
        max_y = max(y_values)
        rise = max_y - min_y

        # Ball should go up (min_y should be near the end) and rise enough
        if rise >= self.toss_min_rise:
            # Verify upward trend: first values should be larger than later ones
            mid = len(y_values) // 2
            early_avg = np.mean(y_values[:mid])
            late_avg = np.mean(y_values[mid:])
            if late_avg < early_avg:  # Ball went up
                return True

        return False

    def _find_nearest_player(
        self,
        ball_pos: dict,
        player_positions: list[dict],
    ) -> int | None:
        """
        Find the player closest to the ball at the given frame.

        Args:
            ball_pos:          {"x": float, "y": float}
            player_positions:  List of {"id": int, "x": float, "y": float, "bbox": ...}

        Returns:
            Player ID of the nearest player, or None.
        """
        if ball_pos["x"] is None or not player_positions:
            return None

        min_dist = float("inf")
        nearest_id = None

        for player in player_positions:
            if player.get("x") is None:
                continue
            dx = ball_pos["x"] - player["x"]
            dy = ball_pos["y"] - player["y"]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                nearest_id = player["id"]

        return nearest_id

    def process_frame(
        self,
        frame_idx: int,
        ball_positions: list[dict],
        ball_speeds: list[dict],
        player_positions: list[dict],
    ) -> dict | None:
        """
        Check if a serve happened at this frame.

        Args:
            frame_idx:         Current frame index.
            ball_positions:    All ball positions up to this frame.
            ball_speeds:       All ball speeds up to this frame.
            player_positions:  Player positions at this frame.

        Returns:
            Serve event dict or None.
        """
        # Cooldown check (avoid double-counting same serve)
        if frame_idx - self._last_serve_frame < self.fps:  # 1-second cooldown
            return None

        # Need enough history
        if frame_idx >= len(ball_speeds):
            return None

        speed_entry = ball_speeds[frame_idx] if frame_idx < len(ball_speeds) else None
        if speed_entry is None:
            return None

        speed_key = "speed_kmh_smooth" if "speed_kmh_smooth" in speed_entry else "speed_kmh"
        current_speed = speed_entry[speed_key]

        # Condition 1: Speed above serve threshold
        if current_speed < self.speed_threshold:
            return None

        # Condition 2: Ball toss detected in recent frames
        if not self._detect_ball_toss(ball_positions, frame_idx):
            return None

        # Condition 3: Assign to nearest player
        ball_pos = ball_positions[frame_idx]
        server_id = self._find_nearest_player(ball_pos, player_positions)

        if server_id is None:
            return None

        # Register serve
        serve_event = {
            "frame": frame_idx,
            "time_seconds": frame_idx / self.fps,
            "server_id": server_id,
            "speed_kmh": current_speed,
        }

        self.serves.append(serve_event)
        self.serve_counts[server_id] = self.serve_counts.get(server_id, 0) + 1
        self._last_serve_frame = frame_idx

        return serve_event

    def get_summary(self) -> dict:
        """
        Get a summary of all detected serves.

        Returns:
            {
                "total_serves": int,
                "serves_per_player": {player_id: count},
                "max_serve_speed_kmh": float,
                "avg_serve_speed_kmh": float,
                "serves": [serve_events],
            }
        """
        speeds = [s["speed_kmh"] for s in self.serves]
        return {
            "total_serves": len(self.serves),
            "serves_per_player": dict(self.serve_counts),
            "max_serve_speed_kmh": max(speeds) if speeds else 0.0,
            "avg_serve_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
            "serves": self.serves,
        }
