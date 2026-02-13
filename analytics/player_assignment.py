"""
Player assignment utilities.

Assigns player identities and tracks which side of the court
each player is on for serve attribution and analytics.
"""

import numpy as np


def assign_player_sides(
    player_positions: list[dict],
    court_center_y: float,
) -> list[dict]:
    """
    Assign 'near' or 'far' side labels to players based on court position.

    Players below court_center_y are 'near', above are 'far'.

    Args:
        player_positions:  List of {"id": int, "x": float, "y": float, ...}
        court_center_y:    Y-coordinate of the net in pixel space.

    Returns:
        Same list with added "side" field ("near" or "far").
    """
    for player in player_positions:
        if player.get("y") is not None:
            player["side"] = "near" if player["y"] > court_center_y else "far"
        else:
            player["side"] = "unknown"
    return player_positions


def extract_player_positions_from_boxes(
    boxes: list,
    track_ids: list[int],
) -> list[dict]:
    """
    Convert bounding boxes to center-point player positions.

    Args:
        boxes:     List of [x1, y1, x2, y2] bounding boxes.
        track_ids: Corresponding persistent track IDs.

    Returns:
        List of {"id": int, "x": float, "y": float, "bbox": [x1,y1,x2,y2]}
    """
    players = []
    for bbox, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        players.append({
            "id": int(tid),
            "x": float(cx),
            "y": float(cy),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        })
    return players


def compute_player_stats(
    all_player_positions: list[list[dict]],
    fps: float,
) -> dict:
    """
    Compute per-player movement statistics.

    Args:
        all_player_positions:  List (per frame) of player position lists.
        fps:                   Video frames per second.

    Returns:
        Dict mapping player_id → {
            "total_distance_px": float,
            "avg_speed_px_per_sec": float,
            "frames_visible": int,
        }
    """
    player_tracks: dict[int, list[tuple[float, float]]] = {}

    for frame_players in all_player_positions:
        for player in frame_players:
            pid = player["id"]
            if player["x"] is not None:
                player_tracks.setdefault(pid, []).append((player["x"], player["y"]))

    stats = {}
    for pid, positions in player_tracks.items():
        total_dist = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_dist += np.sqrt(dx**2 + dy**2)

        frames_visible = len(positions)
        duration = frames_visible / fps

        stats[pid] = {
            "total_distance_px": total_dist,
            "avg_speed_px_per_sec": total_dist / duration if duration > 0 else 0,
            "frames_visible": frames_visible,
        }

    return stats
