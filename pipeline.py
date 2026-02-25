"""
Pickleball Video Analysis — End-to-End Pipeline

Orchestrates all models (player detection, ball detection, court homography)
and analytics modules (speed estimation, serve detection) to process a
pickleball match video and produce annotated output with statistics.

Usage:
    python pipeline.py video.mp4
    python pipeline.py video.mp4 --ball-model yolo --output output/result.mp4
    python pipeline.py video.mp4 --ball-model tracknet --tracknet-weights weights/tracknet/tracknet_best.pth
    python pipeline.py --help
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    PLAYER_MODEL, BALL_MODEL_YOLO, BALL_MODEL_TRACKNET,
    COURT_SEGMENT_MODEL, VIDEO, ANALYTICS,
    WEIGHTS_DIR, OUTPUT_DIR,
)
from models.court_homography import CourtHomographyDetector
from analytics.speed import compute_ball_speed, get_max_speed
from analytics.serve_detection import ServeDetector
from analytics.player_assignment import (
    extract_player_positions_from_boxes,
    assign_player_sides,
    compute_player_stats,
)


class PickleballAnalysisPipeline:
    """
    End-to-end pickleball video analysis pipeline.

    Supports two ball detection backends:
    - 'yolo'     : YOLOv8 bounding-box detector (simpler, faster)
    - 'tracknet' : Heatmap regression model (higher accuracy)

    Court detection uses YOLO segmentation + homography projection
    (replaces fragile Hough Transform / ResNet keypoint approach).

    Outputs:
    - Annotated video with player boxes, ball trail, court overlay
    - JSON statistics: speeds, serve counts, player stats
    """

    def __init__(
        self,
        player_weights: str | None = None,
        ball_model_type: str = "yolo",
        ball_yolo_weights: str | None = None,
        tracknet_weights: str | None = None,
        court_segment_weights: str | None = None,
        device: str = "0",
    ):
        self.device = device
        self.ball_model_type = ball_model_type

        # ── Load player model ──
        from ultralytics import YOLO
        player_path = player_weights or str(WEIGHTS_DIR / "player_detector" / "weights" / "best.pt")
        if Path(player_path).exists():
            self.player_model = YOLO(player_path)
        else:
            # Fall back to pre-trained COCO model
            print(f"[INFO] Player weights not found at {player_path}, using pre-trained yolov8x.pt")
            self.player_model = YOLO("yolov8x.pt")

        # ── Load ball model ──
        if ball_model_type == "yolo":
            ball_path = ball_yolo_weights or str(WEIGHTS_DIR / "ball_detector" / "weights" / "best.pt")
            if Path(ball_path).exists():
                self.ball_model = YOLO(ball_path)
            else:
                print(f"[WARN] Ball YOLO weights not found at {ball_path}. Ball detection will be skipped.")
                self.ball_model = None
        elif ball_model_type == "tracknet":
            import torch
            from models.tracknet import TrackNet

            self.ball_model = TrackNet(num_input_frames=BALL_MODEL_TRACKNET["num_input_frames"])
            tn_path = tracknet_weights or str(WEIGHTS_DIR / "tracknet" / "tracknet_best.pth")
            if Path(tn_path).exists():
                self.ball_model.load_state_dict(torch.load(tn_path, map_location="cpu"))
                self.ball_model.eval()
                if device != "cpu":
                    self.ball_model = self.ball_model.cuda()
            else:
                print(f"[WARN] TrackNet weights not found at {tn_path}. Ball detection will be skipped.")
                self.ball_model = None
        else:
            raise ValueError(f"Unknown ball model type: {ball_model_type}")

        # ── Load court segmentation model (homography-based) ──
        court_seg_path = court_segment_weights or COURT_SEGMENT_MODEL["weights"]
        if Path(court_seg_path).exists():
            self.court_detector = CourtHomographyDetector(
                model_path=court_seg_path,
                conf=COURT_SEGMENT_MODEL["confidence_threshold"],
            )
            self.court_loaded = True
        else:
            print(f"[WARN] Court segmentation weights not found at {court_seg_path}. "
                  f"Court detection & speed estimation will be skipped.")
            self.court_detector = None
            self.court_loaded = False

    def _detect_players(self, frame: np.ndarray) -> list[dict]:
        """Detect and track players in a frame."""
        results = self.player_model.track(
            frame,
            persist=True,
            conf=PLAYER_MODEL["confidence_threshold"],
            classes=[0],  # person class
            verbose=False,
        )

        players = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = (
                results[0].boxes.id.cpu().numpy().astype(int)
                if results[0].boxes.id is not None
                else list(range(len(boxes)))
            )
            players = extract_player_positions_from_boxes(boxes.tolist(), track_ids.tolist())

        return players

    def _detect_ball_yolo(self, frame: np.ndarray) -> dict:
        """Detect ball using YOLOv8."""
        if self.ball_model is None:
            return {"x": None, "y": None}

        results = self.ball_model(
            frame,
            conf=BALL_MODEL_YOLO["confidence_threshold"],
            verbose=False,
        )

        if len(results[0].boxes) > 0:
            best = results[0].boxes[results[0].boxes.conf.argmax()]
            box = best.xyxy[0].cpu().numpy()
            return {"x": float((box[0] + box[2]) / 2), "y": float((box[1] + box[3]) / 2)}

        return {"x": None, "y": None}

    def _detect_ball_tracknet(self, frame_buffer: list[np.ndarray]) -> dict:
        """Detect ball using TrackNet."""
        import torch

        if self.ball_model is None or len(frame_buffer) < BALL_MODEL_TRACKNET["num_input_frames"]:
            return {"x": None, "y": None}

        cfg = BALL_MODEL_TRACKNET
        frames = []
        for f in frame_buffer[-cfg["num_input_frames"]:]:
            resized = cv2.resize(f, (cfg["input_width"], cfg["input_height"]))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(resized)

        stacked = np.concatenate(frames, axis=2)
        tensor = torch.tensor(stacked).permute(2, 0, 1).unsqueeze(0)
        if self.device != "cpu":
            tensor = tensor.cuda()

        with torch.no_grad():
            heatmap = self.ball_model(tensor)

        pos = self.ball_model.detect_ball(heatmap, threshold=cfg["heatmap_threshold"])
        if pos is not None:
            # Scale to original frame dimensions
            orig_h, orig_w = frame_buffer[-1].shape[:2]
            sx = orig_w / cfg["input_width"]
            sy = orig_h / cfg["input_height"]
            return {"x": float(pos[0] * sx), "y": float(pos[1] * sy)}

        return {"x": None, "y": None}

    def _detect_court(self, frame: np.ndarray):
        """
        Detect court via YOLO segmentation + homography.

        Returns:
            (corners, H, mask) or (None, None, None).
        """
        if not self.court_loaded or self.court_detector is None:
            return None, None, None
        return self.court_detector.detect(frame)

    def _draw_annotations(
        self,
        frame: np.ndarray,
        players: list[dict],
        ball_pos: dict,
        ball_trail: list[dict],
        court_H: np.ndarray | None,
        speed_kmh: float | None,
        serve_event: dict | None,
        serve_counts: dict,
    ) -> np.ndarray:
        """Draw all annotations on a frame."""
        annotated = frame.copy()

        # Draw court lines via homography projection
        if court_H is not None:
            annotated = CourtHomographyDetector.project_court_lines(
                annotated, court_H,
                color=VIDEO["annotation_color_court"],
                thickness=2,
            )

        # Draw player bounding boxes
        for player in players:
            bbox = player.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = VIDEO["annotation_color_player"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"P{player['id']}"
                if player['id'] in serve_counts:
                    label += f" | Serves: {serve_counts[player['id']]}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw ball trail
        trail_points = [
            (int(p["x"]), int(p["y"]))
            for p in ball_trail[-30:]  # Last 30 positions
            if p["x"] is not None
        ]
        for i in range(1, len(trail_points)):
            cv2.line(annotated, trail_points[i - 1], trail_points[i],
                     VIDEO["annotation_color_ball"], 2)

        # Draw current ball position
        if ball_pos["x"] is not None:
            cv2.circle(annotated, (int(ball_pos["x"]), int(ball_pos["y"])),
                       8, VIDEO["annotation_color_ball"], -1)

        # Draw speed
        if speed_kmh is not None and speed_kmh > 0:
            cv2.putText(annotated, f"Ball: {speed_kmh:.0f} km/h", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Draw serve event
        if serve_event is not None:
            cv2.putText(annotated, f"SERVE! P{serve_event['server_id']} - {serve_event['speed_kmh']:.0f} km/h",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        show_preview: bool = False,
    ) -> dict:
        """
        Process a full video through the analysis pipeline.

        Args:
            video_path:    Path to input video.
            output_path:   Path for annotated output video (None = auto).
            show_preview:  If True, display live preview window.

        Returns:
            Dictionary with all analytics results.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO["default_fps"]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output writer
        if output_path is None:
            output_path = str(OUTPUT_DIR / f"{video_path.stem}_analyzed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*VIDEO["output_codec"])
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("=" * 60)
        print("  Pickleball Analysis Pipeline")
        print("=" * 60)
        print(f"  Video     : {video_path.name}")
        print(f"  Resolution: {width}×{height} @ {fps:.1f} FPS")
        print(f"  Frames    : {total_frames}")
        print(f"  Ball model: {self.ball_model_type}")
        print(f"  Output    : {output_path}")
        print()

        # State
        ball_positions: list[dict] = []
        all_player_positions: list[list[dict]] = []
        frame_buffer: list[np.ndarray] = []
        homography = None
        serve_detector = ServeDetector(fps=fps)
        ball_speeds: list[dict] = []

        # Detect court on first frame (assume fixed camera)
        start_time = time.time()

        pbar = tqdm(total=total_frames, desc="Processing")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Court detection (once on first frame) ──
            if frame_idx == 0:
                corners, court_H, court_mask = self._detect_court(frame)
                if court_H is not None:
                    homography = court_H
                    print("[OK] Court detected, homography computed")
                else:
                    print("[WARN] Court detection failed — no homography.")
                    homography = None

            # ── Player detection ──
            players = self._detect_players(frame)
            all_player_positions.append(players)

            # ── Ball detection ──
            if self.ball_model_type == "yolo":
                ball_pos = self._detect_ball_yolo(frame)
            else:
                frame_buffer.append(frame)
                ball_pos = self._detect_ball_tracknet(frame_buffer)
                # Keep buffer bounded
                if len(frame_buffer) > BALL_MODEL_TRACKNET["num_input_frames"] + 5:
                    frame_buffer.pop(0)

            ball_pos["frame"] = frame_idx
            ball_positions.append(ball_pos)

            # ── Speed computation ──
            current_speed = None
            if homography is not None and len(ball_positions) >= 2:
                speeds = compute_ball_speed(ball_positions[-2:], homography, fps)
                if speeds:
                    ball_speeds.append(speeds[-1])
                    key = "speed_kmh_smooth" if "speed_kmh_smooth" in speeds[-1] else "speed_kmh"
                    current_speed = speeds[-1][key]

            # ── Serve detection ──
            serve_event = serve_detector.process_frame(
                frame_idx, ball_positions, ball_speeds, players
            )

            # ── Annotate frame ──
            annotated = self._draw_annotations(
                frame, players, ball_pos, ball_positions,
                homography,
                current_speed, serve_event, serve_detector.serve_counts,
            )

            writer.write(annotated)

            if show_preview:
                cv2.imshow("Pickleball Analysis", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0

        # ── Compile results ──
        serve_summary = serve_detector.get_summary()
        player_stats = compute_player_stats(all_player_positions, fps)

        ball_detected = sum(1 for p in ball_positions if p["x"] is not None)
        max_speed = get_max_speed(ball_speeds) if ball_speeds else {"speed_kmh": 0}

        results = {
            "video": str(video_path),
            "output": output_path,
            "frames_processed": frame_idx,
            "processing_fps": round(proc_fps, 1),
            "elapsed_seconds": round(elapsed, 1),
            "ball_detection_rate": round(ball_detected / max(frame_idx, 1), 3),
            "max_ball_speed_kmh": round(max_speed.get("speed_kmh", 0), 1),
            "serves": serve_summary,
            "player_stats": {str(k): v for k, v in player_stats.items()},
        }

        # Save results JSON
        results_path = Path(output_path).with_suffix(".json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("  Results Summary")
        print("=" * 60)
        print(f"  Frames processed : {frame_idx}")
        print(f"  Processing speed : {proc_fps:.1f} FPS ({elapsed:.1f}s)")
        print(f"  Ball detection   : {ball_detected}/{frame_idx} ({ball_detected/max(frame_idx,1):.1%})")
        print(f"  Max ball speed   : {max_speed.get('speed_kmh', 0):.1f} km/h")
        print(f"  Total serves     : {serve_summary['total_serves']}")
        print(f"  Serves per player: {serve_summary['serves_per_player']}")
        print(f"  Output video     : {output_path}")
        print(f"  Results JSON     : {results_path}")
        print("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Pickleball Video Analysis Pipeline — track ball, players, court; "
                    "compute ball speed and count serves per player.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py match.mp4
  python pipeline.py match.mp4 --ball-model yolo --show
  python pipeline.py match.mp4 --ball-model tracknet --tracknet-weights weights/tracknet/tracknet_best.pth
  python pipeline.py match.mp4 --output results/annotated.mp4
        """,
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("-o", "--output", default=None, help="Output video path")
    parser.add_argument("--ball-model", choices=["yolo", "tracknet"], default="yolo",
                        help="Ball detection model (default: yolo)")
    parser.add_argument("--player-weights", default=None, help="Player detector weights")
    parser.add_argument("--ball-weights", default=None, help="Ball YOLO weights")
    parser.add_argument("--tracknet-weights", default=None, help="TrackNet weights")
    parser.add_argument("--court-weights", default=None, help="Court segmentation model weights")
    parser.add_argument("--device", default="0", help="Device (0, 1, cpu)")
    parser.add_argument("--show", action="store_true", help="Show live preview")

    args = parser.parse_args()

    pipeline = PickleballAnalysisPipeline(
        player_weights=args.player_weights,
        ball_model_type=args.ball_model,
        ball_yolo_weights=args.ball_weights,
        tracknet_weights=args.tracknet_weights,
        court_segment_weights=args.court_weights,
        device=args.device,
    )

    results = pipeline.process_video(args.video, args.output, args.show)
    return results


if __name__ == "__main__":
    main()
