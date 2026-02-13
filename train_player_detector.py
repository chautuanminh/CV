"""
Train the player detection model using YOLOv8.

This script fine-tunes a pre-trained YOLOv8 model on a tennis-specific
player detection dataset. Players are detected with bounding boxes
and tracked across frames using BotSORT.

Usage:
    python train_player_detector.py
    python train_player_detector.py --epochs 50 --batch 8 --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PLAYER_MODEL, WEIGHTS_DIR


def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("  Player Detection — YOLOv8 Training")
    print("=" * 60)

    # Load base model
    model = YOLO(args.model or PLAYER_MODEL["base_weights"])
    print(f"Base model : {args.model or PLAYER_MODEL['base_weights']}")
    print(f"Dataset    : {args.data}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}")
    print(f"Batch size : {args.batch}")
    print(f"Device     : {args.device}")
    print()

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=PLAYER_MODEL["optimizer"],
        lr0=PLAYER_MODEL["lr"],
        project=str(WEIGHTS_DIR),
        name="player_detector",
        exist_ok=True,
        verbose=True,
    )

    # Validate
    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    val_results = model.val()
    print(f"  mAP@0.5     : {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")

    # Save best weights path
    best_path = WEIGHTS_DIR / "player_detector" / "weights" / "best.pt"
    print(f"\nBest weights saved to: {best_path}")
    return str(best_path)


def test_track(args):
    """Quick test: run tracking on a video clip."""
    from ultralytics import YOLO

    model_path = args.weights or str(WEIGHTS_DIR / "player_detector" / "weights" / "best.pt")
    model = YOLO(model_path)

    print(f"Running player tracking on: {args.video}")
    results = model.track(
        source=args.video,
        tracker=PLAYER_MODEL["tracker"],
        persist=True,
        conf=PLAYER_MODEL["confidence_threshold"],
        classes=[0],  # person class in COCO
        show=args.show,
        save=True,
        project=str(WEIGHTS_DIR / "player_detector"),
        name="track_test",
    )
    print("Tracking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 player detector")
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # Train command
    train_parser = sub.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", default=PLAYER_MODEL["dataset_yaml"],
                              help="Path to dataset YAML")
    train_parser.add_argument("--model", default=None, help="Base model weights")
    train_parser.add_argument("--epochs", type=int, default=PLAYER_MODEL["epochs"])
    train_parser.add_argument("--imgsz", type=int, default=PLAYER_MODEL["imgsz"])
    train_parser.add_argument("--batch", type=int, default=PLAYER_MODEL["batch_size"])
    train_parser.add_argument("--device", default="0", help="Device: 0, 1, cpu")

    # Track test command
    track_parser = sub.add_parser("track", help="Test tracking on a video")
    track_parser.add_argument("video", help="Path to video file")
    track_parser.add_argument("--weights", default=None, help="Model weights path")
    track_parser.add_argument("--show", action="store_true", help="Display live")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "track":
        test_track(args)
    else:
        parser.print_help()
