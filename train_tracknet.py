"""
Train the TrackNet model for ball detection via heatmap regression.

Uses consecutive frames as input and predicts a 2D Gaussian heatmap
at the ball's center position. Higher accuracy than YOLO for fast,
small, blurry balls — but requires center-point annotations.

Usage:
    python train_tracknet.py train --frames-dir data/frames --annotations data/ball_annotations.json
    python train_tracknet.py detect --weights weights/tracknet_best.pth --video match.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BALL_MODEL_TRACKNET, WEIGHTS_DIR
from models.tracknet import TrackNet
from data.dataset import TrackNetDataset


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  TrackNet — Ball Heatmap Regression Training")
    print("=" * 60)
    print(f"Device: {device}")

    cfg = BALL_MODEL_TRACKNET

    # Dataset
    dataset = TrackNetDataset(
        frames_dir=args.frames_dir,
        annotations_file=args.annotations,
        num_frames=cfg["num_input_frames"],
        input_height=cfg["input_height"],
        input_width=cfg["input_width"],
        sigma=cfg["gaussian_sigma"],
    )

    # Train / val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    print(f"Train samples: {train_size}  Val samples: {val_size}")

    # Model
    model = TrackNet(num_input_frames=cfg["num_input_frames"]).to(device)
    criterion = nn.BCELoss()  # Binary cross-entropy for heatmap
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float("inf")
    save_dir = WEIGHTS_DIR / "tracknet"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for frames, heatmaps in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]", leave=False):
            frames, heatmaps = frames.to(device), heatmaps.to(device)
            pred = model(frames)
            loss = criterion(pred, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, heatmaps in val_loader:
                frames, heatmaps = frames.to(device), heatmaps.to(device)
                pred = model(frames)
                loss = criterion(pred, heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "tracknet_best.pth")

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_dir / f"tracknet_epoch{epoch}.pth")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Best weights: {save_dir / 'tracknet_best.pth'}")


def detect_video(args):
    """Run TrackNet inference on a video."""
    import cv2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = BALL_MODEL_TRACKNET

    model = TrackNet(num_input_frames=cfg["num_input_frames"]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    buffer = []
    positions = []

    print(f"Running TrackNet on {total} frames...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (cfg["input_width"], cfg["input_height"]))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = resized.astype(np.float32) / 255.0
        buffer.append(resized)

        if len(buffer) == cfg["num_input_frames"]:
            # Stack frames
            stacked = np.concatenate(buffer, axis=2)  # (H, W, 9)
            tensor = torch.tensor(stacked).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmap = model(tensor)
            pos = model.detect_ball(heatmap, threshold=cfg["heatmap_threshold"])

            # Scale back to original resolution
            if pos is not None:
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                sx = orig_w / cfg["input_width"]
                sy = orig_h / cfg["input_height"]
                pos = (int(pos[0] * sx), int(pos[1] * sy))

            positions.append({"frame": frame_idx, "x": pos[0] if pos else None, "y": pos[1] if pos else None})
            buffer.pop(0)

        frame_idx += 1

    cap.release()

    import pandas as pd
    df = pd.DataFrame(positions)
    output = Path(args.video).stem + "_tracknet_positions.csv"
    df.to_csv(output, index=False)
    detected = df["x"].notna().sum()
    print(f"Detected ball in {detected}/{len(df)} frames ({detected/len(df):.1%})")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNet ball detection")
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Train TrackNet")
    t.add_argument("--frames-dir", required=True, help="Directory with extracted frames")
    t.add_argument("--annotations", required=True, help="Ball annotations JSON")
    t.add_argument("--device", default="cuda:0")

    d = sub.add_parser("detect", help="Run detection on video")
    d.add_argument("--video", required=True)
    d.add_argument("--weights", required=True)
    d.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "detect":
        detect_video(args)
    else:
        parser.print_help()
