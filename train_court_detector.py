"""
Train the court keypoint detection model (ResNet50 backbone).

Predicts 14 normalized (x,y) keypoints used for homography computation.

Usage:
    python train_court_detector.py train --images-dir data/court_images --annotations data/court_keypoints.json
    python train_court_detector.py predict --weights weights/court/court_best.pth --image frame.jpg
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
from config import COURT_MODEL, WEIGHTS_DIR
from models.court_keypoint import CourtKeypointModel
from data.dataset import CourtKeypointDataset


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = COURT_MODEL

    print("=" * 60)
    print("  Court Keypoint Detection — ResNet50 Training")
    print("=" * 60)
    print(f"Device: {device}")

    # Dataset
    dataset = CourtKeypointDataset(
        images_dir=args.images_dir,
        annotations_file=args.annotations,
        input_size=cfg["input_size"],
        augment=True,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    print(f"Train: {train_size}  Val: {val_size}")

    # Model
    model = CourtKeypointModel(
        num_keypoints=cfg["num_keypoints"],
        pretrained=True,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    save_dir = WEIGHTS_DIR / "court"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, keypoints in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False):
            images, keypoints = images.to(device), keypoints.to(device)
            pred = model(images)
            loss = criterion(pred, keypoints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_errors = []
        with torch.no_grad():
            for images, keypoints in val_loader:
                images, keypoints = images.to(device), keypoints.to(device)
                pred = model(images)
                loss = criterion(pred, keypoints)
                val_loss += loss.item()

                # Mean keypoint error in pixels (assuming 640×360 output)
                pred_np = pred.cpu().numpy().reshape(-1, cfg["num_keypoints"], 2)
                gt_np = keypoints.cpu().numpy().reshape(-1, cfg["num_keypoints"], 2)
                pred_np[:, :, 0] *= 640
                pred_np[:, :, 1] *= 360
                gt_np[:, :, 0] *= 640
                gt_np[:, :, 1] *= 360
                errors = np.linalg.norm(pred_np - gt_np, axis=2).mean(axis=1)
                val_errors.extend(errors.tolist())

        val_loss /= len(val_loader)
        mean_kp_error = np.mean(val_errors)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | train_loss: {train_loss:.6f} | "
                  f"val_loss: {val_loss:.6f} | kp_error: {mean_kp_error:.2f}px")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "court_best.pth")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Weights: {save_dir / 'court_best.pth'}")


def predict(args):
    """Predict keypoints on a single image."""
    import cv2
    from torchvision import transforms

    device = torch.device("cpu")
    cfg = COURT_MODEL

    model = CourtKeypointModel(num_keypoints=cfg["num_keypoints"], pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    image = cv2.imread(args.image)
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(rgb).unsqueeze(0).to(device)

    keypoints = model.predict_keypoints(tensor, w, h)

    # Draw keypoints on image
    for i, (kx, ky) in enumerate(keypoints):
        cv2.circle(image, (int(kx), int(ky)), 5, (0, 0, 255), -1)
        cv2.putText(image, str(i), (int(kx) + 8, int(ky) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    output_path = Path(args.image).stem + "_keypoints.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image: {output_path}")
    print(f"Keypoints:\n{keypoints}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Court keypoint detector")
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Train court model")
    t.add_argument("--images-dir", required=True)
    t.add_argument("--annotations", required=True)
    t.add_argument("--device", default="cuda:0")

    p = sub.add_parser("predict", help="Predict on image")
    p.add_argument("--image", required=True)
    p.add_argument("--weights", required=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()
