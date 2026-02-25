"""
Pipeline: YOLO Court Segmentation → Crop → Hough Line Detection
=================================================================
1. Run the YOLO segmentation model to detect the court region.
2. Crop (or mask) just the court portion from the original image.
3. Pass the cropped court image through the Hough line detector.

Usage:
    python test/segment_then_detect.py <image> [--model models/court_segment.pt]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Allow importing the existing court_line_detection module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from court_line_detection import load_and_grayscale, detect_court_lines, draw_lines


def segment_court(model_path: str, image_path: str, conf: float = 0.25):
    """
    Run YOLO segmentation to get the court mask.

    Returns:
        original  : the original BGR image
        court_crop: the image cropped to the court bounding box with background blacked out
        bbox      : (x1, y1, x2, y2) bounding box of the court region
        mask      : binary mask of the court (same size as original)
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, save=False, verbose=False)

    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = original.shape[:2]
    result = results[0]

    if result.masks is None or len(result.masks) == 0:
        print("[WARN] No court segmentation mask detected. Falling back to full image.")
        return original, original.copy(), (0, 0, w, h), np.ones((h, w), dtype=np.uint8) * 255

    # Pick the mask with the highest confidence
    best_idx = int(result.boxes.conf.argmax())
    mask_data = result.masks.data[best_idx].cpu().numpy()  # (mask_h, mask_w)

    # Resize mask to original image size
    mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > 0.5).astype(np.uint8) * 255

    # Bounding box of the mask
    ys, xs = np.where(mask_binary > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    # Apply mask to isolate the court, then crop to bbox
    masked_image = cv2.bitwise_and(original, original, mask=mask_binary)
    court_crop = masked_image[y1:y2, x1:x2]

    return original, court_crop, (x1, y1, x2, y2), mask_binary


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Court Segmentation → Hough Line Detection pipeline"
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "court_segment.pt"),
        help="Path to YOLO segmentation model (default: models/court_segment.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Segmentation confidence threshold")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=80)
    parser.add_argument("--min-line-length", type=int, default=80)
    parser.add_argument("--max-line-gap", type=int, default=15)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(args.image) or "."
    os.makedirs(out_dir, exist_ok=True)
    basename = Path(args.image).stem

    # ── Step 1: YOLO Segmentation ────────────────────────────────────
    print(f"[1/5] Running YOLO segmentation on: {args.image}")
    print(f"      Model: {args.model}")
    original, court_crop, bbox, mask = segment_court(args.model, args.image, args.conf)
    x1, y1, x2, y2 = bbox
    print(f"      Court bbox: ({x1}, {y1}) → ({x2}, {y2})")
    print(f"      Crop size : {court_crop.shape[1]}×{court_crop.shape[0]}")

    # Save the segmentation mask overlay
    mask_overlay = original.copy()
    mask_colored = np.zeros_like(original)
    mask_colored[:, :, 1] = mask  # green channel
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
    cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mask_path = os.path.join(out_dir, f"{basename}_segmentation.jpg")
    cv2.imwrite(mask_path, mask_overlay)
    print(f"      Segmentation overlay → {mask_path}")

    # ── Step 2: Save the cropped court ───────────────────────────────
    crop_path = os.path.join(out_dir, f"{basename}_court_crop.jpg")
    cv2.imwrite(crop_path, court_crop)
    print(f"[2/5] Cropped court saved → {crop_path}")

    # ── Step 3: Grayscale ────────────────────────────────────────────
    gray = cv2.cvtColor(court_crop, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(out_dir, f"{basename}_court_grayscale.jpg")
    cv2.imwrite(gray_path, gray)
    print(f"[3/5] Grayscale court saved → {gray_path}")

    # ── Step 4: Hough line detection on the cropped court ────────────
    print("[4/5] Running Canny + Hough Transform on cropped court …")
    edges, lines = detect_court_lines(
        gray,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
    )
    num_lines = 0 if lines is None else len(lines)
    print(f"      Detected {num_lines} line segments")

    edges_path = os.path.join(out_dir, f"{basename}_court_edges.jpg")
    cv2.imwrite(edges_path, edges)

    # ── Step 5: Draw lines on cropped court ──────────────────────────
    result_crop = draw_lines(court_crop, lines, color=(0, 0, 255), thickness=2)
    lines_path = os.path.join(out_dir, f"{basename}_court_hough_lines.jpg")
    cv2.imwrite(lines_path, result_crop)

    # Also project lines back onto the original image
    result_full = original.copy()
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            cv2.line(result_full, (lx1 + x1, ly1 + y1), (lx2 + x1, ly2 + y1),
                     (0, 0, 255), 2)
    full_path = os.path.join(out_dir, f"{basename}_full_hough_lines.jpg")
    cv2.imwrite(full_path, result_full)

    print(f"[5/5] Results saved:")
    print(f"      Segmentation  : {mask_path}")
    print(f"      Court crop    : {crop_path}")
    print(f"      Grayscale     : {gray_path}")
    print(f"      Edge map      : {edges_path}")
    print(f"      Hough (crop)  : {lines_path}")
    print(f"      Hough (full)  : {full_path}")
    print(f"\n✅ Pipeline complete — {num_lines} court lines detected.")


if __name__ == "__main__":
    main()
