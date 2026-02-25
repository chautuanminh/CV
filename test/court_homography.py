"""
Test script: Court Homography via YOLO Segmentation
====================================================
Uses the shared CourtHomographyDetector from models/court_homography.py.

Usage:
    python test/court_homography.py <image> [--model models/court_segment.pt]
    python test/court_homography.py <image> --output-dir results/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.court_homography import CourtHomographyDetector


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Court Segmentation → Homography-Based Court Calibration"
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "court_segment.pt"),
        help="Path to the YOLO segmentation model (default: models/court_segment.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Segmentation confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save outputs (default: same folder as input)")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(args.image) or "."
    os.makedirs(out_dir, exist_ok=True)
    basename = Path(args.image).stem

    # ── Load image ───────────────────────────────────────────────────
    original = cv2.imread(args.image)
    if original is None:
        raise FileNotFoundError(f"Cannot load: {args.image}")

    # ── Create detector ──────────────────────────────────────────────
    detector = CourtHomographyDetector(model_path=args.model, conf=args.conf)

    # ── 1. Detect court ──────────────────────────────────────────────
    print(f"[1/4] YOLO segmentation:  {args.image}")
    corners, H, mask_binary = detector.detect(original)

    if mask_binary is not None:
        mask_vis = detector.draw_mask_overlay(original, mask_binary)
        seg_path = os.path.join(out_dir, f"{basename}_1_segmentation.jpg")
        cv2.imwrite(seg_path, mask_vis)
        print(f"      Mask overlay  → {seg_path}")
    else:
        print("      [WARN] No court mask detected.")

    # ── 2. Corners ───────────────────────────────────────────────────
    if corners is not None:
        print("[2/4] Extracting 4 court corners from mask contour …")
        corner_vis = detector.draw_corners(original, corners)
        corner_path = os.path.join(out_dir, f"{basename}_2_corners.jpg")
        cv2.imwrite(corner_path, corner_vis)
        print(f"      Corners image → {corner_path}")
        for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
            print(f"        {lbl}: ({int(pt[0])}, {int(pt[1])})")
    else:
        print("[2/4] Corner extraction failed — aborting.")
        return

    # ── 3. Homography ────────────────────────────────────────────────
    if H is not None:
        import numpy as np
        print("[3/4] Computing homography …")
        print(f"      H =\n{np.round(H, 4)}")
    else:
        print("[3/4] Homography computation failed — aborting.")
        return

    # ── 4. Draw court lines ──────────────────────────────────────────
    print("[4/4] Projecting standard court lines onto image …")
    result = detector.project_court_lines(original, H)
    result_path = os.path.join(out_dir, f"{basename}_3_homography_lines.jpg")
    cv2.imwrite(result_path, result)
    print(f"      Final result  → {result_path}")

    print("\n✅ Homography calibration complete!")
    print(f"   Segmentation : {seg_path}")
    print(f"   Corners      : {corner_path}")
    print(f"   Court lines  : {result_path}")


if __name__ == "__main__":
    main()
