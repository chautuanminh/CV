"""
Court Line Detection using Hough Transform
============================================
1. Converts input image to grayscale (black and white)
2. Applies Canny edge detection
3. Uses Hough Line Transform to detect court lines
4. Visualises the results side by side
"""

import cv2
import numpy as np
import argparse
import os


def load_and_grayscale(image_path: str):
    """Load an image and convert to grayscale."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_court_lines(gray: np.ndarray, 
                       canny_low: int = 50, 
                       canny_high: int = 150,
                       hough_threshold: int = 100,
                       min_line_length: int = 100,
                       max_line_gap: int = 10):
    """
    Detect court lines using Canny + Probabilistic Hough Transform.
    
    Returns:
        edges: the Canny edge map
        lines: array of detected line segments (x1, y1, x2, y2)
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return edges, lines


def draw_lines(image: np.ndarray, lines, color=(0, 0, 255), thickness=2):
    """Draw detected lines on a copy of the image."""
    output = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
    return output


def main():
    parser = argparse.ArgumentParser(description="Detect pickleball court lines with Hough Transform")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny lower threshold (default: 50)")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny upper threshold (default: 150)")
    parser.add_argument("--hough-threshold", type=int, default=100, help="Hough accumulator threshold (default: 100)")
    parser.add_argument("--min-line-length", type=int, default=100, help="Minimum line length in pixels (default: 100)")
    parser.add_argument("--max-line-gap", type=int, default=10, help="Maximum gap between line segments (default: 10)")
    parser.add_argument("--output-dir", default=None, help="Directory to save results (default: same as input)")
    args = parser.parse_args()

    # ── 1. Load & grayscale ──────────────────────────────────────────
    print(f"[1/4] Loading image: {args.image}")
    original, gray = load_and_grayscale(args.image)
    print(f"      Image size: {original.shape[1]}x{original.shape[0]}")

    # ── 2. Save grayscale ────────────────────────────────────────────
    out_dir = args.output_dir or os.path.dirname(args.image) or "."
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.image))[0]
    gray_path = os.path.join(out_dir, f"{basename}_grayscale.jpg")
    cv2.imwrite(gray_path, gray)
    print(f"[2/4] Grayscale saved → {gray_path}")

    # ── 3. Hough line detection ──────────────────────────────────────
    print("[3/4] Running Canny + Hough Line Transform …")
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

    edges_path = os.path.join(out_dir, f"{basename}_edges.jpg")
    cv2.imwrite(edges_path, edges)
    print(f"      Edge map saved → {edges_path}")

    # ── 4. Draw & save result ────────────────────────────────────────
    result = draw_lines(original, lines, color=(0, 0, 255), thickness=2)
    result_path = os.path.join(out_dir, f"{basename}_court_lines.jpg")
    cv2.imwrite(result_path, result)
    print(f"[4/4] Court lines overlay saved → {result_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n✅ Done! Output files:")
    print(f"   Grayscale   : {gray_path}")
    print(f"   Edge map    : {edges_path}")
    print(f"   Court lines : {result_path}")


if __name__ == "__main__":
    main()
