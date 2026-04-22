#!/usr/bin/env python3
"""Combine N videos into an NxM grid with labels.

Usage:
    python tools/grid_video.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 -o grid.mp4
    python tools/grid_video.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 -o grid.mp4 --cols 2
    python tools/grid_video.py video1.mp4 video2.mp4 -o grid.mp4 --labels "Dry μ=1.1" "Icy μ=0.3"
"""
import argparse
import math
import sys

import imageio.v2 as imageio
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def add_label(frame: np.ndarray, text: str, font_size: int = 24) -> np.ndarray:
    """Burn a text label into the top-left of the frame."""
    if not HAS_PIL or not text:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # Draw text with black outline for readability
    x, y = 8, 6
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Combine videos into an NxM grid.")
    parser.add_argument("inputs", nargs="+", help="Input video files")
    parser.add_argument("-o", "--output", default="grid.mp4", help="Output video path")
    parser.add_argument("--cols", type=int, default=0, help="Number of columns (0=auto square)")
    parser.add_argument("--labels", nargs="*", default=None, help="Per-video labels")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--font_size", type=int, default=24, help="Label font size")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames (0=all)")
    args = parser.parse_args()

    n = len(args.inputs)
    cols = args.cols if args.cols > 0 else math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    labels = args.labels or [None] * n

    # Open all readers
    readers = [imageio.get_reader(p) for p in args.inputs]

    # Get cell size from first frame of first video
    first_frame = readers[0].get_data(0)
    cell_h, cell_w = first_frame.shape[:2]

    grid_h = rows * cell_h
    grid_w = cols * cell_w

    writer = imageio.get_writer(args.output, fps=args.fps)
    print(f"Grid: {rows}x{cols}  cell: {cell_w}x{cell_h}  output: {grid_w}x{grid_h}")

    frame_idx = 0
    try:
        while True:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            any_frame = False
            for i, reader in enumerate(readers):
                try:
                    frame = reader.get_data(frame_idx)
                    any_frame = True
                except (IndexError, Exception):
                    frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                # Resize if needed
                if frame.shape[0] != cell_h or frame.shape[1] != cell_w:
                    if HAS_PIL:
                        frame = np.array(Image.fromarray(frame).resize((cell_w, cell_h)))
                    else:
                        frame = frame[:cell_h, :cell_w]
                # Ensure 3 channels
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                # Add label
                if i < len(labels) and labels[i]:
                    frame = add_label(frame, labels[i], font_size=args.font_size)
                r, c = divmod(i, cols)
                grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = frame

            if not any_frame:
                break
            writer.append_data(grid)
            frame_idx += 1
    finally:
        writer.close()
        for r in readers:
            r.close()

    print(f"Wrote {frame_idx} frames to {args.output}")


if __name__ == "__main__":
    main()
