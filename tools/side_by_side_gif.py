#!/usr/bin/env python3
"""Composite two GIFs side by side, and measure how different they are.

This is **not** a renderer: every skeleton pixel comes from
``~/dev/motion-bricks-cpp/tools/render_motion.py``, which paints each arm's plan
document on its own.  This tool only pastes the two rendered frame sequences next
to each other (with a caption band naming the arms) so a reader can see both at
once, and it reports the exact pixel difference it measured while doing so --
"they look the same" is then a number, not an impression.

    python3 tools/side_by_side_gif.py --left a.gif --right b.gif \
        --left-label "BATCH B=16" --right-label "SINGLE B=1" --out pair.gif
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG = (14, 17, 22)
BAND = (24, 30, 38)
TEXT = (198, 208, 220)
LABEL = (86, 214, 214)
ALERT = (255, 150, 40)


def read_frames(path: Path) -> tuple[list[np.ndarray], int]:
    """Every frame of *path* as RGB arrays, plus the frame duration in ms."""
    images = []
    with Image.open(path) as image:
        count = getattr(image, "n_frames", 1)
        duration = int(image.info.get("duration", 33))
        for index in range(count):
            image.seek(index)
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
    return images, duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--band", type=int, default=34)
    args = parser.parse_args()

    left, duration = read_frames(args.left)
    right, right_duration = read_frames(args.right)
    if len(left) != len(right):
        raise SystemExit(f"frame counts differ: {len(left)} vs {len(right)}")
    if left[0].shape != right[0].shape:
        raise SystemExit(f"frame sizes differ: {left[0].shape} vs {right[0].shape}")

    height, width, _ = left[0].shape
    canvas_width = width * 2 + args.gap
    frame_height = height + args.band
    rendered = []
    for index, (a, b) in enumerate(zip(left, right)):
        canvas = Image.new("RGB", (canvas_width, frame_height), BG)
        canvas.paste(Image.fromarray(a), (0, args.band))
        canvas.paste(Image.fromarray(b), (width + args.gap, args.band))
        pixels = a.shape[0] * a.shape[1]
        changed = int(np.count_nonzero(
            np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas_width, args.band - 1], fill=BAND)
        draw.text((10, 4), args.left_label, fill=LABEL)
        draw.text((width + args.gap + 10, 4), args.right_label, fill=LABEL)
        draw.text((10, args.band - 14),
                  f"frame {index + 1}/{len(left)}   differing pixels: {changed}/{pixels}",
                  fill=TEXT if changed == 0 else ALERT)
        rendered.append(canvas)

    rendered[0].save(args.out, save_all=True, append_images=rendered[1:],
                     duration=duration or right_duration, loop=0, optimize=True)

    diffs = [np.abs(a.astype(np.int16) - b.astype(np.int16)) for a, b in zip(left, right)]
    changed = sum(int(np.count_nonzero(d.max(axis=2))) for d in diffs)
    total = sum(d.shape[0] * d.shape[1] for d in diffs)
    print(f"{args.out}  frames={len(rendered)}  {canvas_width}x{frame_height}  "
          f"{args.out.stat().st_size:,} bytes")
    print(f"  pixels compared {total:,}  differing {changed:,} "
          f"({100.0 * changed / total:.6f}%)  "
          f"max channel delta {max(int(d.max()) for d in diffs)}")
    print(f"  sha256 left  {hashlib.sha256(args.left.read_bytes()).hexdigest()}")
    print(f"  sha256 right {hashlib.sha256(args.right.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
