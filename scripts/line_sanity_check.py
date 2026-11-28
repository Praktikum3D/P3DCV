"""Does the plumb line residual really know the right camera? One real fisheye frame, two cameras.

Detects lines on a TartanAir `lcam_fish` frame and evaluates `plumbline_residual` with the true
camera of that frame (equidistant, 195 degrees, f = 1000 / rad(195) = 293.8 px at 1000 x 1000) and
with the focal length multiplied by a factor, 1.4 by default, the same mistake that made the
pinhole run of 26.11. collapse. The residual at the true camera must come out clearly lower, and
its gradient must point back towards the true focal length. That is the entire signal the line
initialiser, the line loss and the per scene refinement live on, so it is worth one command.

Both normalisations are reported, and with `--sweep` a whole range of focal factors, which is what
shows that the trace normalisation has no minimum at all while the in plane one does. Pass
`--no-chain` to see the same numbers on raw detector segments, which have no signal either.

The lines are drawn on the frame and written next to the numbers.

Run inside the container:
    python scripts/line_sanity_check.py \
        --frame /praktikum/data-sample/AmericanDiner/Data_hard/P000/image_lcam_fish/000000_lcam_fish_image.png \
        --out /praktikum/renders/wochenende/line_detection_sample.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.camera_augmentation import FISH_F, FISH_FOV_DEG, FISH_SIZE  # noqa: E402
from training.line_detection import (  # noqa: E402
    DEFAULT_MAX_COUNT,
    DEFAULT_MIN_LENGTH,
    DEFAULT_NUM_SAMPLES,
    detect_line_segments,
    draw_segments,
)
from training.line_geometry import DEFAULT_TRIM, plumbline_residual_mean  # noqa: E402


def image_circle_mask(size: int, radius: float) -> np.ndarray:
    """The pixels the 195 degree fisheye actually images: a circle spanning the image width."""
    y, x = np.mgrid[0:size, 0:size]
    return np.hypot(x + 0.5 - size / 2.0, y + 0.5 - size / 2.0) <= radius


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True, nargs="+", help="one or more square lcam_fish pngs")
    parser.add_argument("--out", default="", help="where to write the line figure")
    parser.add_argument("--size", type=int, default=FISH_SIZE, help="rescale the frame to this before detecting")
    parser.add_argument("--wrong-scale", type=float, default=1.4, help="focal length factor of the wrong camera")
    parser.add_argument("--min-length", type=float, default=DEFAULT_MIN_LENGTH,
                        help="fraction of the image side, or pixels when 1 or above")
    parser.add_argument("--max-count", type=int, default=DEFAULT_MAX_COUNT)
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--detector", default="auto")
    parser.add_argument("--no-chain", action="store_true", help="use the raw detector segments")
    parser.add_argument("--sweep", action="store_true", help="also print a focal length sweep")
    args = parser.parse_args()

    size = int(args.size)
    scale = 1.0
    frames, found, elapsed = [], [], 0.0
    for path in args.frame:
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if image.shape[0] != image.shape[1]:
            raise SystemExit(f"expected a square fisheye frame, got {image.shape[:2]}")
        scale = size / image.shape[0]
        if scale != 1.0:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        started = time.perf_counter()
        lines = detect_line_segments(
            image,
            mask=image_circle_mask(size, size / 2.0 - 2.0),
            min_length=args.min_length,
            max_count=args.max_count,
            num_samples=args.samples,
            detector=args.detector,
            chain=not args.no_chain,
        )
        elapsed += (time.perf_counter() - started) * 1000.0
        frames.append(image)
        found.append(lines)

    if not any(len(lines) for lines in found):
        raise SystemExit("no lines found, nothing to compare")

    # The true camera of lcam_fish, scaled with the image: equidistant, f pixels per radian.
    f_true = FISH_F * scale
    centre = size / 2.0
    pixels = torch.from_numpy(np.concatenate([lines.pixels for lines in found if len(lines)]))
    scores = np.concatenate([lines.score for lines in found if len(lines)])
    print(f"frames     {len(args.frame)}, first {args.frame[0]}")
    print(f"size       {size} x {size}, {FISH_FOV_DEG} degree equidistant, f_true {f_true:.2f} px")
    print(f"detection  {'chained' if not args.no_chain else 'raw segments'}, "
          f"{elapsed / len(args.frame):.1f} ms per frame")
    print(f"lines      {len(pixels)}, lengths {scores.min():.1f} to {scores.max():.1f} px")

    def evaluate(factor, normalize, trim=DEFAULT_TRIM):
        intrinsics = torch.tensor([f_true * factor, f_true * factor, centre, centre], dtype=torch.float32)
        intrinsics.requires_grad_(True)
        value = plumbline_residual_mean(intrinsics, "fisheye", pixels, normalize=normalize, trim=trim)
        value.backward()
        return float(value), float(intrinsics.grad[0])

    print()
    print(f"{'normalize':>10} {'camera':>8} {'f [px]':>10} {'trimmed mean':>14} "
          f"{'plain mean':>12} {'d / d f':>12}")
    for normalize in ("inplane", "trace"):
        for name, factor in (("true", 1.0), (f"x{args.wrong_scale:g}", float(args.wrong_scale))):
            trimmed, grad = evaluate(factor, normalize)
            plain = evaluate(factor, normalize, trim=0.0)[0]
            print(f"{normalize:>10} {name:>8} {f_true * factor:10.2f} {trimmed:14.3e} "
                  f"{plain:12.3e} {grad:12.3e}")
        true_mean = evaluate(1.0, normalize)[0]
        wrong_mean, wrong_grad = evaluate(float(args.wrong_scale), normalize)
        verdict = "towards the true f" if wrong_grad > 0 else "AWAY from the true f"
        print(f"{'':>10} wrong / true = {wrong_mean / max(true_mean, 1e-20):6.2f}, "
              f"the gradient at the wrong camera points {verdict}")

    if args.sweep:
        factors = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.8, 2.5]
        print()
        for normalize in ("inplane", "trace"):
            values = [evaluate(factor, normalize)[0] for factor in factors]
            best = factors[int(np.argmin(values))]
            print(f"{normalize:>10} sweep, minimum at f x {best:.2f}")
            print("           " + "  ".join(f"{f:.2f}:{v:.1e}" for f, v in zip(factors, values)))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # Two panels of the first frame: what the detector reports and what survives the linking.
        raw = detect_line_segments(
            frames[0], mask=image_circle_mask(size, size / 2.0 - 2.0),
            min_length=0.02, max_count=10000, num_samples=2, chain=False,
        )
        left = draw_segments(frames[0], raw.pixels, color=(255, 196, 32))
        right = draw_segments(frames[0], found[0].pixels, color=(255, 32, 32))
        for canvas, caption in ((left, f"{len(raw)} raw segments"), (right, f"{len(found[0])} linked lines")):
            cv2.circle(canvas, (size // 2, size // 2), int(size / 2 - 2), (64, 200, 64), 1)
            cv2.putText(canvas, caption, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * size / 500.0, (255, 255, 255), max(1, size // 500))
        figure = np.concatenate([left, np.full((size, 4, 3), 255, np.uint8), right], axis=1)
        cv2.imwrite(args.out, cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
        print(f"\nfigure     {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
