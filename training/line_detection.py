"""Find the pixels of straight 3D lines in an image, which on a wide angle image are curves.

This is the detection half of the line infrastructure; `training/line_geometry.py` holds the
differentiable half. Three consumers share both: a focal length and alpha initialiser, a plumb line
loss during training, and a per scene refinement after training.

What a straight 2D segment means on a wide angle image
------------------------------------------------------
A straight 3D line projects to a straight image line only under a pinhole camera. Under a fisheye
or a (E)UCM camera it projects to a CURVE, and the wider the camera, the stronger the bend. A
straight segment detector run on such an image therefore does not find whole lines. It finds short,
locally straight pieces of those curves, plus whatever is genuinely straight near the optical axis
where the distortion is weak.

Each such piece is a legitimate constraint. The plumb line condition is not a statement about
straightness in the image at all: the viewing rays of ANY set of pixels that image one and the same
3D line are coplanar with the projection centre, and that holds for a two pixel piece of a curve
just as well as for the whole curve. So short segments give no WRONG constraint.

They give almost no constraint either, and that is the point worth knowing before using this
module. A segment is reported by the detector precisely BECAUSE it is straight in the image, and a
straight image segment is exactly coplanar under a pinhole camera. The residual of such a segment
therefore falls monotonically as the focal length grows and the camera approaches a pinhole: on a
real 1000 x 1000 `lcam_fish` frame the mean residual over 64 LSD segments is 4.8e-6 at the true
camera and 8.0e-7 at 1.4 times its focal length, twenty times lower at a camera that is badly
wrong. Selection bias, not geometry. Plain LSD segments cannot estimate a camera.

Chaining
--------
The cure is the standard one in plumb line autocalibration: link the short collinear pieces back
into the curve they came from. `link_segments` grows a chain from the longest unused segment,
appending at either end any unused segment whose endpoint is within `max_gap` pixels and whose
direction turns by less than `max_turn_deg`, and it keeps the chain only if it turns consistently
in one direction (a projected 3D line bends one way, it does not wiggle) and if its traced length
reaches `min_length`. The result is a long, gently curved polyline whose pixels belong to one 3D
line, with the curvature left in, which is exactly the information the residual needs. Over eight
real `data-sample` frames the residual over the chains then has its minimum exactly at the true
focal length and rises on both sides, by a factor of 51 at 0.8 times it and 32 at 1.4 times it
(1000 x 1000, `scripts/line_sanity_check.py --sweep`).

Detect on the source, not on the view
-------------------------------------
Resolution decides whether any of this works, and the answer is not the comfortable one. The
residual itself is scale free: detect at 1000 and scale the pixels down to 256, and the curve over
focal lengths is the same to three digits. What is not scale free is the detector. Its endpoint
noise is a fraction of a pixel at any resolution, so in ANGLE it grows as the image shrinks, and at
256 pixels it buries the signal. Measured over eight real frames at the operating point the trained
model sits at today, a field of view of 188 degrees instead of 195, so a focal length 1.0372 times
the true one: detecting on the 256 pixel view the wrong camera scores LOWER than the true one and
the gradient points further away from 195 degrees. A line loss built on 256 pixel detections would
work against the thing it is meant to fix. It is not a weak constraint there, it is an adversarial
one.

The way out costs nothing in correctness: detect on the sharp source frame and map the pixels into
the view analytically with `camera_augmentation.map_fisheye_pixels`, which unprojects with the
known equidistant source law and reprojects with the drawn camera. The lines then sit in exactly
the pixels of the image the model sees, carrying the accuracy of the source. That is what
`TartanAirDataset` does by default; see its DEFAULT_LINE_NATIVE_SIZE for the measured table.

Cost, averaged over eight `data-sample` frames on this GTX 1660 box, detection and linking only:
57 ms for a 256 x 256 view, 218 ms at 512 and about 1 s for a full 1000 x 1000 frame. A busy frame
costs several times a sparse one, so these are an order of magnitude and not a promise.

A chain is still reported as a `LineSegments` row: `endpoints` holds its chord (first and last
point) and `score` its traced length, while `pixels` follows the actual polyline, and `pixels` is
all the residual ever reads. Chaining is on by default and `chain=False` returns the plain detector
segments, which is what the comparison above was measured with.

Detector
--------
OpenCV's Line Segment Detector (`cv2.createLineSegmentDetector`) is used where the build has it
(OpenCV removed it in 4.1 over a patent and brought it back in 4.8; the container here runs 5.1.0
and has it). It is parameter free, subpixel accurate and returns short segments, which is exactly
the regime above. Where it is missing, `cv2.HoughLinesP` on Canny edges stands in. Both go through
`detect_line_segments`, which returns the same `LineSegments` either way.

Pixel coordinate convention
---------------------------
OpenCV puts the centre of pixel (col, row) at the integer coordinate (col, row); this repository
(`utils_gaussian.camera_rays`, `pixel_angles`) puts it at (col + 0.5, row + 0.5). Everything leaving
this module is in the repository convention, so the half pixel is added here, once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

DEFAULT_NUM_SAMPLES = 8
"""Pixels sampled along each segment, endpoints included."""

DEFAULT_MIN_LENGTH = 0.2
"""Shortest line kept, as a fraction of the image side when below 1, in pixels when above.

A fraction by default so the same number works on a 256 pixel training view and on a 1000 pixel
source frame. A fifth of the side is where the curvature of a chain clearly beats the endpoint
noise of its pieces; shorter chains are coplanar under almost any camera and only add noise.
Measured over eight `data-sample` frames, trimmed mean residual at 1.4 times the true focal length
divided by the one at the true focal length, higher is a sharper signal:

    min_length      0.12    0.15    0.20
    1000 px          7.8    12.6    32.3
    256 px           1.4     1.7     2.3
"""

DEFAULT_PIECE_LENGTH = 6.0
"""Shortest raw detector segment that may enter a chain, in pixels. Below this the endpoints of a
segment are no more accurate than the segment is long and the chain direction gets noisy."""

DEFAULT_MAX_COUNT = 32
"""Lines kept per view, the longest first. Also the padded width of the batch tensors."""

DEFAULT_MAX_GAP = 4.0
"""Largest jump between the endpoints of two segments of one chain, in pixels."""

DEFAULT_MAX_TURN_DEG = 30.0
"""Largest direction change between two consecutive segments of a chain, in degrees."""

DEFAULT_MAX_TOTAL_TURN_DEG = 120.0
"""Largest total bend of a chain. A 3D line never images as more than half a circle, and a chain
that turns further has clearly run into another object."""

PAD_VALUE = -1.0
"""Padding of `line_pixels` beyond `line_count`. Negative, so a padded entry can never be mistaken
for a pixel, and `line_geometry.valid_segment_mask` recognises it without a count."""


@dataclass(frozen=True)
class LineSegments:
    """Detected segments of one image.

    * `endpoints` [n, 4] float32, x0, y0, x1, y1 in pixels, repository convention.
    * `score` [n] float32, the segment length in pixels; the list is sorted by it, longest first.
      Length is the confidence here: LSD's own NFA score is `None` in most OpenCV builds, and for
      the plumb line a long segment is worth more anyway, since its rays span a wider angle.
    * `pixels` [n, K, 2] float32, K points sampled evenly along each segment, endpoints included.
    """

    endpoints: np.ndarray
    score: np.ndarray
    pixels: np.ndarray

    def __len__(self) -> int:
        return int(self.endpoints.shape[0])


def _lsd_available() -> bool:
    """Whether this OpenCV build can construct a Line Segment Detector.

    `hasattr` alone is not enough: builds between OpenCV 4.1 and 4.8 keep the symbol and raise on
    construction, so the detector is actually built once and the result cached by the caller.
    """
    if not hasattr(cv2, "createLineSegmentDetector"):
        return False
    try:
        cv2.createLineSegmentDetector()
    except Exception:  # pragma: no cover - depends on the OpenCV build
        return False
    return True


_LSD_CACHE: Optional[object] = None


def _get_lsd() -> object:
    global _LSD_CACHE
    if _LSD_CACHE is None:
        _LSD_CACHE = cv2.createLineSegmentDetector()
    return _LSD_CACHE


def available_detectors() -> Tuple[str, ...]:
    """Detectors this OpenCV build supports, best first. `detector="auto"` takes the first."""
    found = []
    if _lsd_available():
        found.append("lsd")
    if hasattr(cv2, "HoughLinesP") and hasattr(cv2, "Canny"):
        found.append("hough")
    return tuple(found)


def to_gray_uint8(image) -> np.ndarray:
    """Grayscale H x W uint8 from a torch CHW tensor in [0, 1] or a numpy HWC array.

    Accepts torch [3, h, w] or [1, h, w] float (the layout the dataset and the model use), numpy
    [h, w, 3] or [h, w] in uint8 or in float [0, 1], and passes through an already gray uint8 image.
    """
    if isinstance(image, torch.Tensor):
        array = image.detach().to("cpu").float().numpy()
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0.0, 255.0)
        array = array.astype(np.uint8)
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"expected 1 or 3 channels, got {array.shape[2]}")
    elif array.ndim != 2:
        raise ValueError(f"expected an image, got shape {array.shape}")
    return np.ascontiguousarray(array)


def _mask_array(mask, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """A boolean H x W mask from a torch [1, h, w] / [h, w] tensor or a numpy array, or None."""
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        array = mask.detach().to("cpu").numpy()
    else:
        array = np.asarray(mask)
    array = np.squeeze(array)
    if array.shape != shape:
        raise ValueError(f"mask shape {array.shape} does not match the image {shape}")
    return array > 0.5


def sample_segment_pixels(endpoints: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES) -> np.ndarray:
    """K points spread evenly along each straight segment, [n, 4] -> [n, K, 2] float32.

    The endpoints are included (t runs over 0 .. 1), so K = 1 degenerates to the first endpoint and
    K = 2 is just the endpoints.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    pts = np.asarray(endpoints, dtype=np.float32).reshape(-1, 4)
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)[None, :, None]  # [1, K, 1]
    p0 = pts[:, None, 0:2]  # [n, 1, 2]
    p1 = pts[:, None, 2:4]  # [n, 1, 2]
    return (p0 + t * (p1 - p0)).astype(np.float32)


def sample_polyline(polyline: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES) -> np.ndarray:
    """K points spread evenly BY ARC LENGTH along a polyline, [m, 2] -> [K, 2] float32.

    Evenly by arc length and not by vertex, so a chain built from one long and several short
    segments does not put most of its samples into the short end.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    poly = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    if len(poly) == 1:
        return np.repeat(poly.astype(np.float32), num_samples, axis=0)
    step = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.repeat(poly[:1].astype(np.float32), num_samples, axis=0)
    wanted = np.linspace(0.0, total, num_samples)
    index = np.clip(np.searchsorted(cumulative, wanted, side="right") - 1, 0, len(step) - 1)
    local = (wanted - cumulative[index]) / np.maximum(step[index], 1e-12)
    points = poly[index] + local[:, None] * (poly[index + 1] - poly[index])
    return points.astype(np.float32)


def polyline_length(polyline: np.ndarray) -> float:
    """Traced length of a polyline, the sum of its segment lengths."""
    poly = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def _turns(polyline: np.ndarray) -> np.ndarray:
    """Signed direction change at every interior vertex of a polyline, in radians."""
    delta = np.diff(polyline, axis=0)
    length = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(length[:, None], 1e-12)
    if len(unit) < 2:
        return np.zeros(0)
    cross = unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0]
    dot = (unit[:-1] * unit[1:]).sum(1)
    return np.arctan2(cross, dot)


def link_segments(
    endpoints: np.ndarray,
    *,
    max_gap: float = DEFAULT_MAX_GAP,
    max_turn_deg: float = DEFAULT_MAX_TURN_DEG,
    max_total_turn_deg: float = DEFAULT_MAX_TOTAL_TURN_DEG,
    consistent_turn: bool = True,
) -> list:
    """Link short collinear segments back into the curves they were cut out of.

    `endpoints` [n, 4]. Returns a list of polylines, each [m, 2], ordered along the curve; a segment
    nobody linked to comes back as its own two point polyline. Every input segment ends up in
    exactly one polyline.

    Greedy and deterministic: the longest unused segment starts a chain, then the chain grows at
    both ends, each time taking the nearest unused segment whose endpoint is within `max_gap` and
    whose direction continues the chain to within `max_turn_deg`. Greedy is enough here because the
    gap tolerance is a few pixels, so the candidates at a tip are almost never ambiguous.

    A chain is rejected afterwards if it turns by more than `max_total_turn_deg` in total, or, with
    `consistent_turn`, if it turns left at one vertex and right at another: the image of a 3D line
    bends one way and only one way, so a chain that wiggles has joined two different objects. A
    rejected chain is split back into its own segments, which keeps every segment available to the
    caller instead of silently dropping it.

    Why this matters at all is in the module docstring: without the linking the detector hands out
    segments selected for being straight, and a residual built on them prefers a pinhole camera.
    """
    pts = np.asarray(endpoints, dtype=np.float64).reshape(-1, 4)
    n = len(pts)
    if n == 0:
        return []

    # Two rows per segment, one per endpoint: `tip` is the endpoint a chain could attach to and
    # `far` is the other end of the same segment, which is where the chain continues.
    tips = np.concatenate([pts[:, 0:2], pts[:, 2:4]], axis=0)  # [2n, 2]
    far = np.concatenate([pts[:, 2:4], pts[:, 0:2]], axis=0)  # [2n, 2]
    owner = np.concatenate([np.arange(n), np.arange(n)])  # [2n]
    direction = far - tips
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)

    lengths = np.linalg.norm(pts[:, 2:4] - pts[:, 0:2], axis=1)
    used = np.zeros(n, dtype=bool)
    cos_limit = math.cos(math.radians(max_turn_deg))
    chains = []

    for seed in np.argsort(-lengths):
        if used[seed]:
            continue
        used[seed] = True
        members = [int(seed)]
        chain = [pts[seed, 0:2].copy(), pts[seed, 2:4].copy()]
        growing = True
        while growing:
            growing = False
            for end in (0, -1):
                tip = chain[end]
                previous = chain[1] if end == 0 else chain[-2]
                heading = tip - previous
                heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
                distance = np.linalg.norm(tips - tip, axis=1)
                ok = (~used[owner]) & (distance < max_gap) & (direction @ heading >= cos_limit)
                if not ok.any():
                    continue
                pick = int(np.argmin(np.where(ok, distance, np.inf)))
                used[owner[pick]] = True
                members.append(int(owner[pick]))
                # The two endpoints that meet become one vertex at their midpoint. Keeping both and
                # the hop between them would put a vertex pair a pixel or two apart into the
                # polyline, whose direction is pure endpoint noise, and every turn test below would
                # read that noise instead of the curvature.
                if end == 0:
                    chain[0] = 0.5 * (chain[0] + tips[pick])
                    chain.insert(0, far[pick].copy())
                else:
                    chain[-1] = 0.5 * (chain[-1] + tips[pick])
                    chain.append(far[pick].copy())
                growing = True

        poly = np.asarray(chain, dtype=np.float64)
        turns = _turns(poly)
        wiggle = math.radians(5.0)  # below this a turn is endpoint noise, not a bend
        wiggles = consistent_turn and len(turns) > 1 and (turns.max() > wiggle) and (turns.min() < -wiggle)
        over_turned = abs(float(turns.sum())) > math.radians(max_total_turn_deg)
        if wiggles or over_turned:
            chains.extend(pts[m].reshape(2, 2) for m in members)
        else:
            chains.append(poly)
    return chains


def _inside_mask(polyline: np.ndarray, mask: np.ndarray) -> bool:
    """Whether a polyline stays on valid pixels over its whole length.

    Walked at one pixel steps rather than at the K sample points: the data mask of an augmented
    camera is not always convex, so a line can leave it between two samples. The lookup is nearest
    neighbour on the OpenCV grid, so the repository convention is undone by the same half pixel
    that `detect_line_segments` added.
    """
    h, w = mask.shape
    poly = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        steps = max(2, int(np.ceil(float(np.hypot(x1 - x0, y1 - y0)))) + 1)
        t = np.linspace(0.0, 1.0, steps)
        xs = np.round(x0 + t * (x1 - x0) - 0.5).astype(np.int64)
        ys = np.round(y0 + t * (y1 - y0) - 0.5).astype(np.int64)
        if np.any(xs < 0) or np.any(ys < 0) or np.any(xs >= w) or np.any(ys >= h):
            return False
        if not bool(mask[ys, xs].all()):
            return False
    return True


def _detect_lsd(gray: np.ndarray) -> np.ndarray:
    """Raw segments of the Line Segment Detector, [n, 4] in OpenCV coordinates."""
    result = _get_lsd().detect(gray)
    lines = result[0] if isinstance(result, tuple) else result
    if lines is None:
        return np.zeros((0, 4), dtype=np.float32)
    # OpenCV 4 returns [n, 1, 4], OpenCV 5 returns [n, 4].
    return np.asarray(lines, dtype=np.float32).reshape(-1, 4)


def _detect_hough(gray: np.ndarray, min_length: float) -> np.ndarray:
    """Raw segments of the probabilistic Hough transform on Canny edges, [n, 4].

    The fallback for builds without LSD. The Canny thresholds follow the usual median rule, so the
    same numbers work on a dark and on a bright image; `maxLineGap` is kept small, because bridging
    a gap is exactly what would glue two pieces of a curve into one wrong "straight" segment.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blurred))
    lower = int(max(0.0, 0.66 * median))
    upper = int(min(255.0, 1.33 * median))
    edges = cv2.Canny(blurred, lower, max(upper, lower + 1))
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 360.0,
        threshold=max(10, int(min_length)),
        minLineLength=float(min_length),
        maxLineGap=2.0,
    )
    if lines is None:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(lines, dtype=np.float32).reshape(-1, 4)


def resolve_min_length(min_length: float, image_hw: Tuple[int, int]) -> float:
    """A length given as a fraction of the image side (below 1) or in pixels (1 and above)."""
    if min_length < 1.0:
        return float(min_length) * float(min(image_hw))
    return float(min_length)


def image_circle_mask(size: int, margin: float = 2.0) -> np.ndarray:
    """The circle a full frame fisheye images, [size, size] bool, in this repository's convention.

    `lcam_fish` fills a circle spanning the image width and leaves the corners black. The border of
    that circle is a long, perfectly smooth arc that the detector happily reports and that belongs
    to no 3D line at all, so every caller working on an unaugmented fisheye frame has to mask it;
    `margin` keeps the few pixels of interpolated edge out as well.
    """
    y, x = np.mgrid[0:size, 0:size]
    return np.hypot(x + 0.5 - size / 2.0, y + 0.5 - size / 2.0) <= size / 2.0 - margin


def detect_line_segments(
    image,
    *,
    mask=None,
    min_length: float = DEFAULT_MIN_LENGTH,
    max_count: int = DEFAULT_MAX_COUNT,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    detector: str = "auto",
    chain: bool = True,
    piece_length: float = DEFAULT_PIECE_LENGTH,
    max_gap: float = DEFAULT_MAX_GAP,
    max_turn_deg: float = DEFAULT_MAX_TURN_DEG,
) -> LineSegments:
    """Image curves that each belong to one 3D line, longest first.

    * `image` torch [3, h, w] / [1, h, w] in [0, 1], or numpy [h, w, 3] / [h, w].
    * `mask` optional validity mask of the same height and width (the sample's `data_mask`, say).
      A line that touches an invalid pixel anywhere along its length is dropped, so no line can be
      an artefact of the border between data and padding.
    * `min_length` shortest line kept, a fraction of the image side below 1 and pixels above.
    * `max_count` how many of the longest lines survive.
    * `num_samples` K, the pixels sampled along each line for the plumb line residual.
    * `detector` "auto", "lsd" or "hough"; see `available_detectors`.
    * `chain` link the raw detector segments into curves first, see the module docstring. Keep it
      on for anything that estimates a camera; off gives the plain detector segments.
    * `piece_length`, `max_gap`, `max_turn_deg` steer the linking.

    Returns `LineSegments` with the chord endpoints, the traced lengths as scores and the sampled
    pixels, all in this repository's pixel convention (centre of pixel (col, row) at
    (col + 0.5, row + 0.5)).
    """
    gray = to_gray_uint8(image)
    mask_array = _mask_array(mask, gray.shape)
    min_px = resolve_min_length(min_length, gray.shape)

    choices = available_detectors()
    if detector == "auto":
        if not choices:
            raise RuntimeError(
                "this OpenCV build offers neither createLineSegmentDetector nor HoughLinesP"
            )
        detector = choices[0]
    if detector not in ("lsd", "hough"):
        raise ValueError(f"unknown detector {detector!r}, expected auto, lsd or hough")
    if detector not in choices:
        raise RuntimeError(f"detector {detector!r} is not available in this OpenCV build")

    # Chaining needs the short pieces, so the detector is asked for everything above `piece_length`
    # and `min_length` is applied to the finished lines; without chaining the two are the same.
    floor = min(float(piece_length), min_px) if chain else min_px
    raw = _detect_lsd(gray) if detector == "lsd" else _detect_hough(gray, floor)
    if len(raw):
        raw = raw + 0.5  # OpenCV pixel centres to this repository's
        keep = np.hypot(raw[:, 2] - raw[:, 0], raw[:, 3] - raw[:, 1]) >= floor
        raw = raw[keep]

    if chain:
        lines = link_segments(raw, max_gap=max_gap, max_turn_deg=max_turn_deg)
    else:
        lines = [row.reshape(2, 2) for row in np.asarray(raw, dtype=np.float64).reshape(-1, 4)]

    kept, scores = [], []
    for poly in lines:
        length = polyline_length(poly)
        if length < min_px:
            continue
        if mask_array is not None and not _inside_mask(poly, mask_array):
            continue
        kept.append(poly)
        scores.append(length)

    order = np.argsort(-np.asarray(scores, dtype=np.float64))[: max(0, int(max_count))] if kept else []
    endpoints = np.zeros((len(order), 4), dtype=np.float32)
    score = np.zeros((len(order),), dtype=np.float32)
    pixels = np.zeros((len(order), int(num_samples), 2), dtype=np.float32)
    for row, index in enumerate(order):
        poly = kept[int(index)]
        endpoints[row] = np.concatenate([poly[0], poly[-1]]).astype(np.float32)
        score[row] = scores[int(index)]
        pixels[row] = sample_polyline(poly, num_samples)
    return LineSegments(endpoints=endpoints, score=score, pixels=pixels)


def detect_lines_in_views(
    images,
    *,
    masks=None,
    min_length: float = DEFAULT_MIN_LENGTH,
    max_count: int = DEFAULT_MAX_COUNT,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    detector: str = "auto",
    chain: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the detector on every view of one sample and pad the result into batch tensors.

    * `images` [v, 3, h, w] torch or a sequence of per view images,
    * `masks` [v, 1, h, w] torch or None.

    Returns `line_pixels` [v, max_count, K, 2] float32 and `line_count` [v] int64. Rows beyond the
    count of a view are filled with `PAD_VALUE`. The padded width is `max_count` for every view and
    every sample, not the largest count found, because the default collate of the data loader can
    only stack samples of equal shape.
    """
    views = list(images)
    mask_list: Sequence = [None] * len(views) if masks is None else list(masks)
    if len(mask_list) != len(views):
        raise ValueError("masks must have one entry per view")

    pixels = torch.full((len(views), int(max_count), int(num_samples), 2), PAD_VALUE, dtype=torch.float32)
    counts = torch.zeros(len(views), dtype=torch.long)
    for i, view in enumerate(views):
        segments = detect_line_segments(
            view,
            mask=mask_list[i],
            min_length=min_length,
            max_count=max_count,
            num_samples=num_samples,
            detector=detector,
            chain=chain,
        )
        n = len(segments)
        if n:
            pixels[i, :n] = torch.from_numpy(segments.pixels)
        counts[i] = n
    return pixels, counts


def draw_segments(image, lines: np.ndarray, color: Tuple[int, int, int] = (255, 32, 32)) -> np.ndarray:
    """The image with the lines drawn on it, H x W x 3 uint8 RGB. For figures only.

    `lines` is either [n, 4] chord endpoints or [n, K, 2] sampled pixels; pass the sampled pixels of
    a chained detection, they follow the curve, while the chord is a straight shortcut across it.
    """
    if isinstance(image, torch.Tensor):
        array = image.detach().to("cpu").float().numpy()
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    canvas = np.ascontiguousarray(array.copy())
    data = np.asarray(lines, dtype=np.float32)
    if data.size == 0:
        return canvas
    polylines = data.reshape(-1, 2, 2) if data.ndim == 2 else data.reshape(len(data), -1, 2)
    for poly in polylines:
        points = np.round(poly - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
    return canvas
