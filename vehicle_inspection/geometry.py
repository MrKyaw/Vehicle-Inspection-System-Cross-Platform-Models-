"""Detection geometry used by every deployment target.

The functions in this module are intentionally free of NumPy, PIL, OpenCV,
torch, or ONNX Runtime dependencies.  The Rust/WASM preprocessor and the
Python model servers in ``src/`` all perform a 224x224 (or 640x640) resize;
this module captures the *math* those runtimes must agree on:

* axis-aligned box validation and clamping,
* intersection-over-union (IoU),
* deterministic non-maximum suppression (NMS),
* aspect-preserving letterbox resize and its inverse mapping.

All coordinates are expressed in pixels of the source image unless a
function explicitly says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import GeometryError


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned rectangle in pixel coordinates.

    ``x2`` and ``y2`` are exclusive edges, matching the convention used by
    most detector export pipelines.  Degenerate boxes (zero width or height)
    are allowed because clamping a box that lies fully outside an image
    legitimately produces one; they simply have zero area and zero IoU.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x1", self.x1),
            ("y1", self.y1),
            ("x2", self.x2),
            ("y2", self.y2),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise GeometryError(f"box coordinate {name} must be a number, got {value!r}")
            if not math.isfinite(value):
                raise GeometryError(f"box coordinate {name} must be finite, got {value!r}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise GeometryError(
                "box edges must satisfy x2 >= x1 and y2 >= y1, "
                f"got ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        """Width in pixels (never negative)."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height in pixels (never negative)."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point ``(x, y)``."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)`` for JSON serialisation."""
        return (self.x1, self.y1, self.x2, self.y2)


def clamp_box(box: Box, image_width: int, image_height: int) -> Box:
    """Clip ``box`` to the image rectangle ``[0, width] x [0, height]``.

    Boxes fully outside the image collapse to a zero-area box on the nearest
    edge rather than being dropped; the caller can inspect :attr:`Box.area`.
    """
    if image_width <= 0 or image_height <= 0:
        raise GeometryError("image dimensions must be positive")
    x1 = min(max(box.x1, 0.0), float(image_width))
    y1 = min(max(box.y1, 0.0), float(image_height))
    x2 = min(max(box.x2, 0.0), float(image_width))
    y2 = min(max(box.y2, 0.0), float(image_height))
    return Box(x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    """Return the intersection-over-union of two boxes in ``[0, 1]``.

    Two degenerate (zero-area) boxes at the same location share no area and
    therefore have IoU ``0.0``.
    """
    inter_width = min(a.x2, b.x2) - max(a.x1, b.x1)
    inter_height = min(a.y2, b.y2) - max(a.y1, b.y1)
    intersection = max(0.0, inter_width) * max(0.0, inter_height)
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def non_max_suppression(
    boxes: Sequence[Box],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> list[int]:
    """Return indices of boxes kept by greedy non-maximum suppression.

    Ordering is deterministic: candidates are sorted by descending score and
    then by ascending index, so equal scores never depend on sort stability
    or dictionary order.  Boxes with ``score < score_threshold`` are skipped.
    A box is suppressed when its IoU with an already-kept box is *strictly
    greater* than ``iou_threshold``.
    """
    if len(boxes) != len(scores):
        raise GeometryError(
            f"boxes and scores must have equal length, got {len(boxes)} and {len(scores)}"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise GeometryError(f"iou_threshold must be within [0, 1], got {iou_threshold!r}")
    for score in scores:
        if not math.isfinite(score):
            raise GeometryError(f"scores must be finite, got {score!r}")
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda index: (-scores[index], index))
    kept: list[int] = []
    for index in order:
        if scores[index] < score_threshold:
            continue
        suppressed = False
        for kept_index in kept:
            if iou(boxes[index], boxes[kept_index]) > iou_threshold:
                suppressed = True
                break
        if suppressed:
            continue
        kept.append(index)
    return kept


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Geometry of an aspect-preserving resize into a square/rect canvas.

    ``scale`` maps original pixels to resized pixels.  ``pad_x`` and ``pad_y``
    are the offsets of the resized image inside the target canvas.
    """

    original_width: int
    original_height: int
    target_width: int
    target_height: int
    new_width: int
    new_height: int
    scale: float
    pad_x: float
    pad_y: float

    def to_original_box(self, box: Box) -> Box:
        """Invert letterbox coordinates back to the source image.

        Detector outputs are produced in canvas coordinates; this method maps
        them back and clamps to the original image bounds.
        """
        x1 = (box.x1 - self.pad_x) / self.scale
        y1 = (box.y1 - self.pad_y) / self.scale
        x2 = (box.x2 - self.pad_x) / self.scale
        y2 = (box.y2 - self.pad_y) / self.scale
        return clamp_box(
            Box(x1, y1, x2, y2),
            self.original_width,
            self.original_height,
        )


def letterbox(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> LetterboxTransform:
    """Compute aspect-preserving resize plus centering offsets.

    The shorter side is padded symmetrically, matching the letterbox strategy
    used by YOLO-family detectors.  ``round`` keeps the output pixel-aligned;
    at least one pixel is preserved for extreme aspect ratios.
    """
    if image_width <= 0 or image_height <= 0:
        raise GeometryError("image dimensions must be positive")
    if target_width <= 0 or target_height <= 0:
        raise GeometryError("target dimensions must be positive")

    scale = min(target_width / image_width, target_height / image_height)
    new_width = int(round(image_width * scale))
    new_height = int(round(image_height * scale))
    if new_width < 1:
        new_width = 1
    if new_height < 1:
        new_height = 1

    return LetterboxTransform(
        original_width=image_width,
        original_height=image_height,
        target_width=target_width,
        target_height=target_height,
        new_width=new_width,
        new_height=new_height,
        scale=scale,
        pad_x=(target_width - new_width) / 2.0,
        pad_y=(target_height - new_height) / 2.0,
    )
