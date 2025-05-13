"""Built-in demonstration scenario.

The demo exercises the whole pipeline offline: a manifest, raw detections
including a near-duplicate, NMS suppression, one low-confidence detection,
per-class temperature calibration, aggregate-risk escalation, and a rendered
report.  It intentionally produces ``fail`` so reviewers can see the safety
path, not only the happy path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import ModelManifest, Preprocessing, Thresholds
from .policy import Detection
from .geometry import Box


@dataclass(frozen=True, slots=True)
class DemoScenario:
    """All inputs needed to reproduce the demo inspection."""

    manifest: ModelManifest
    detections: tuple[Detection, ...]
    image_width: int
    image_height: int
    source: str = "demo"


def build_demo() -> DemoScenario:
    """Return a deterministic inspection scenario (no files, no network)."""
    manifest = ModelManifest(
        name="vehicle-damage-resnet18",
        version="1.0.0",
        input_shape=(1, 3, 640, 640),
        labels=("scratch", "dent", "crack", "chip"),
        preprocessing=Preprocessing(
            resize=(640, 640),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            pad_value=114,
        ),
        thresholds=Thresholds(score=0.35, iou=0.5, review=0.60, fail=0.85),
        platforms=("onnx", "pytorch", "wasm"),
        calibration={"crack": 1.4, "dent": 1.2},
        metadata={"task": "vehicle-damage-detection", "license": "MIT"},
    )
    detections = (
        Detection("scratch", 0.82, Box(120.0, 120.0, 300.0, 220.0), "minor"),
        # Near-duplicate of the first scratch: NMS should remove it.
        Detection("scratch", 0.78, Box(125.0, 122.0, 298.0, 218.0), "minor"),
        Detection("dent", 0.74, Box(900.0, 380.0, 1100.0, 560.0), "moderate"),
        Detection("crack", 0.71, Box(300.0, 480.0, 360.0, 700.0), "severe"),
        Detection("chip", 0.22, Box(40.0, 640.0, 90.0, 690.0), "minor"),
    )
    return DemoScenario(
        manifest=manifest,
        detections=detections,
        image_width=1280,
        image_height=720,
    )
