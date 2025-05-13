"""Dependency-free vehicle inspection domain library.

``vehicle_inspection`` turns raw detector output into a defensible inspection
verdict: geometry agreement (IoU/NMS/letterbox), manifest validation,
confidence calibration, class-level policy, and a stable JSON/Markdown report.

The package uses only the Python standard library and never performs network
I/O, making it safe to run in CI, air-gapped review machines, and tests.
"""

from __future__ import annotations

from .calibration import (
    TemperatureCalibrator,
    apply_temperature,
    bucket_confidence,
    expected_calibration_error,
    softmax,
)
from .demo import DemoScenario, build_demo
from .errors import (
    CalibrationError,
    GeometryError,
    ManifestError,
    PolicyError,
    ReportError,
    VehicleInspectionError,
)
from .geometry import Box, LetterboxTransform, clamp_box, iou, letterbox, non_max_suppression
from .manifest import (
    ModelManifest,
    Preprocessing,
    Thresholds,
    load_manifest,
)
from .policy import (
    DEFAULT_RULES,
    SEVERITY_WEIGHTS,
    ClassRule,
    Detection,
    Finding,
    IgnoredDetection,
    InspectionPolicy,
    InspectionResult,
    Verdict,
    locate_zone,
)
from .report import REPORT_SCHEMA_VERSION, build_report, render_markdown, report_to_json

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_RULES",
    "REPORT_SCHEMA_VERSION",
    "SEVERITY_WEIGHTS",
    "Box",
    "CalibrationError",
    "ClassRule",
    "DemoScenario",
    "Detection",
    "Finding",
    "GeometryError",
    "IgnoredDetection",
    "InspectionPolicy",
    "InspectionResult",
    "LetterboxTransform",
    "ManifestError",
    "ModelManifest",
    "PolicyError",
    "Preprocessing",
    "ReportError",
    "TemperatureCalibrator",
    "Thresholds",
    "VehicleInspectionError",
    "Verdict",
    "__version__",
    "apply_temperature",
    "bucket_confidence",
    "build_demo",
    "build_report",
    "clamp_box",
    "expected_calibration_error",
    "iou",
    "letterbox",
    "load_manifest",
    "locate_zone",
    "non_max_suppression",
    "render_markdown",
    "report_to_json",
    "softmax",
]
