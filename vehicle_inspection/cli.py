"""Command-line interface for offline vehicle inspection.

Usage::

    python -m vehicle_inspection --demo
    python -m vehicle_inspection detections.json --json
    python -m vehicle_inspection detections.json --manifest model.json --output report.md

The CLI is deliberately thin: all decisions live in the library modules so
the same behaviour is available to tests and future service wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .calibration import TemperatureCalibrator
from .demo import build_demo
from .errors import PolicyError, ReportError, VehicleInspectionError
from .geometry import Box, non_max_suppression
from .manifest import ModelManifest, load_manifest
from .policy import Detection, InspectionPolicy
from .report import build_report, render_markdown, report_to_json


@dataclass(frozen=True, slots=True)
class DetectionFile:
    """Decoded contents of a detections JSON file."""

    detections: tuple[Detection, ...]
    image_width: int | None
    image_height: int | None


def _read_json(path: Path, context: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"could not read {context} {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"{context} {path} is not valid JSON: {exc.msg}") from exc


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PolicyError(f"{path} must be a positive integer")
    return value


def load_detection_file(path: str | Path) -> DetectionFile:
    """Load detections from JSON.

    Two shapes are accepted for convenience: a bare list of detections, or an
    object with ``detections`` plus optional ``image`` dimensions.
    """
    source = Path(path)
    data = _read_json(source, "detections file")
    image_width: int | None = None
    image_height: int | None = None

    if isinstance(data, list):
        raw_detections = data
    elif isinstance(data, dict):
        for key in data:
            if key not in {"detections", "image"}:
                raise PolicyError(f"detections file contains unknown key {key!r}")
        if "detections" not in data:
            raise PolicyError("detections file object must contain a 'detections' array")
        raw_detections = data["detections"]
        if not isinstance(raw_detections, list):
            raise PolicyError("'detections' must be a JSON array")
        if "image" in data:
            image = data["image"]
            if not isinstance(image, dict):
                raise PolicyError("'image' must be a JSON object")
            for key in image:
                if key not in {"width", "height"}:
                    raise PolicyError(f"'image' contains unknown key {key!r}")
            if "width" not in image or "height" not in image:
                raise PolicyError("'image' must contain both width and height")
            image_width = _positive_int(image["width"], "image.width")
            image_height = _positive_int(image["height"], "image.height")
    else:
        raise PolicyError("detections file must contain a JSON array or object")

    detections = tuple(Detection.from_dict(item) for item in raw_detections)
    return DetectionFile(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (kept separate for testability)."""
    parser = argparse.ArgumentParser(
        prog="python -m vehicle_inspection",
        description=(
            "Offline vehicle-inspection decision engine. "
            "Accepts detector JSON, applies NMS + policy, prints a report."
        ),
    )
    parser.add_argument(
        "detections",
        nargs="?",
        help="JSON detections file (omit when using --demo)",
    )
    parser.add_argument("--manifest", help="optional model manifest JSON")
    parser.add_argument("--image-width", type=int, default=1280, help="image width (default: 1280)")
    parser.add_argument(
        "--image-height", type=int, default=720, help="image height (default: 720)"
    )
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of Markdown")
    parser.add_argument("--output", help="write the rendered report to this path")
    parser.add_argument("--demo", action="store_true", help="run the built-in fail scenario")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _resolve_manifest(path: str | None) -> ModelManifest | None:
    if path is None:
        return None
    return load_manifest(path)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[dict[str, Any], str]:
    if args.demo and args.detections:
        parser.error("pass either a detections file or --demo, not both")
    if not args.demo and not args.detections:
        parser.error("a detections file is required unless --demo is used")

    if args.demo:
        scenario = build_demo()
        detections = scenario.detections
        manifest: ModelManifest | None = scenario.manifest
        image_width = scenario.image_width
        image_height = scenario.image_height
        source = scenario.source
    else:
        payload = load_detection_file(args.detections)
        detections = payload.detections
        manifest = _resolve_manifest(args.manifest)
        if payload.image_width is None:
            image_width = _positive_int(args.image_width, "--image-width")
        else:
            image_width = payload.image_width
        if payload.image_height is None:
            image_height = _positive_int(args.image_height, "--image-height")
        else:
            image_height = payload.image_height
        source = str(args.detections)

    if manifest is not None:
        known = set(manifest.labels)
        for detection in detections:
            if detection.label not in known:
                raise PolicyError(
                    f"detection label {detection.label!r} is not declared in manifest labels"
                )

    score_floor = manifest.thresholds.score if manifest is not None else 0.35
    iou_threshold = manifest.thresholds.iou if manifest is not None else 0.5
    boxes = [detection.box for detection in detections]
    scores = [detection.confidence for detection in detections]
    kept = non_max_suppression(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_floor,
    )
    suppressed_count = len(detections) - len(kept)
    selected = [detections[index] for index in kept]

    calibrator: TemperatureCalibrator | None
    if manifest is not None and manifest.calibration:
        calibrator = TemperatureCalibrator(manifest.calibration)
    else:
        calibrator = None

    result = InspectionPolicy().inspect(
        selected,
        image_width=image_width,
        image_height=image_height,
        calibrator=calibrator,
    )
    report = build_report(
        result,
        manifest=manifest,
        source=source,
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        suppressed_count=suppressed_count,
    )
    return report, "json" if args.json else "markdown"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code (0 success, 2 domain error)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, output_format = _run(args, parser)
        rendered = report_to_json(report) if output_format == "json" else render_markdown(report)
        if args.output:
            try:
                Path(args.output).write_text(rendered, encoding="utf-8")
            except OSError as exc:
                raise ReportError(f"could not write report to {args.output}: {exc}") from exc
            print(f"Report written to {args.output}")
        else:
            print(rendered, end="" if rendered.endswith("\n") else "\n")
    except VehicleInspectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
