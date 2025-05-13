"""Stable JSON report schema and Markdown renderer.

The JSON document is the machine-readable contract; the Markdown rendering is
a projection of that same document, so a dashboard and a human reviewer can
never disagree.  Schema versioning is explicit so downstream consumers can
reject breaking changes.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .errors import ReportError
from .manifest import ModelManifest
from .policy import InspectionResult

REPORT_SCHEMA_VERSION = "1.0"
_REQUIRED_TOP_LEVEL = ("schema_version", "inspection", "model", "findings", "reasons")


def build_report(
    result: InspectionResult,
    manifest: ModelManifest | None = None,
    source: str | None = None,
    generated_at: str | None = None,
    suppressed_count: int = 0,
) -> dict[str, Any]:
    """Build the canonical report dictionary for one inspection.

    ``generated_at`` is injected rather than read from the clock so the core
    library stays deterministic and testable; the CLI supplies a UTC
    timestamp.  ``suppressed_count`` records detections removed by NMS
    before the policy ran.
    """
    if not isinstance(result, InspectionResult):
        raise ReportError("result must be an InspectionResult")
    if (
        not isinstance(suppressed_count, int)
        or isinstance(suppressed_count, bool)
        or suppressed_count < 0
    ):
        raise ReportError("suppressed_count must be a non-negative integer")

    model: dict[str, Any] | None
    if manifest is None:
        model = None
    else:
        model = {
            "name": manifest.name,
            "version": manifest.version,
            "input_shape": list(manifest.input_shape),
            "labels": list(manifest.labels),
            "platforms": list(manifest.platforms),
            "thresholds": {
                "score": manifest.thresholds.score,
                "iou": manifest.thresholds.iou,
                "review": manifest.thresholds.review,
                "fail": manifest.thresholds.fail,
            },
            "calibration": dict(manifest.calibration),
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source": source,
        "model": model,
        "inspection": {
            "verdict": result.verdict,
            "risk_score": result.risk_score,
            "image": {"width": result.image_width, "height": result.image_height},
        },
        "findings": [finding.as_dict() for finding in result.findings],
        "ignored": [ignored.as_dict() for ignored in result.ignored],
        "suppressed_count": suppressed_count,
        "reasons": list(result.reasons),
    }


def report_to_json(report: Mapping[str, Any], indent: int = 2) -> str:
    """Serialise a report with sorted keys for stable diffs."""
    if not isinstance(report, Mapping):
        raise ReportError("report must be a mapping")
    if not isinstance(indent, int) or isinstance(indent, bool) or indent < 0:
        raise ReportError("indent must be a non-negative integer")
    return json.dumps(report, indent=indent, sort_keys=True) + "\n"


def _require_schema(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ReportError("report must be a mapping")
    for key in _REQUIRED_TOP_LEVEL:
        if key not in report:
            raise ReportError(f"report is missing required key {key!r}")


def _cell(value: object) -> str:
    """Escape a value for a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a report dictionary as a review-friendly Markdown document."""
    _require_schema(report)
    inspection = report["inspection"]
    image = inspection["image"]

    lines: list[str] = [
        "# Vehicle Inspection Report",
        "",
        f"- **Verdict:** `{inspection['verdict']}`",
        f"- **Risk score:** {inspection['risk_score']:.2f}",
        f"- **Image:** {image['width']}x{image['height']}",
        f"- **Schema:** {report['schema_version']}",
    ]
    if report.get("source"):
        lines.append(f"- **Source:** {_cell(report['source'])}")
    if report.get("generated_at"):
        lines.append(f"- **Generated:** {_cell(report['generated_at'])}")

    model = report["model"]
    if model is None:
        lines.extend(["", "## Model", "", "_No manifest supplied._"])
    else:
        platforms = ", ".join(str(platform) for platform in model["platforms"])
        lines.extend(
            [
                "",
                "## Model",
                "",
                f"- **Name:** {_cell(model['name'])} `{_cell(model['version'])}`",
                f"- **Input shape:** `{model['input_shape']}`",
                f"- **Platforms:** {_cell(platforms)}",
            ]
        )

    lines.extend(["", "## Findings", ""])
    findings = report["findings"]
    if not findings:
        lines.append("_No findings above the class confidence floors._")
    else:
        lines.extend(
            [
                "| Class | Severity | Zone | Confidence | Raw | Area ratio |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for finding in findings:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(finding["label"]),
                        _cell(finding["severity"]),
                        _cell(finding["zone"]),
                        f"{finding['confidence']:.3f}",
                        f"{finding['raw_confidence']:.3f}",
                        f"{finding['area_ratio']:.4f}",
                    ]
                )
                + " |"
            )

    ignored = report["ignored"]
    if ignored:
        lines.extend(["", "## Ignored detections", ""])
        for item in ignored:
            lines.append(
                f"- `{_cell(item['label'])}` at {item['confidence']:.3f}: {_cell(item['reason'])}"
            )

    if report["suppressed_count"]:
        lines.extend(
            [
                "",
                f"_{report['suppressed_count']} overlapping or low-score detection(s) "
                "were removed before policy evaluation._",
            ]
        )

    lines.extend(["", "## Reasons", ""])
    for reason in report["reasons"]:
        lines.append(f"- {_cell(reason)}")

    lines.append("")
    return "\n".join(lines)
