"""Inspection decision policy.

Raw detections are not verdicts.  This module converts detector output into
inspection *findings* (what was found, where, how severe, how confident) and
aggregates them into one of three operational verdicts:

``pass``
    Only cosmetic, low-confidence findings (or none).  No human needed.
``needs_review``
    A moderate/severe defect, or a finding confident enough that a human
    should confirm it before the vehicle is released.
``fail``
    A severe defect above its fail-confidence, or aggregate risk at/above the
    configured fail threshold.  The vehicle should not be released.

The aggregate-risk rule exists because several medium-confidence defects can
be collectively worse than any single one — a case a naive "max confidence"
policy misses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .calibration import TemperatureCalibrator
from .errors import PolicyError
from .geometry import Box

Verdict = Literal["pass", "needs_review", "fail"]

SEVERITY_WEIGHTS: Mapping[str, int] = MappingProxyType({"minor": 1, "moderate": 2, "severe": 3})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_image_size(width: int, height: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise PolicyError("image_width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise PolicyError("image_height must be a positive integer")


@dataclass(frozen=True, slots=True)
class Detection:
    """A single raw detector output."""

    label: str
    confidence: float
    box: Box
    severity: str = "minor"

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise PolicyError("detection label must be a non-empty string")
        if not _is_number(self.confidence):
            raise PolicyError("detection confidence must be a number")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise PolicyError("detection confidence must be within [0, 1]")
        if not isinstance(self.box, Box):
            raise PolicyError("detection box must be a Box instance")
        if self.severity not in SEVERITY_WEIGHTS:
            raise PolicyError(
                f"unknown severity {self.severity!r}; expected one of {sorted(SEVERITY_WEIGHTS)}"
            )

    @classmethod
    def from_dict(cls, data: Any) -> "Detection":
        """Parse one detection from JSON-shaped data."""
        if not isinstance(data, dict):
            raise PolicyError("each detection must be a JSON object")
        allowed = {"label", "confidence", "box", "severity"}
        for key in data:
            if key not in allowed:
                raise PolicyError(f"detection contains unknown key {key!r}")
        for required in ("label", "confidence", "box"):
            if required not in data:
                raise PolicyError(f"detection.{required} is required")

        raw_box = data["box"]
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise PolicyError("detection.box must be an array of four numbers [x1, y1, x2, y2]")
        for index, coordinate in enumerate(raw_box):
            if not _is_number(coordinate) or not math.isfinite(coordinate):
                raise PolicyError(f"detection.box[{index}] must be a finite number")
        box = Box(*(float(coordinate) for coordinate in raw_box))

        confidence = data["confidence"]
        if not _is_number(confidence):
            raise PolicyError("detection.confidence must be a number")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise PolicyError("detection.confidence must be within [0, 1]")

        return cls(
            label=data["label"],
            confidence=float(confidence),
            box=box,
            severity=data.get("severity", "minor"),
        )

    def area_ratio(self, image_width: int, image_height: int) -> float:
        """Return the fraction of the image covered by this detection."""
        _validate_image_size(image_width, image_height)
        return self.box.area / (image_width * image_height)


@dataclass(frozen=True, slots=True)
class ClassRule:
    """Per-class inspection thresholds and default severity."""

    severity: str = "minor"
    min_confidence: float = 0.35
    review_confidence: float = 0.6
    fail_confidence: float = 0.9

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_WEIGHTS:
            raise PolicyError(
                f"unknown severity {self.severity!r}; expected one of {sorted(SEVERITY_WEIGHTS)}"
            )
        for name, value in (
            ("min_confidence", self.min_confidence),
            ("review_confidence", self.review_confidence),
            ("fail_confidence", self.fail_confidence),
        ):
            if not _is_number(value) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PolicyError(f"{name} must be a finite number within [0, 1]")
        if not self.min_confidence <= self.review_confidence <= self.fail_confidence:
            raise PolicyError(
                "class confidences must satisfy min_confidence <= review_confidence <= fail_confidence"
            )


DEFAULT_RULES: Mapping[str, ClassRule] = MappingProxyType(
    {
        "scratch": ClassRule("minor", 0.35, 0.60, 0.95),
        "chip": ClassRule("minor", 0.40, 0.70, 0.98),
        "dent": ClassRule("moderate", 0.45, 0.60, 0.90),
        "crack": ClassRule("severe", 0.50, 0.55, 0.80),
    }
)


def locate_zone(box: Box, image_width: int, image_height: int) -> str:
    """Map a box centre to a coarse vehicle zone.

    The image is divided into a 3x3 grid.  This is deliberately coarse: the
    research model predicts defect classes, not panels, so inventing precise
    panel names would overstate what the data supports.
    """
    _validate_image_size(image_width, image_height)
    centre_x, centre_y = box.center
    relative_x = centre_x / image_width
    relative_y = centre_y / image_height

    if relative_x < 1.0 / 3.0:
        horizontal = "left"
    elif relative_x < 2.0 / 3.0:
        horizontal = "center"
    else:
        horizontal = "right"

    if relative_y < 1.0 / 3.0:
        vertical = "front"
    elif relative_y < 2.0 / 3.0:
        vertical = "middle"
    else:
        vertical = "rear"

    return f"{vertical}-{horizontal}"


@dataclass(frozen=True, slots=True)
class Finding:
    """An accepted defect with location, severity, and calibrated confidence."""

    label: str
    severity: str
    confidence: float
    raw_confidence: float
    zone: str
    box: Box
    area_ratio: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation (no custom objects)."""
        return {
            "label": self.label,
            "severity": self.severity,
            "confidence": self.confidence,
            "raw_confidence": self.raw_confidence,
            "zone": self.zone,
            "box": {
                "x1": self.box.x1,
                "y1": self.box.y1,
                "x2": self.box.x2,
                "y2": self.box.y2,
            },
            "area_ratio": self.area_ratio,
        }


@dataclass(frozen=True, slots=True)
class IgnoredDetection:
    """A detection that never became a finding, with the reason why."""

    label: str
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {"label": self.label, "confidence": self.confidence, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Deterministic output of :meth:`InspectionPolicy.inspect`."""

    verdict: Verdict
    findings: tuple[Finding, ...]
    ignored: tuple[IgnoredDetection, ...]
    risk_score: float
    reasons: tuple[str, ...]
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class InspectionPolicy:
    """Configurable, deterministic aggregation policy."""

    rules: Mapping[str, ClassRule] = field(default_factory=lambda: dict(DEFAULT_RULES))
    fail_risk_threshold: float = 2.5
    max_findings: int = 50

    def __post_init__(self) -> None:
        if not self.rules:
            raise PolicyError("rules must not be empty")
        for label, rule in self.rules.items():
            if not isinstance(label, str) or not label.strip():
                raise PolicyError("rule labels must be non-empty strings")
            if not isinstance(rule, ClassRule):
                raise PolicyError(f"rule for {label!r} must be a ClassRule")
        if (
            not _is_number(self.fail_risk_threshold)
            or not math.isfinite(self.fail_risk_threshold)
            or self.fail_risk_threshold <= 0.0
        ):
            raise PolicyError("fail_risk_threshold must be a positive finite number")
        if (
            not isinstance(self.max_findings, int)
            or isinstance(self.max_findings, bool)
            or self.max_findings <= 0
        ):
            raise PolicyError("max_findings must be a positive integer")

    def inspect(
        self,
        detections: Sequence[Detection],
        image_width: int,
        image_height: int,
        calibrator: TemperatureCalibrator | None = None,
    ) -> InspectionResult:
        """Apply class rules, calibration, and aggregate escalation.

        Raises :class:`PolicyError` for unknown classes, non-``Detection``
        items, or more accepted findings than ``max_findings``.  Unknown
        classes are rejected rather than silently ignored: a safety system
        should never discard a defect because a manifest drifted.
        """
        _validate_image_size(image_width, image_height)

        accepted: list[Finding] = []
        ignored: list[IgnoredDetection] = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                raise PolicyError(f"detections[{index}] must be a Detection instance")
            rule = self.rules.get(detection.label)
            if rule is None:
                raise PolicyError(
                    f"no policy rule for label {detection.label!r}; "
                    f"known labels: {sorted(self.rules)}"
                )
            if calibrator is None:
                confidence = detection.confidence
            else:
                confidence = calibrator.calibrate(detection.label, detection.confidence)
            if confidence < rule.min_confidence:
                ignored.append(
                    IgnoredDetection(
                        label=detection.label,
                        confidence=confidence,
                        reason=(
                            f"below {detection.label} confidence floor "
                            f"({rule.min_confidence:.2f})"
                        ),
                    )
                )
                continue
            accepted.append(
                Finding(
                    label=detection.label,
                    severity=rule.severity,
                    confidence=confidence,
                    raw_confidence=detection.confidence,
                    zone=locate_zone(detection.box, image_width, image_height),
                    box=detection.box,
                    area_ratio=detection.area_ratio(image_width, image_height),
                )
            )

        if len(accepted) > self.max_findings:
            raise PolicyError(
                f"{len(accepted)} findings exceed max_findings={self.max_findings}; "
                "refusing to produce an inspection report"
            )

        accepted.sort(
            key=lambda finding: (
                -finding.confidence,
                finding.label,
                finding.box.x1,
                finding.box.y1,
                finding.box.x2,
                finding.box.y2,
            )
        )

        risk_score = round(
            sum(SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in accepted),
            4,
        )
        reasons = self._reasons(accepted, risk_score)
        if ignored:
            reasons.append(
                f"{len(ignored)} detection(s) ignored below class confidence floors."
            )

        return InspectionResult(
            verdict=self._verdict(accepted, risk_score),
            findings=tuple(accepted),
            ignored=tuple(ignored),
            risk_score=risk_score,
            reasons=tuple(reasons),
            image_width=image_width,
            image_height=image_height,
        )

    def _verdict(self, findings: Sequence[Finding], risk_score: float) -> Verdict:
        for finding in findings:
            rule = self.rules[finding.label]
            if finding.severity == "severe" and finding.confidence >= rule.fail_confidence:
                return "fail"
        if risk_score >= self.fail_risk_threshold:
            return "fail"
        for finding in findings:
            rule = self.rules[finding.label]
            if finding.severity != "minor" or finding.confidence >= rule.review_confidence:
                return "needs_review"
        return "pass"

    def _reasons(self, findings: Sequence[Finding], risk_score: float) -> list[str]:
        reasons: list[str] = []
        for finding in findings:
            rule = self.rules[finding.label]
            if finding.severity == "severe" and finding.confidence >= rule.fail_confidence:
                reasons.append(
                    f"Severe {finding.label} at {finding.confidence:.2f} confidence "
                    f"meets the {rule.fail_confidence:.2f} fail threshold."
                )
        if risk_score >= self.fail_risk_threshold:
            reasons.append(
                f"Aggregate risk {risk_score:.2f} reached the fail threshold "
                f"{self.fail_risk_threshold:.2f}."
            )
        if reasons:
            return reasons

        for finding in findings:
            rule = self.rules[finding.label]
            if finding.severity != "minor" or finding.confidence >= rule.review_confidence:
                reasons.append(
                    f"{finding.severity.capitalize()} {finding.label} in the {finding.zone} "
                    f"at {finding.confidence:.2f} confidence requires human review."
                )
        if reasons:
            return reasons

        if findings:
            return ["Minor cosmetic findings only; all are below human-review confidence."]
        return ["No defects above the class confidence floors."]
