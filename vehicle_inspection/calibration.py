"""Confidence calibration helpers implemented with the standard library.

Detector confidences are frequently over-confident.  Two complementary,
offline techniques are provided:

* **Temperature scaling** for per-class detector scores (a one-parameter
  logistic rescaling) and for full class-probability vectors.
* **Expected calibration error (ECE)** plus confidence bucketing, so the
  report can state how trustworthy a confidence band is.

No NumPy, SciPy, or sklearn is required; every value is computed with
:mod:`math`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .errors import CalibrationError

_EPSILON = 1e-12


def _validate_confidence(value: float, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalibrationError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CalibrationError(f"{path} must be within [0, 1], got {value!r}")
    return result


def softmax(logits: Sequence[float]) -> list[float]:
    """Return a numerically stable softmax over ``logits``."""
    if not logits:
        raise CalibrationError("softmax requires at least one logit")
    values: list[float] = []
    for index, logit in enumerate(logits):
        if not isinstance(logit, (int, float)) or isinstance(logit, bool):
            raise CalibrationError(f"logits[{index}] must be a number")
        if not math.isfinite(logit):
            raise CalibrationError(f"logits[{index}] must be finite")
        values.append(float(logit))
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def apply_temperature(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Rescale a probability vector with temperature ``T``.

    Probabilities are floored at :data:`_EPSILON` before converting to
    logits.  ``T > 1`` softens the distribution, ``T < 1`` sharpens it.
    Input values are renormalised, so small rounding drift from an upstream
    softmax is tolerated.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationError("temperature must be a positive finite number")
    if not probabilities:
        raise CalibrationError("probabilities must not be empty")

    total = 0.0
    checked: list[float] = []
    for index, probability in enumerate(probabilities):
        value = _validate_confidence(probability, f"probabilities[{index}]")
        checked.append(value)
        total += value
    if total <= 0.0:
        raise CalibrationError("probabilities must contain at least one non-zero value")

    logits = [math.log(max(value / total, _EPSILON)) / temperature for value in checked]
    return softmax(logits)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Return binned ECE for paired confidence / correctness values.

    Each sample contributes to a uniform-width confidence bin.  Empty bins
    are skipped.  The result is ``sum_bins (n_bin / N) * |accuracy - confidence|``.
    """
    if len(confidences) != len(correct):
        raise CalibrationError("confidences and correct must have equal length")
    if not confidences:
        raise CalibrationError("at least one confidence value is required")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise CalibrationError("n_bins must be a positive integer")

    bin_counts = [0] * n_bins
    bin_confidence = [0.0] * n_bins
    bin_correct = [0] * n_bins

    for index, confidence in enumerate(confidences):
        value = _validate_confidence(confidence, f"confidences[{index}]")
        if not isinstance(correct[index], bool):
            raise CalibrationError(f"correct[{index}] must be a boolean")
        bin_index = min(int(value * n_bins), n_bins - 1)
        bin_counts[bin_index] += 1
        bin_confidence[bin_index] += value
        if correct[index]:
            bin_correct[bin_index] += 1

    total_samples = len(confidences)
    error = 0.0
    for bin_index in range(n_bins):
        count = bin_counts[bin_index]
        if count == 0:
            continue
        average_confidence = bin_confidence[bin_index] / count
        accuracy = bin_correct[bin_index] / count
        error += (count / total_samples) * abs(accuracy - average_confidence)
    return error


def bucket_confidence(confidence: float) -> str:
    """Map a confidence to a human-readable reliability band."""
    value = _validate_confidence(confidence, "confidence")
    if value < 0.5:
        return "low"
    if value < 0.75:
        return "medium"
    if value < 0.9:
        return "high"
    return "very_high"


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    """Per-class temperature scaling for detector confidence scores.

    ``temperatures`` maps a class label to its fitted temperature; labels
    without an entry use ``default``.  Calibration is monotonic, so it never
    changes detection ranking — only the reported confidence and any policy
    decisions that depend on absolute thresholds.
    """

    temperatures: Mapping[str, float] = field(default_factory=dict)
    default: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.default, (int, float))
            or isinstance(self.default, bool)
            or not math.isfinite(self.default)
            or self.default <= 0.0
        ):
            raise CalibrationError("default temperature must be positive and finite")
        for label, temperature in self.temperatures.items():
            if not isinstance(label, str) or not label.strip():
                raise CalibrationError("calibration labels must be non-empty strings")
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or not math.isfinite(temperature)
                or temperature <= 0.0
            ):
                raise CalibrationError(f"temperature for {label!r} must be positive and finite")

    def calibrate(self, label: str, confidence: float) -> float:
        """Return the temperature-scaled confidence for ``label``.

        Exact ``0.0`` and ``1.0`` are preserved to avoid logit overflow and
        to keep perfect/none verdicts stable.
        """
        if not isinstance(label, str) or not label.strip():
            raise CalibrationError("label must be a non-empty string")
        value = _validate_confidence(confidence, "confidence")
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        temperature = self.temperatures.get(label, self.default)
        if temperature == 1.0:
            return value
        logit = math.log(value / (1.0 - value)) / temperature
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)
