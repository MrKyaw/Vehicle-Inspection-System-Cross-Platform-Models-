"""Validation of JSON model manifests.

A manifest is the contract between a trained inspection model and every
deployment surface (ONNX Runtime on a server, PyTorch on a workstation, or
ONNX Runtime Web compiled with the Rust/WASM preprocessor).  Parsing is
strict: unknown keys, wrong types, duplicate labels, and impossible
thresholds fail loudly instead of silently producing bad pre-processing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ManifestError

SUPPORTED_PLATFORMS = frozenset({"onnx", "pytorch", "wasm"})
SUPPORTED_COLOR_ORDERS = frozenset({"rgb", "bgr"})
SUPPORTED_LAYOUTS = frozenset({"nchw", "nhwc"})


def _is_number(value: Any) -> bool:
    """Return ``True`` for real numbers, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{path} must be a JSON object")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    for key in data:
        if key not in allowed:
            raise ManifestError(f"{path} contains unknown key {key!r}")


def _expect_positive_int(value: Any, path: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{path} must be a positive integer",
    )
    return value


def _float_tuple(value: Any, path: str) -> tuple[float, ...]:
    _require(isinstance(value, list), f"{path} must be a JSON array")
    result: list[float] = []
    for item in value:
        _require(_is_number(item) and math.isfinite(item), f"{path} entries must be finite numbers")
        result.append(float(item))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Preprocessing:
    """Input tensor contract for the exported model."""

    resize: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pad_value: int = 114
    color_order: str = "rgb"
    layout: str = "nchw"

    def __post_init__(self) -> None:
        if len(self.resize) != 2:
            raise ManifestError("preprocessing.resize must contain two positive integers")
        for index, dimension in enumerate(self.resize):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ManifestError(
                    f"preprocessing.resize[{index}] must be a positive integer"
                )
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ManifestError("preprocessing.mean and preprocessing.std must have three entries")
        for value in (*self.mean, *self.std):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ManifestError("preprocessing mean/std values must be numbers")
            if not math.isfinite(value):
                raise ManifestError("preprocessing mean/std values must be finite")
        if any(value <= 0.0 for value in self.std):
            raise ManifestError("preprocessing.std values must be positive")
        if (
            not isinstance(self.pad_value, int)
            or isinstance(self.pad_value, bool)
            or not 0 <= self.pad_value <= 255
        ):
            raise ManifestError("preprocessing.pad_value must be within [0, 255]")
        if self.color_order not in SUPPORTED_COLOR_ORDERS:
            raise ManifestError(
                f"preprocessing.color_order must be one of {sorted(SUPPORTED_COLOR_ORDERS)}"
            )
        if self.layout not in SUPPORTED_LAYOUTS:
            raise ManifestError(f"preprocessing.layout must be one of {sorted(SUPPORTED_LAYOUTS)}")

    @classmethod
    def from_dict(cls, data: Any) -> "Preprocessing":
        """Parse and validate a ``preprocessing`` object."""
        mapping = _expect_mapping(data, "preprocessing")
        _reject_unknown(
            mapping,
            frozenset({"resize", "mean", "std", "pad_value", "color_order", "layout"}),
            "preprocessing",
        )
        _require("resize" in mapping, "preprocessing.resize is required")
        _require("mean" in mapping, "preprocessing.mean is required")
        _require("std" in mapping, "preprocessing.std is required")

        raw_resize = mapping["resize"]
        _require(isinstance(raw_resize, list), "preprocessing.resize must be a JSON array")
        _require(len(raw_resize) == 2, "preprocessing.resize must contain exactly two entries")
        resize = (
            _expect_positive_int(raw_resize[0], "preprocessing.resize[0]"),
            _expect_positive_int(raw_resize[1], "preprocessing.resize[1]"),
        )
        mean = _float_tuple(mapping["mean"], "preprocessing.mean")
        std = _float_tuple(mapping["std"], "preprocessing.std")
        _require(len(mean) == 3, "preprocessing.mean must contain three entries")
        _require(len(std) == 3, "preprocessing.std must contain three entries")

        pad_value = _expect_positive_int_or_zero(
            mapping.get("pad_value", 114), "preprocessing.pad_value"
        )
        color_order = mapping.get("color_order", "rgb")
        layout = mapping.get("layout", "nchw")
        _require(isinstance(color_order, str), "preprocessing.color_order must be a string")
        _require(isinstance(layout, str), "preprocessing.layout must be a string")
        return cls(
            resize=resize,
            mean=(mean[0], mean[1], mean[2]),
            std=(std[0], std[1], std[2]),
            pad_value=pad_value,
            color_order=color_order,
            layout=layout,
        )


def _expect_positive_int_or_zero(value: Any, path: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255,
        f"{path} must be an integer within [0, 255]",
    )
    return value


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Score and decision thresholds shared by detection and policy."""

    score: float = 0.35
    iou: float = 0.5
    review: float = 0.6
    fail: float = 0.85

    def __post_init__(self) -> None:
        for name, value in (
            ("score", self.score),
            ("iou", self.iou),
            ("review", self.review),
            ("fail", self.fail),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ManifestError(f"thresholds.{name} must be a number")
            if not math.isfinite(value):
                raise ManifestError(f"thresholds.{name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ManifestError(f"thresholds.{name} must be within [0, 1]")
        if self.review >= self.fail:
            raise ManifestError("thresholds.review must be lower than thresholds.fail")

    @classmethod
    def from_dict(cls, data: Any) -> "Thresholds":
        """Parse and validate a ``thresholds`` object."""
        mapping = _expect_mapping(data, "thresholds")
        _reject_unknown(mapping, frozenset({"score", "iou", "review", "fail"}), "thresholds")
        values: dict[str, float] = {}
        for key, default in (
            ("score", 0.35),
            ("iou", 0.5),
            ("review", 0.6),
            ("fail", 0.85),
        ):
            raw = mapping.get(key, default)
            _require(_is_number(raw) and math.isfinite(raw), f"thresholds.{key} must be a finite number")
            values[key] = float(raw)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated description of an exported inspection model."""

    name: str
    version: str
    input_shape: tuple[int, int, int, int]
    labels: tuple[str, ...]
    preprocessing: Preprocessing
    thresholds: Thresholds
    platforms: tuple[str, ...]
    input_name: str = "input"
    output_name: str = "output"
    calibration: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ManifestError("name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ManifestError("version must be a non-empty string")
        if len(self.input_shape) != 4:
            raise ManifestError("input_shape must contain four dimensions (N, C, H, W)")
        for index, dimension in enumerate(self.input_shape):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ManifestError(f"input_shape[{index}] must be a positive integer")
        if self.input_shape[0] != 1:
            raise ManifestError("input_shape[0] (batch) must be 1 for inspection exports")
        if not self.labels:
            raise ManifestError("labels must not be empty")
        seen: set[str] = set()
        for label in self.labels:
            if not isinstance(label, str) or not label.strip():
                raise ManifestError("every label must be a non-empty string")
            if label in seen:
                raise ManifestError(f"duplicate label {label!r}")
            seen.add(label)
        if not self.platforms:
            raise ManifestError("platforms must not be empty")
        for platform in self.platforms:
            if platform not in SUPPORTED_PLATFORMS:
                raise ManifestError(
                    f"unsupported platform {platform!r}; expected one of {sorted(SUPPORTED_PLATFORMS)}"
                )
        if (
            not isinstance(self.input_name, str)
            or not self.input_name.strip()
            or not isinstance(self.output_name, str)
            or not self.output_name.strip()
        ):
            raise ManifestError("input_name and output_name must be non-empty strings")
        for label, temperature in self.calibration.items():
            if label not in seen:
                raise ManifestError(f"calibration label {label!r} is not present in labels")
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or not math.isfinite(temperature)
                or temperature <= 0.0
            ):
                raise ManifestError(f"calibration temperature for {label!r} must be positive")
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ManifestError("metadata keys and values must be strings")

    @classmethod
    def from_dict(cls, data: Any) -> "ModelManifest":
        """Parse a manifest from a decoded JSON object."""
        mapping = _expect_mapping(data, "manifest")
        _reject_unknown(
            mapping,
            frozenset(
                {
                    "name",
                    "version",
                    "input_shape",
                    "labels",
                    "preprocessing",
                    "thresholds",
                    "platforms",
                    "input_name",
                    "output_name",
                    "calibration",
                    "metadata",
                }
            ),
            "manifest",
        )
        for required in ("name", "version", "input_shape", "labels", "preprocessing", "platforms"):
            _require(required in mapping, f"manifest.{required} is required")

        name = mapping["name"]
        version = mapping["version"]
        _require(isinstance(name, str), "manifest.name must be a string")
        _require(isinstance(version, str), "manifest.version must be a string")

        raw_shape = mapping["input_shape"]
        _require(isinstance(raw_shape, list), "manifest.input_shape must be a JSON array")
        _require(len(raw_shape) == 4, "manifest.input_shape must contain four entries")
        input_shape = tuple(
            _expect_positive_int(item, f"manifest.input_shape[{index}]")
            for index, item in enumerate(raw_shape)
        )

        raw_labels = mapping["labels"]
        _require(isinstance(raw_labels, list), "manifest.labels must be a JSON array")
        labels = tuple(raw_labels)

        raw_platforms = mapping["platforms"]
        _require(isinstance(raw_platforms, list), "manifest.platforms must be a JSON array")
        platforms = tuple(raw_platforms)
        for platform in platforms:
            _require(isinstance(platform, str), "manifest.platforms entries must be strings")
        _require(
            len(set(platforms)) == len(platforms),
            "manifest.platforms must not contain duplicates",
        )

        preprocessing = Preprocessing.from_dict(mapping["preprocessing"])
        thresholds_data = mapping.get("thresholds", {})
        thresholds = Thresholds.from_dict(thresholds_data)

        raw_calibration = mapping.get("calibration", {})
        calibration_mapping = _expect_mapping(raw_calibration, "manifest.calibration")
        calibration: dict[str, float] = {}
        for label, temperature in calibration_mapping.items():
            _require(
                _is_number(temperature) and math.isfinite(temperature) and temperature > 0.0,
                f"manifest.calibration[{label!r}] must be a positive number",
            )
            calibration[label] = float(temperature)

        raw_metadata = mapping.get("metadata", {})
        metadata_mapping = _expect_mapping(raw_metadata, "manifest.metadata")
        metadata: dict[str, str] = {}
        for key, value in metadata_mapping.items():
            _require(
                isinstance(key, str) and isinstance(value, str),
                "manifest.metadata keys and values must be strings",
            )
            metadata[key] = value

        input_name = mapping.get("input_name", "input")
        output_name = mapping.get("output_name", "output")
        _require(isinstance(input_name, str), "manifest.input_name must be a string")
        _require(isinstance(output_name, str), "manifest.output_name must be a string")

        return cls(
            name=name,
            version=version,
            input_shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
            labels=labels,
            preprocessing=preprocessing,
            thresholds=thresholds,
            platforms=platforms,
            input_name=input_name,
            output_name=output_name,
            calibration=MappingProxyType(calibration),
            metadata=MappingProxyType(metadata),
        )


def load_manifest(path: str | Path) -> ModelManifest:
    """Load and validate a manifest from ``path``.

    File-system and JSON errors are normalised to :class:`ManifestError` so
    CLI callers only need to handle one exception family.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not read manifest {source}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {source} is not valid JSON: {exc.msg}") from exc
    return ModelManifest.from_dict(data)
