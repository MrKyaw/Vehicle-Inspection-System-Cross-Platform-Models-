"""Tests for the dependency-free ``vehicle_inspection`` package.

Coverage target: 100% line and branch coverage of ``vehicle_inspection``.
Every public function is exercised for happy paths, boundaries, and malformed
input.  No test touches the network or requires optional dependencies.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import runpy
import sys

import pytest

from vehicle_inspection import (
    DEFAULT_RULES,
    REPORT_SCHEMA_VERSION,
    Box,
    CalibrationError,
    ClassRule,
    Detection,
    Finding,
    GeometryError,
    IgnoredDetection,
    InspectionPolicy,
    InspectionResult,
    LetterboxTransform,
    ManifestError,
    ModelManifest,
    PolicyError,
    Preprocessing,
    ReportError,
    TemperatureCalibrator,
    Thresholds,
    VehicleInspectionError,
    __version__,
    apply_temperature,
    bucket_confidence,
    build_demo,
    build_report,
    clamp_box,
    expected_calibration_error,
    iou,
    letterbox,
    load_manifest,
    locate_zone,
    non_max_suppression,
    render_markdown,
    report_to_json,
    softmax,
)
from vehicle_inspection.cli import (
    DetectionFile,
    build_parser,
    load_detection_file,
    main,
)
from vehicle_inspection.report import ReportError as ReportModuleError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def manifest_kwargs() -> dict[str, object]:
    return {
        "name": "vehicle-damage",
        "version": "1.0.0",
        "input_shape": (1, 3, 640, 640),
        "labels": ("scratch", "dent", "crack"),
        "preprocessing": Preprocessing(
            resize=(640, 640),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        "thresholds": Thresholds(),
        "platforms": ("onnx", "pytorch", "wasm"),
        "calibration": {"crack": 1.2},
        "metadata": {"task": "inspection"},
    }


def make_manifest(**overrides: object) -> ModelManifest:
    values = manifest_kwargs()
    values.update(overrides)
    return ModelManifest(**values)  # type: ignore[arg-type]


def manifest_payload() -> dict[str, object]:
    return {
        "name": "vehicle-damage",
        "version": "1.0.0",
        "input_shape": [1, 3, 640, 640],
        "labels": ["scratch", "dent", "crack"],
        "preprocessing": {
            "resize": [640, 640],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "pad_value": 114,
            "color_order": "rgb",
            "layout": "nchw",
        },
        "thresholds": {"score": 0.35, "iou": 0.5, "review": 0.6, "fail": 0.85},
        "platforms": ["onnx", "pytorch", "wasm"],
        "calibration": {"crack": 1.2},
        "metadata": {"task": "inspection"},
    }


def detection(
    label: str = "scratch",
    confidence: float = 0.8,
    box: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 50.0),
    severity: str = "minor",
) -> Detection:
    return Detection(label, confidence, Box(*box), severity)


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def detection_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": "scratch",
        "confidence": 0.8,
        "box": [10.0, 10.0, 50.0, 50.0],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestBox:
    def test_properties(self) -> None:
        box = Box(10.0, 20.0, 40.0, 60.0)
        assert box.width == 30.0
        assert box.height == 40.0
        assert box.area == 1200.0
        assert box.center == (25.0, 40.0)
        assert box.as_tuple() == (10.0, 20.0, 40.0, 60.0)

    def test_zero_area_is_allowed(self) -> None:
        assert Box(5.0, 5.0, 5.0, 9.0).area == 0.0

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_non_finite_coordinates_rejected(self, index: int) -> None:
        coordinates = [0.0, 0.0, 1.0, 1.0]
        coordinates[index] = float("nan") if index == 2 else float("inf")
        with pytest.raises(GeometryError, match="finite"):
            Box(*coordinates)

    def test_inverted_x_rejected(self) -> None:
        with pytest.raises(GeometryError, match="x2 >= x1"):
            Box(10, 0, 5, 5)

    def test_inverted_y_rejected(self) -> None:
        with pytest.raises(GeometryError, match="y2 >= y1"):
            Box(0, 10, 5, 5)

    def test_nan_constructed_raises(self) -> None:
        with pytest.raises(GeometryError):
            Box(float("nan"), 0, 1, 1)

    @pytest.mark.parametrize("value", ["1", None, True])
    def test_non_numeric_coordinate_rejected(self, value: object) -> None:
        with pytest.raises(GeometryError, match="must be a number"):
            Box(value, 0, 1, 1)  # type: ignore[arg-type]


class TestClamp:
    def test_inside_unchanged(self) -> None:
        assert clamp_box(Box(1, 2, 3, 4), 10, 10) == Box(1, 2, 3, 4)

    def test_clips_all_edges(self) -> None:
        assert clamp_box(Box(-5, -6, 15, 16), 10, 12) == Box(0, 0, 10, 12)

    def test_fully_outside_collapses(self) -> None:
        clamped = clamp_box(Box(-20, -20, -10, -10), 10, 10)
        assert clamped == Box(0, 0, 0, 0)
        assert clamped.area == 0.0

    @pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-1, 10)])
    def test_invalid_image_dimensions(self, width: int, height: int) -> None:
        with pytest.raises(GeometryError, match="positive"):
            clamp_box(Box(0, 0, 1, 1), width, height)


class TestIoU:
    def test_identical(self) -> None:
        assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == 1.0

    def test_disjoint(self) -> None:
        assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0

    def test_touching_edges(self) -> None:
        assert iou(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == 0.0

    def test_partial_overlap(self) -> None:
        assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 5)) == pytest.approx(0.5)

    def test_degenerate_pair_returns_zero(self) -> None:
        assert iou(Box(5, 5, 5, 5), Box(5, 5, 5, 5)) == 0.0

    def test_degenerate_against_area_box(self) -> None:
        assert iou(Box(5, 5, 5, 5), Box(0, 0, 10, 10)) == 0.0


class TestNMS:
    def test_empty(self) -> None:
        assert non_max_suppression([], []) == []

    def test_length_mismatch(self) -> None:
        with pytest.raises(GeometryError, match="equal length"):
            non_max_suppression([Box(0, 0, 1, 1)], [])

    @pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
    def test_invalid_iou_threshold(self, threshold: float) -> None:
        with pytest.raises(GeometryError, match="iou_threshold"):
            non_max_suppression([Box(0, 0, 1, 1)], [0.9], iou_threshold=threshold)

    def test_non_finite_score(self) -> None:
        with pytest.raises(GeometryError, match="finite"):
            non_max_suppression([Box(0, 0, 1, 1)], [float("inf")])

    def test_suppresses_overlap(self) -> None:
        boxes = [Box(0, 0, 10, 10), Box(1, 1, 11, 11), Box(50, 50, 60, 60)]
        kept = non_max_suppression(boxes, [0.9, 0.8, 0.7])
        assert kept == [0, 2]

    def test_score_threshold_skips(self) -> None:
        boxes = [Box(0, 0, 10, 10), Box(50, 50, 60, 60)]
        assert non_max_suppression(boxes, [0.2, 0.1], score_threshold=0.15) == [0]

    def test_strict_iou_boundary_is_kept(self) -> None:
        boxes = [Box(0, 0, 10, 10), Box(0, 0, 10, 5)]
        assert non_max_suppression(boxes, [0.9, 0.8], iou_threshold=0.5) == [0, 1]

    def test_ties_broken_by_index(self) -> None:
        boxes = [Box(0, 0, 10, 10), Box(0, 0, 10, 10), Box(50, 50, 60, 60)]
        assert non_max_suppression(boxes, [0.8, 0.8, 0.8]) == [0, 2]

    def test_equal_scores_without_overlap_all_kept(self) -> None:
        boxes = [Box(50, 50, 60, 60), Box(0, 0, 10, 10)]
        assert non_max_suppression(boxes, [0.5, 0.5]) == [0, 1]

    def test_returns_descending_score_order(self) -> None:
        boxes = [Box(0, 0, 1, 1), Box(10, 10, 11, 11), Box(20, 20, 21, 21)]
        assert non_max_suppression(boxes, [0.1, 0.9, 0.5]) == [1, 2, 0]


class TestLetterbox:
    def test_landscape_to_square(self) -> None:
        transform = letterbox(1280, 720, 640, 640)
        assert isinstance(transform, LetterboxTransform)
        assert transform.scale == 0.5
        assert (transform.new_width, transform.new_height) == (640, 360)
        assert transform.pad_x == 0.0
        assert transform.pad_y == 140.0

    def test_portrait_to_square(self) -> None:
        transform = letterbox(720, 1280, 640, 640)
        assert (transform.new_width, transform.new_height) == (360, 640)
        assert transform.pad_x == 140.0
        assert transform.pad_y == 0.0

    def test_exact_size(self) -> None:
        transform = letterbox(640, 640, 640, 640)
        assert transform.scale == 1.0
        assert transform.pad_x == 0.0
        assert transform.pad_y == 0.0

    def test_extreme_aspect_preserves_one_pixel_width(self) -> None:
        transform = letterbox(1, 1000, 10, 10)
        assert transform.new_width == 1
        assert transform.new_height == 10

    def test_extreme_aspect_preserves_one_pixel_height(self) -> None:
        transform = letterbox(1000, 1, 10, 10)
        assert transform.new_width == 10
        assert transform.new_height == 1

    def test_inverse_mapping(self) -> None:
        transform = letterbox(1280, 720, 640, 640)
        canvas_box = Box(0.0, 140.0, 320.0, 320.0)
        original = transform.to_original_box(canvas_box)
        assert original == Box(0.0, 0.0, 640.0, 360.0)

    def test_inverse_mapping_clamps_outside_canvas(self) -> None:
        transform = letterbox(1280, 720, 640, 640)
        original = transform.to_original_box(Box(-100.0, -100.0, 5000.0, 5000.0))
        assert original == Box(0.0, 0.0, 1280.0, 720.0)

    @pytest.mark.parametrize(
        "args",
        [
            (0, 10, 10, 10),
            (10, 0, 10, 10),
            (10, 10, 0, 10),
            (10, 10, 10, 0),
        ],
    )
    def test_invalid_dimensions(self, args: tuple[int, int, int, int]) -> None:
        with pytest.raises(GeometryError, match="positive"):
            letterbox(*args)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_valid(self) -> None:
        preprocessing = Preprocessing((640, 640), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        assert preprocessing.pad_value == 114
        assert preprocessing.color_order == "rgb"
        assert preprocessing.layout == "nchw"

    def test_bgr_nhwc(self) -> None:
        preprocessing = Preprocessing(
            (32, 64), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), pad_value=0, color_order="bgr", layout="nhwc"
        )
        assert (preprocessing.color_order, preprocessing.layout) == ("bgr", "nhwc")

    def test_bad_resize_length(self) -> None:
        with pytest.raises(ManifestError, match="resize"):
            Preprocessing((640,), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))

    def test_bad_resize_zero(self) -> None:
        with pytest.raises(ManifestError, match="resize"):
            Preprocessing((0, 640), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))

    def test_bad_resize_type(self) -> None:
        with pytest.raises(ManifestError, match="resize"):
            Preprocessing(("640", "640"), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))  # type: ignore[arg-type]

    def test_bad_mean_type(self) -> None:
        with pytest.raises(ManifestError, match="numbers"):
            Preprocessing((640, 640), ("0.5", 0.5, 0.5), (0.2, 0.2, 0.2))  # type: ignore[arg-type]

    def test_bad_padding_type(self) -> None:
        with pytest.raises(ManifestError, match="pad_value"):
            Preprocessing(
                (640, 640), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), pad_value="114"  # type: ignore[arg-type]
            )

    def test_bad_mean_length(self) -> None:
        with pytest.raises(ManifestError, match="mean"):
            Preprocessing((640, 640), (0.5, 0.5), (0.2, 0.2, 0.2))

    def test_bad_std_length(self) -> None:
        with pytest.raises(ManifestError, match="mean"):
            Preprocessing((640, 640), (0.5, 0.5, 0.5), (0.2, 0.2))

    def test_non_finite_mean(self) -> None:
        with pytest.raises(ManifestError, match="finite"):
            Preprocessing((640, 640), (float("nan"), 0.5, 0.5), (0.2, 0.2, 0.2))

    def test_non_positive_std(self) -> None:
        with pytest.raises(ManifestError, match="std"):
            Preprocessing((640, 640), (0.5, 0.5, 0.5), (0.0, 0.2, 0.2))

    def test_bad_padding(self) -> None:
        with pytest.raises(ManifestError, match="pad_value"):
            Preprocessing((640, 640), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), pad_value=256)

    def test_bad_color_order(self) -> None:
        with pytest.raises(ManifestError, match="color_order"):
            Preprocessing(
                (640, 640), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), color_order="cmyk"
            )

    def test_bad_layout(self) -> None:
        with pytest.raises(ManifestError, match="layout"):
            Preprocessing((640, 640), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), layout="hwc")

    def test_from_dict_valid(self) -> None:
        preprocessing = Preprocessing.from_dict(manifest_payload()["preprocessing"])
        assert preprocessing.resize == (640, 640)
        assert preprocessing.pad_value == 114

    def test_from_dict_defaults(self) -> None:
        preprocessing = Preprocessing.from_dict(
            {"resize": [32, 32], "mean": [0, 0, 0], "std": [1, 1, 1]}
        )
        assert preprocessing.pad_value == 114
        assert preprocessing.color_order == "rgb"
        assert preprocessing.layout == "nchw"

    def test_from_dict_not_object(self) -> None:
        with pytest.raises(ManifestError, match="JSON object"):
            Preprocessing.from_dict([])

    def test_from_dict_unknown_key(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["gamma"] = 2.2
        with pytest.raises(ManifestError, match="unknown key"):
            Preprocessing.from_dict(payload)

    @pytest.mark.parametrize("key", ["resize", "mean", "std"])
    def test_from_dict_missing_required(self, key: str) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        del payload[key]
        with pytest.raises(ManifestError, match=f"{key} is required"):
            Preprocessing.from_dict(payload)

    def test_from_dict_resize_not_array(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["resize"] = 640
        with pytest.raises(ManifestError, match="resize"):
            Preprocessing.from_dict(payload)

    def test_from_dict_resize_wrong_length(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["resize"] = [640]
        with pytest.raises(ManifestError, match="resize"):
            Preprocessing.from_dict(payload)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "640"])
    def test_from_dict_resize_invalid_entry(self, value: object) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["resize"] = [value, 640]
        with pytest.raises(ManifestError, match="resize"):
            Preprocessing.from_dict(payload)

    def test_from_dict_mean_not_array(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["mean"] = 0.5
        with pytest.raises(ManifestError, match="mean"):
            Preprocessing.from_dict(payload)

    def test_from_dict_mean_non_number(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["mean"] = [0.5, "x", 0.5]
        with pytest.raises(ManifestError, match="mean"):
            Preprocessing.from_dict(payload)

    def test_from_dict_mean_non_finite(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["mean"] = [0.5, float("inf"), 0.5]
        with pytest.raises(ManifestError, match="finite"):
            Preprocessing.from_dict(payload)

    def test_from_dict_mean_wrong_length(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["mean"] = [0.5, 0.5]
        with pytest.raises(ManifestError, match="three entries"):
            Preprocessing.from_dict(payload)

    def test_from_dict_std_wrong_length(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["std"] = [0.5]
        with pytest.raises(ManifestError, match="three entries"):
            Preprocessing.from_dict(payload)

    @pytest.mark.parametrize("value", [-1, 256, True, "114"])
    def test_from_dict_bad_padding(self, value: object) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["pad_value"] = value
        with pytest.raises(ManifestError, match="pad_value"):
            Preprocessing.from_dict(payload)

    def test_from_dict_non_string_color(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["color_order"] = 1
        with pytest.raises(ManifestError, match="color_order"):
            Preprocessing.from_dict(payload)

    def test_from_dict_non_string_layout(self) -> None:
        payload = copy.deepcopy(manifest_payload()["preprocessing"])
        payload["layout"] = 1
        with pytest.raises(ManifestError, match="layout"):
            Preprocessing.from_dict(payload)


class TestThresholds:
    def test_defaults(self) -> None:
        thresholds = Thresholds()
        assert (thresholds.score, thresholds.iou, thresholds.review, thresholds.fail) == (
            0.35,
            0.5,
            0.6,
            0.85,
        )

    def test_from_dict(self) -> None:
        thresholds = Thresholds.from_dict({"score": 0.4, "iou": 0.6, "review": 0.5, "fail": 0.9})
        assert thresholds.score == 0.4
        assert thresholds.fail == 0.9

    def test_from_dict_defaults(self) -> None:
        thresholds = Thresholds.from_dict({})
        assert thresholds == Thresholds()

    def test_from_dict_unknown_key(self) -> None:
        with pytest.raises(ManifestError, match="unknown key"):
            Thresholds.from_dict({"min": 0.5})

    def test_from_dict_not_object(self) -> None:
        with pytest.raises(ManifestError, match="JSON object"):
            Thresholds.from_dict(0.5)

    @pytest.mark.parametrize("value", ["0.5", True, None, float("nan")])
    def test_from_dict_invalid_value(self, value: object) -> None:
        with pytest.raises(ManifestError, match="finite number"):
            Thresholds.from_dict({"score": value})

    @pytest.mark.parametrize("name", ["score", "iou", "review", "fail"])
    def test_out_of_range(self, name: str) -> None:
        with pytest.raises(ManifestError, match="within"):
            Thresholds.from_dict({name: 1.5})

    def test_non_finite(self) -> None:
        with pytest.raises(ManifestError, match="finite"):
            Thresholds(score=float("inf"))

    def test_non_number(self) -> None:
        with pytest.raises(ManifestError, match="number"):
            Thresholds(score="0.5")  # type: ignore[arg-type]

    def test_review_must_be_lower_than_fail(self) -> None:
        with pytest.raises(ManifestError, match="review must be lower"):
            Thresholds(review=0.9, fail=0.9)


class TestModelManifest:
    def test_valid(self) -> None:
        manifest = make_manifest()
        assert manifest.name == "vehicle-damage"
        assert manifest.input_name == "input"
        assert manifest.output_name == "output"
        assert manifest.calibration["crack"] == 1.2

    def test_blank_name(self) -> None:
        with pytest.raises(ManifestError, match="name"):
            make_manifest(name="  ")

    @pytest.mark.parametrize("field", ["name", "version", "input_name", "output_name"])
    def test_non_string_identifier(self, field: str) -> None:
        with pytest.raises(ManifestError):
            make_manifest(**{field: 1})

    def test_blank_version(self) -> None:
        with pytest.raises(ManifestError, match="version"):
            make_manifest(version="")

    def test_wrong_shape_length(self) -> None:
        with pytest.raises(ManifestError, match="four dimensions"):
            make_manifest(input_shape=(1, 3, 640))

    @pytest.mark.parametrize("dimension", [0, -3, True, 3.5])
    def test_invalid_shape_dimension(self, dimension: object) -> None:
        with pytest.raises(ManifestError, match="positive integer"):
            make_manifest(input_shape=(1, 3, dimension, 640))

    def test_batch_must_be_one(self) -> None:
        with pytest.raises(ManifestError, match="batch"):
            make_manifest(input_shape=(2, 3, 640, 640))

    def test_empty_labels(self) -> None:
        with pytest.raises(ManifestError, match="labels"):
            make_manifest(labels=())

    def test_non_string_label(self) -> None:
        with pytest.raises(ManifestError, match="non-empty string"):
            make_manifest(labels=("scratch", 1))

    def test_blank_label(self) -> None:
        with pytest.raises(ManifestError, match="non-empty string"):
            make_manifest(labels=("scratch", " "))

    def test_duplicate_label(self) -> None:
        with pytest.raises(ManifestError, match="duplicate"):
            make_manifest(labels=("scratch", "scratch"))

    def test_empty_platforms(self) -> None:
        with pytest.raises(ManifestError, match="platforms"):
            make_manifest(platforms=())

    def test_unsupported_platform(self) -> None:
        with pytest.raises(ManifestError, match="unsupported platform"):
            make_manifest(platforms=("tensorrt",))

    def test_blank_input_name(self) -> None:
        with pytest.raises(ManifestError, match="input_name"):
            make_manifest(input_name=" ")

    def test_blank_output_name(self) -> None:
        with pytest.raises(ManifestError, match="output_name"):
            make_manifest(output_name=" ")

    def test_calibration_label_must_exist(self) -> None:
        with pytest.raises(ManifestError, match="calibration label"):
            make_manifest(calibration={"rust": 1.2})

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
    def test_calibration_temperature_must_be_positive(self, temperature: float) -> None:
        with pytest.raises(ManifestError, match="temperature"):
            make_manifest(calibration={"crack": temperature})

    def test_calibration_temperature_must_be_numeric(self) -> None:
        with pytest.raises(ManifestError, match="temperature"):
            make_manifest(calibration={"crack": "1.2"})  # type: ignore[dict-item]

    def test_metadata_must_be_strings(self) -> None:
        with pytest.raises(ManifestError, match="metadata"):
            make_manifest(metadata={"task": 1})

    def test_from_dict_roundtrip(self) -> None:
        manifest = ModelManifest.from_dict(manifest_payload())
        assert manifest.labels == ("scratch", "dent", "crack")
        assert manifest.thresholds.review == 0.6
        assert dict(manifest.metadata) == {"task": "inspection"}

    def test_from_dict_minimal(self) -> None:
        payload = manifest_payload()
        for key in ("thresholds", "calibration", "metadata", "input_name", "output_name"):
            if key in payload:
                payload.pop(key)
        manifest = ModelManifest.from_dict(payload)
        assert manifest.thresholds == Thresholds()
        assert dict(manifest.calibration) == {}
        assert manifest.input_name == "input"

    def test_from_dict_not_object(self) -> None:
        with pytest.raises(ManifestError, match="JSON object"):
            ModelManifest.from_dict([])

    def test_from_dict_unknown_key(self) -> None:
        payload = manifest_payload()
        payload["optimizer"] = "sgd"
        with pytest.raises(ManifestError, match="unknown key"):
            ModelManifest.from_dict(payload)

    @pytest.mark.parametrize(
        "key", ["name", "version", "input_shape", "labels", "preprocessing", "platforms"]
    )
    def test_from_dict_missing_required(self, key: str) -> None:
        payload = manifest_payload()
        del payload[key]
        with pytest.raises(ManifestError, match=f"{key} is required"):
            ModelManifest.from_dict(payload)

    def test_from_dict_non_string_name(self) -> None:
        payload = manifest_payload()
        payload["name"] = 1
        with pytest.raises(ManifestError, match="name"):
            ModelManifest.from_dict(payload)

    def test_from_dict_non_string_version(self) -> None:
        payload = manifest_payload()
        payload["version"] = 1
        with pytest.raises(ManifestError, match="version"):
            ModelManifest.from_dict(payload)

    def test_from_dict_shape_not_array(self) -> None:
        payload = manifest_payload()
        payload["input_shape"] = "1,3,640,640"
        with pytest.raises(ManifestError, match="JSON array"):
            ModelManifest.from_dict(payload)

    def test_from_dict_shape_wrong_length(self) -> None:
        payload = manifest_payload()
        payload["input_shape"] = [1, 3, 640]
        with pytest.raises(ManifestError, match="four entries"):
            ModelManifest.from_dict(payload)

    def test_from_dict_shape_invalid_entry(self) -> None:
        payload = manifest_payload()
        payload["input_shape"] = [1, 3, 640, "640"]
        with pytest.raises(ManifestError, match="input_shape"):
            ModelManifest.from_dict(payload)

    def test_from_dict_labels_not_array(self) -> None:
        payload = manifest_payload()
        payload["labels"] = "scratch"
        with pytest.raises(ManifestError, match="labels"):
            ModelManifest.from_dict(payload)

    def test_from_dict_labels_bad_entries(self) -> None:
        payload = manifest_payload()
        payload["labels"] = ["scratch", 1]
        with pytest.raises(ManifestError, match="non-empty string"):
            ModelManifest.from_dict(payload)

    def test_from_dict_platforms_not_array(self) -> None:
        payload = manifest_payload()
        payload["platforms"] = "onnx"
        with pytest.raises(ManifestError, match="platforms"):
            ModelManifest.from_dict(payload)

    def test_from_dict_platform_entry_not_string(self) -> None:
        payload = manifest_payload()
        payload["platforms"] = ["onnx", 1]
        with pytest.raises(ManifestError, match="platforms entries"):
            ModelManifest.from_dict(payload)

    def test_from_dict_duplicate_platform(self) -> None:
        payload = manifest_payload()
        payload["platforms"] = ["onnx", "onnx"]
        with pytest.raises(ManifestError, match="duplicates"):
            ModelManifest.from_dict(payload)

    def test_from_dict_calibration_not_object(self) -> None:
        payload = manifest_payload()
        payload["calibration"] = []
        with pytest.raises(ManifestError, match="calibration"):
            ModelManifest.from_dict(payload)

    @pytest.mark.parametrize("value", [0, -1, True, "1.2", float("inf")])
    def test_from_dict_bad_calibration_value(self, value: object) -> None:
        payload = manifest_payload()
        payload["calibration"] = {"crack": value}
        with pytest.raises(ManifestError, match="positive number"):
            ModelManifest.from_dict(payload)

    def test_from_dict_metadata_not_object(self) -> None:
        payload = manifest_payload()
        payload["metadata"] = []
        with pytest.raises(ManifestError, match="metadata"):
            ModelManifest.from_dict(payload)

    def test_from_dict_metadata_non_string_value(self) -> None:
        payload = manifest_payload()
        payload["metadata"] = {"task": 1}
        with pytest.raises(ManifestError, match="keys and values"):
            ModelManifest.from_dict(payload)

    def test_from_dict_input_name_not_string(self) -> None:
        payload = manifest_payload()
        payload["input_name"] = 1
        with pytest.raises(ManifestError, match="input_name"):
            ModelManifest.from_dict(payload)

    def test_from_dict_output_name_not_string(self) -> None:
        payload = manifest_payload()
        payload["output_name"] = 1
        with pytest.raises(ManifestError, match="output_name"):
            ModelManifest.from_dict(payload)


class TestLoadManifest:
    def test_valid_file(self, tmp_path) -> None:
        path = tmp_path / "manifest.json"
        write_json(path, manifest_payload())
        manifest = load_manifest(path)
        assert manifest.name == "vehicle-damage"

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(ManifestError, match="could not read"):
            load_manifest(tmp_path / "missing.json")

    def test_directory_path(self, tmp_path) -> None:
        with pytest.raises(ManifestError, match="could not read"):
            load_manifest(tmp_path)

    def test_invalid_json(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestError, match="not valid JSON"):
            load_manifest(path)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_normal(self) -> None:
        result = softmax([1.0, 2.0, 3.0])
        assert sum(result) == pytest.approx(1.0)
        assert result[2] > result[1] > result[0]

    def test_extreme_values_are_stable(self) -> None:
        result = softmax([1000.0, 1001.0])
        assert sum(result) == pytest.approx(1.0)
        assert all(math.isfinite(value) for value in result)

    def test_empty(self) -> None:
        with pytest.raises(CalibrationError, match="at least one"):
            softmax([])

    def test_non_number(self) -> None:
        with pytest.raises(CalibrationError, match="must be a number"):
            softmax([1.0, "2"])

    def test_non_finite(self) -> None:
        with pytest.raises(CalibrationError, match="finite"):
            softmax([float("inf")])


class TestApplyTemperature:
    def test_temperature_one_matches_input(self) -> None:
        probabilities = [0.2, 0.3, 0.5]
        assert apply_temperature(probabilities, 1.0) == pytest.approx(probabilities, abs=1e-9)

    def test_high_temperature_softens(self) -> None:
        original = [0.9, 0.1]
        calibrated = apply_temperature(original, 2.0)
        assert calibrated[0] < original[0]
        assert calibrated[0] > calibrated[1]

    def test_low_temperature_sharpens(self) -> None:
        original = [0.6, 0.4]
        calibrated = apply_temperature(original, 0.5)
        assert calibrated[0] > original[0]

    def test_renormalises_drifted_input(self) -> None:
        calibrated = apply_temperature([0.4, 0.4], 1.0)
        assert calibrated == pytest.approx([0.5, 0.5])

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_temperature(self, temperature: float) -> None:
        with pytest.raises(CalibrationError, match="temperature"):
            apply_temperature([0.5, 0.5], temperature)

    def test_empty_probabilities(self) -> None:
        with pytest.raises(CalibrationError, match="not be empty"):
            apply_temperature([], 1.0)

    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), "0.5", True])
    def test_invalid_probability(self, value: object) -> None:
        with pytest.raises(CalibrationError):
            apply_temperature([value, 0.5], 1.0)

    def test_all_zero_probabilities(self) -> None:
        with pytest.raises(CalibrationError, match="non-zero"):
            apply_temperature([0.0, 0.0], 1.0)


class TestExpectedCalibrationError:
    def test_perfect_calibration(self) -> None:
        assert expected_calibration_error([0.0, 0.0, 1.0, 1.0], [False, False, True, True]) == 0.0

    def test_biased_confidence(self) -> None:
        error = expected_calibration_error([0.9, 0.9, 0.9, 0.9], [False, False, False, True])
        assert error == pytest.approx(0.65)

    def test_empty_bins_are_skipped(self) -> None:
        error = expected_calibration_error([0.95], [True], n_bins=10)
        assert error == pytest.approx(0.05)

    def test_confidence_one_uses_last_bin(self) -> None:
        assert expected_calibration_error([1.0], [True], n_bins=4) == 0.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(CalibrationError, match="equal length"):
            expected_calibration_error([0.5], [])

    def test_empty(self) -> None:
        with pytest.raises(CalibrationError, match="at least one"):
            expected_calibration_error([], [])

    @pytest.mark.parametrize("n_bins", [0, -1, 2.5, True])
    def test_invalid_bin_count(self, n_bins: object) -> None:
        with pytest.raises(CalibrationError, match="n_bins"):
            expected_calibration_error([0.5], [True], n_bins=n_bins)

    def test_invalid_confidence(self) -> None:
        with pytest.raises(CalibrationError, match="confidences"):
            expected_calibration_error([1.5], [True])

    def test_non_boolean_correctness(self) -> None:
        with pytest.raises(CalibrationError, match="boolean"):
            expected_calibration_error([0.5], [1])


class TestBucketConfidence:
    @pytest.mark.parametrize(
        "confidence,expected",
        [
            (0.0, "low"),
            (0.499, "low"),
            (0.5, "medium"),
            (0.749, "medium"),
            (0.75, "high"),
            (0.899, "high"),
            (0.9, "very_high"),
            (1.0, "very_high"),
        ],
    )
    def test_buckets(self, confidence: float, expected: str) -> None:
        assert bucket_confidence(confidence) == expected

    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), "0.5"])
    def test_invalid(self, value: object) -> None:
        with pytest.raises(CalibrationError):
            bucket_confidence(value)


class TestTemperatureCalibrator:
    def test_present_label(self) -> None:
        calibrator = TemperatureCalibrator({"scratch": 2.0})
        assert calibrator.calibrate("scratch", 0.9) == pytest.approx(0.75, abs=1e-6)

    def test_default_label(self) -> None:
        calibrator = TemperatureCalibrator({"scratch": 2.0}, default=1.0)
        assert calibrator.calibrate("dent", 0.9) == 0.9

    def test_low_confidence_uses_stable_branch(self) -> None:
        calibrator = TemperatureCalibrator(default=2.0)
        assert calibrator.calibrate("scratch", 0.1) == pytest.approx(0.25, abs=1e-6)

    def test_high_confidence_uses_stable_branch(self) -> None:
        calibrator = TemperatureCalibrator(default=2.0)
        assert calibrator.calibrate("scratch", 0.99) == pytest.approx(0.9087, abs=1e-3)

    def test_zero_and_one_preserved(self) -> None:
        calibrator = TemperatureCalibrator({"scratch": 3.0})
        assert calibrator.calibrate("scratch", 0.0) == 0.0
        assert calibrator.calibrate("scratch", 1.0) == 1.0

    def test_blank_label(self) -> None:
        with pytest.raises(CalibrationError, match="label"):
            TemperatureCalibrator().calibrate(" ", 0.5)

    def test_invalid_confidence(self) -> None:
        with pytest.raises(CalibrationError):
            TemperatureCalibrator().calibrate("scratch", -0.1)

    @pytest.mark.parametrize("default", [0.0, -1.0, float("inf")])
    def test_invalid_default(self, default: float) -> None:
        with pytest.raises(CalibrationError, match="default temperature"):
            TemperatureCalibrator(default=default)

    def test_non_numeric_default(self) -> None:
        with pytest.raises(CalibrationError, match="default temperature"):
            TemperatureCalibrator(default="1.0")  # type: ignore[arg-type]

    def test_invalid_label_in_mapping(self) -> None:
        with pytest.raises(CalibrationError, match="non-empty"):
            TemperatureCalibrator({"": 1.0})

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
    def test_invalid_temperature_in_mapping(self, temperature: float) -> None:
        with pytest.raises(CalibrationError, match="temperature"):
            TemperatureCalibrator({"scratch": temperature})

    def test_non_numeric_temperature_in_mapping(self) -> None:
        with pytest.raises(CalibrationError, match="temperature"):
            TemperatureCalibrator({"scratch": "2.0"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TestDetection:
    def test_valid_defaults(self) -> None:
        item = Detection("scratch", 0.5, Box(0, 0, 1, 1))
        assert item.severity == "minor"

    @pytest.mark.parametrize("label", ["", "  ", 1])
    def test_invalid_label(self, label: object) -> None:
        with pytest.raises(PolicyError, match="label"):
            Detection(label, 0.5, Box(0, 0, 1, 1))

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), "0.5", True])
    def test_invalid_confidence(self, confidence: object) -> None:
        with pytest.raises(PolicyError, match="confidence"):
            Detection("scratch", confidence, Box(0, 0, 1, 1))

    def test_invalid_box_type(self) -> None:
        with pytest.raises(PolicyError, match="Box"):
            Detection("scratch", 0.5, (0, 0, 1, 1))

    def test_invalid_severity(self) -> None:
        with pytest.raises(PolicyError, match="severity"):
            Detection("scratch", 0.5, Box(0, 0, 1, 1), "critical")

    def test_from_dict(self) -> None:
        item = Detection.from_dict(detection_payload(severity="severe"))
        assert item.label == "scratch"
        assert item.severity == "severe"
        assert item.box == Box(10, 10, 50, 50)

    def test_from_dict_default_severity(self) -> None:
        assert Detection.from_dict(detection_payload()).severity == "minor"

    def test_from_dict_not_object(self) -> None:
        with pytest.raises(PolicyError, match="JSON object"):
            Detection.from_dict([])

    def test_from_dict_unknown_key(self) -> None:
        with pytest.raises(PolicyError, match="unknown key"):
            Detection.from_dict(detection_payload(colour="red"))

    @pytest.mark.parametrize("key", ["label", "confidence", "box"])
    def test_from_dict_missing_key(self, key: str) -> None:
        payload = detection_payload()
        del payload[key]
        with pytest.raises(PolicyError, match=f"{key} is required"):
            Detection.from_dict(payload)

    @pytest.mark.parametrize("box", ["10,10,50,50", [1, 2, 3], [1, 2, 3, 4, 5]])
    def test_from_dict_bad_box_shape(self, box: object) -> None:
        with pytest.raises(PolicyError, match=r"box must be an array"):
            Detection.from_dict(detection_payload(box=box))

    @pytest.mark.parametrize("coordinate", ["1", float("inf"), None, True])
    def test_from_dict_bad_coordinate(self, coordinate: object) -> None:
        with pytest.raises(PolicyError, match=r"box\[1\]"):
            Detection.from_dict(detection_payload(box=[10.0, coordinate, 50.0, 50.0]))

    def test_from_dict_inverted_box(self) -> None:
        with pytest.raises(GeometryError):
            Detection.from_dict(detection_payload(box=[50.0, 10.0, 10.0, 50.0]))

    @pytest.mark.parametrize("confidence", ["0.8", None, True, 1.2])
    def test_from_dict_bad_confidence(self, confidence: object) -> None:
        with pytest.raises(PolicyError, match="confidence"):
            Detection.from_dict(detection_payload(confidence=confidence))

    def test_from_dict_bad_severity(self) -> None:
        with pytest.raises(PolicyError, match="severity"):
            Detection.from_dict(detection_payload(severity="critical"))

    def test_area_ratio(self) -> None:
        item = detection(box=(0, 0, 500, 250))
        assert item.area_ratio(1000, 500) == pytest.approx(0.25)

    @pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (1.5, 10)])
    def test_area_ratio_invalid_image(self, width: int, height: int) -> None:
        with pytest.raises(PolicyError, match="image_"):
            detection().area_ratio(width, height)


class TestClassRule:
    def test_defaults(self) -> None:
        rule = ClassRule()
        assert (rule.severity, rule.min_confidence) == ("minor", 0.35)

    def test_invalid_severity(self) -> None:
        with pytest.raises(PolicyError, match="severity"):
            ClassRule(severity="critical")

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, "0.5", True, float("nan")])
    def test_invalid_confidence(self, confidence: object) -> None:
        with pytest.raises(PolicyError, match="confidence"):
            ClassRule(min_confidence=confidence)

    def test_ordering(self) -> None:
        with pytest.raises(PolicyError, match="min_confidence <= review_confidence"):
            ClassRule(min_confidence=0.9, review_confidence=0.5)


class TestLocateZone:
    @pytest.mark.parametrize(
        "box,expected",
        [
            (Box(0, 0, 90, 90), "front-left"),
            (Box(100, 100, 200, 200), "middle-center"),
            (Box(210, 210, 290, 290), "rear-right"),
            (Box(0, 100, 90, 200), "middle-left"),
            (Box(210, 0, 290, 90), "front-right"),
            (Box(100, 210, 200, 290), "rear-center"),
        ],
    )
    def test_zones(self, box: Box, expected: str) -> None:
        assert locate_zone(box, 300, 300) == expected

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(PolicyError, match="positive"):
            locate_zone(Box(0, 0, 1, 1), 0, 100)

    def test_invalid_height(self) -> None:
        with pytest.raises(PolicyError, match="positive"):
            locate_zone(Box(0, 0, 1, 1), 100, 0)


class TestPolicyConfig:
    def test_defaults_cover_known_classes(self) -> None:
        assert set(DEFAULT_RULES) == {"scratch", "chip", "dent", "crack"}

    def test_empty_rules(self) -> None:
        with pytest.raises(PolicyError, match="rules"):
            InspectionPolicy(rules={})

    def test_blank_rule_label(self) -> None:
        with pytest.raises(PolicyError, match="rule labels"):
            InspectionPolicy(rules={" ": ClassRule()})

    def test_rule_must_be_class_rule(self) -> None:
        with pytest.raises(PolicyError, match="ClassRule"):
            InspectionPolicy(rules={"scratch": object()})  # type: ignore[dict-item]

    @pytest.mark.parametrize("threshold", [0, -1, "2.5", float("inf"), True])
    def test_invalid_fail_risk_threshold(self, threshold: object) -> None:
        with pytest.raises(PolicyError, match="fail_risk_threshold"):
            InspectionPolicy(fail_risk_threshold=threshold)

    @pytest.mark.parametrize("maximum", [0, -1, 1.5, True])
    def test_invalid_max_findings(self, maximum: object) -> None:
        with pytest.raises(PolicyError, match="max_findings"):
            InspectionPolicy(max_findings=maximum)


class TestInspect:
    def test_no_detections_passes(self) -> None:
        result = InspectionPolicy().inspect([], 100, 100)
        assert result.verdict == "pass"
        assert result.risk_score == 0.0
        assert result.findings == ()
        assert result.reasons == ("No defects above the class confidence floors.",)

    def test_low_confidence_is_ignored(self) -> None:
        result = InspectionPolicy().inspect([detection("scratch", 0.2)], 100, 100)
        assert result.verdict == "pass"
        assert result.ignored[0].label == "scratch"
        assert "confidence floor" in result.ignored[0].reason
        assert any("ignored" in reason for reason in result.reasons)

    def test_minor_below_review_passes(self) -> None:
        result = InspectionPolicy().inspect([detection("scratch", 0.5)], 100, 100)
        assert result.verdict == "pass"
        assert result.findings[0].severity == "minor"
        assert "Minor cosmetic" in result.reasons[0]

    def test_minor_above_review_needs_review(self) -> None:
        result = InspectionPolicy().inspect([detection("scratch", 0.7)], 100, 100)
        assert result.verdict == "needs_review"
        assert "requires human review" in result.reasons[0]

    def test_moderate_needs_review(self) -> None:
        result = InspectionPolicy().inspect([detection("dent", 0.6)], 100, 100)
        assert result.verdict == "needs_review"
        assert result.findings[0].severity == "moderate"

    def test_severe_above_fail_confidence_fails(self) -> None:
        policy = InspectionPolicy(fail_risk_threshold=100.0)
        result = policy.inspect([detection("crack", 0.85)], 100, 100)
        assert result.verdict == "fail"
        assert "Severe crack" in result.reasons[0]

    def test_aggregate_risk_escalates(self) -> None:
        policy = InspectionPolicy(fail_risk_threshold=2.0)
        result = policy.inspect(
            [detection("crack", 0.7), detection("dent", 0.5)], 100, 100
        )
        assert result.verdict == "fail"
        assert result.risk_score == pytest.approx(3.1)
        assert any("Aggregate risk" in reason for reason in result.reasons)

    def test_severe_below_fail_and_risk_needs_review(self) -> None:
        policy = InspectionPolicy(fail_risk_threshold=100.0)
        result = policy.inspect([detection("crack", 0.7)], 100, 100)
        assert result.verdict == "needs_review"
        assert "meets the" not in result.reasons[0]
        assert "requires human review" in result.reasons[0]

    def test_calibration_can_lower_verdict(self) -> None:
        calibrator = TemperatureCalibrator({"scratch": 25.0})
        uncalibrated = InspectionPolicy().inspect([detection("scratch", 0.7)], 100, 100)
        calibrated = InspectionPolicy().inspect(
            [detection("scratch", 0.7)], 100, 100, calibrator=calibrator
        )
        assert uncalibrated.verdict == "needs_review"
        assert calibrated.verdict == "pass"
        assert calibrated.findings[0].raw_confidence == 0.7
        assert calibrated.findings[0].confidence < 0.6

    def test_calibration_can_push_below_floor(self) -> None:
        calibrator = TemperatureCalibrator({"scratch": 100.0})
        policy = InspectionPolicy(rules={"scratch": ClassRule("minor", 0.55, 0.6, 0.95)})
        result = policy.inspect(
            [detection("scratch", 0.7)], 100, 100, calibrator=calibrator
        )
        assert result.findings == ()
        assert result.ignored

    def test_severity_comes_from_rule(self) -> None:
        result = InspectionPolicy().inspect(
            [detection("dent", 0.6, severity="minor")], 100, 100
        )
        assert result.findings[0].severity == "moderate"

    def test_findings_sorted_deterministically(self) -> None:
        items = [
            detection("scratch", 0.8, (40, 40, 50, 50)),
            detection("chip", 0.8, (30, 30, 40, 40)),
            detection("dent", 0.8, (20, 20, 30, 30)),
        ]
        result = InspectionPolicy().inspect(items, 100, 100)
        assert [finding.label for finding in result.findings] == ["chip", "dent", "scratch"]

    def test_sort_breaks_ties_by_box_when_labels_match(self) -> None:
        items = [
            detection("scratch", 0.8, (40, 40, 50, 50)),
            detection("scratch", 0.8, (10, 10, 20, 20)),
        ]
        result = InspectionPolicy().inspect(items, 100, 100)
        assert result.findings[0].box == Box(10, 10, 20, 20)

    def test_unknown_label_rejected(self) -> None:
        with pytest.raises(PolicyError, match="no policy rule"):
            InspectionPolicy().inspect([detection("rust", 0.9)], 100, 100)

    def test_non_detection_rejected(self) -> None:
        with pytest.raises(PolicyError, match="Detection instance"):
            InspectionPolicy().inspect([object()], 100, 100)  # type: ignore[list-item]

    def test_invalid_image_dimensions(self) -> None:
        with pytest.raises(PolicyError, match="positive"):
            InspectionPolicy().inspect([], 0, 100)

    def test_max_findings_guard(self) -> None:
        policy = InspectionPolicy(max_findings=1)
        with pytest.raises(PolicyError, match="max_findings"):
            policy.inspect([detection("scratch", 0.8), detection("dent", 0.8)], 100, 100)

    def test_custom_rules(self) -> None:
        policy = InspectionPolicy(
            rules={"rust": ClassRule("moderate", 0.1, 0.2, 0.3)},
            fail_risk_threshold=10.0,
        )
        result = policy.inspect([detection("rust", 0.25)], 100, 100)
        assert result.verdict == "needs_review"
        assert result.findings[0].zone == "front-left"

    def test_area_ratio_recorded(self) -> None:
        result = InspectionPolicy().inspect([detection("scratch", 0.8, (0, 0, 50, 100))], 100, 100)
        assert result.findings[0].area_ratio == pytest.approx(0.5)

    def test_result_metadata(self) -> None:
        result = InspectionPolicy().inspect([detection("scratch", 0.8)], 640, 480)
        assert (result.image_width, result.image_height) == (640, 480)


class TestFindingSerialisation:
    def test_finding_as_dict(self) -> None:
        finding = Finding(
            label="scratch",
            severity="minor",
            confidence=0.7,
            raw_confidence=0.8,
            zone="front-left",
            box=Box(1, 2, 3, 4),
            area_ratio=0.1,
        )
        payload = finding.as_dict()
        assert payload["box"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
        assert payload["confidence"] == 0.7

    def test_ignored_as_dict(self) -> None:
        ignored = IgnoredDetection("chip", 0.2, "too low")
        assert ignored.as_dict() == {"label": "chip", "confidence": 0.2, "reason": "too low"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def make_result() -> InspectionResult:
    return InspectionPolicy().inspect(
        [
            detection("crack", 0.9, (100, 100, 300, 300)),
            detection("chip", 0.2, (10, 10, 20, 20)),
        ],
        640,
        480,
    )


class TestBuildReport:
    def test_with_manifest(self) -> None:
        report = build_report(
            make_result(),
            manifest=make_manifest(),
            source="unit.json",
            generated_at="2026-01-01T00:00:00+00:00",
            suppressed_count=2,
        )
        assert report["schema_version"] == REPORT_SCHEMA_VERSION
        assert report["model"]["name"] == "vehicle-damage"
        assert report["model"]["calibration"] == {"crack": 1.2}
        assert report["inspection"]["verdict"] == "fail"
        assert report["inspection"]["image"] == {"width": 640, "height": 480}
        assert report["suppressed_count"] == 2
        assert report["findings"][0]["label"] == "crack"

    def test_without_manifest(self) -> None:
        report = build_report(make_result())
        assert report["model"] is None
        assert report["source"] is None
        assert report["generated_at"] is None

    def test_result_type_guard(self) -> None:
        with pytest.raises(ReportError, match="InspectionResult"):
            build_report(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("count", [-1, 1.5, True])
    def test_invalid_suppressed_count(self, count: object) -> None:
        with pytest.raises(ReportError, match="suppressed_count"):
            build_report(make_result(), suppressed_count=count)  # type: ignore[arg-type]

    def test_empty_findings(self) -> None:
        result = InspectionPolicy().inspect([], 100, 100)
        report = build_report(result)
        assert report["findings"] == []


class TestReportJson:
    def test_roundtrip_and_stable_order(self) -> None:
        report = build_report(make_result(), manifest=make_manifest())
        text = report_to_json(report)
        assert text.endswith("\n")
        assert text.index('"findings"') < text.index('"inspection"')
        assert json.loads(text)["inspection"]["verdict"] == "fail"

    def test_custom_indent(self) -> None:
        report = build_report(make_result())
        text = report_to_json(report, indent=4)
        assert '\n    "schema_version"' in text

    def test_non_mapping(self) -> None:
        with pytest.raises(ReportError, match="mapping"):
            report_to_json([])  # type: ignore[arg-type]

    @pytest.mark.parametrize("indent", [-1, 1.5, True])
    def test_invalid_indent(self, indent: object) -> None:
        with pytest.raises(ReportError, match="indent"):
            report_to_json(build_report(make_result()), indent=indent)  # type: ignore[arg-type]


class TestRenderMarkdown:
    def test_full_report(self) -> None:
        report = build_report(
            make_result(),
            manifest=make_manifest(),
            source="unit.json",
            generated_at="2026-01-01T00:00:00+00:00",
            suppressed_count=1,
        )
        markdown = render_markdown(report)
        assert "# Vehicle Inspection Report" in markdown
        assert "**Verdict:** `fail`" in markdown
        assert "vehicle-damage" in markdown
        assert "onnx, pytorch, wasm" in markdown
        assert "| crack | severe |" in markdown
        assert "Ignored detections" in markdown
        assert "1 overlapping or low-score" in markdown
        assert "Aggregate risk" in markdown
        assert "**Source:** unit.json" in markdown
        assert "**Generated:**" in markdown

    def test_minimal_report(self) -> None:
        markdown = render_markdown(build_report(InspectionPolicy().inspect([], 10, 10)))
        assert "_No manifest supplied._" in markdown
        assert "_No findings above the class confidence floors._" in markdown
        assert "Ignored detections" not in markdown
        assert "overlapping" not in markdown
        assert "**Source:**" not in markdown
        assert "**Generated:**" not in markdown

    def test_escapes_pipes_and_newlines(self) -> None:
        result = InspectionPolicy().inspect([], 10, 10)
        report = build_report(result)
        report["findings"] = [
            {
                "label": "scratch|dent",
                "severity": "minor",
                "zone": "front\nleft",
                "confidence": 0.5,
                "raw_confidence": 0.5,
                "area_ratio": 0.1,
            }
        ]
        report["reasons"] = ["line|one\nline two"]
        markdown = render_markdown(report)
        assert "scratch\\|dent" in markdown
        assert "front left" in markdown
        assert "line\\|one line two" in markdown

    def test_non_mapping(self) -> None:
        with pytest.raises(ReportError, match="mapping"):
            render_markdown([])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "key", ["schema_version", "inspection", "model", "findings", "reasons"]
    )
    def test_missing_required_key(self, key: str) -> None:
        report = build_report(make_result())
        del report[key]
        with pytest.raises(ReportError, match=key):
            render_markdown(report)

    def test_error_exports_match(self) -> None:
        assert ReportError is ReportModuleError


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


class TestDemo:
    def test_scenario_is_complete(self) -> None:
        scenario = build_demo()
        assert scenario.manifest.labels == ("scratch", "dent", "crack", "chip")
        assert len(scenario.detections) == 5
        assert scenario.image_width == 1280
        assert scenario.source == "demo"
        assert scenario.manifest.calibration["crack"] == 1.4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestDetectionFile:
    def test_bare_list(self, tmp_path) -> None:
        path = tmp_path / "detections.json"
        write_json(path, [detection_payload()])
        payload = load_detection_file(path)
        assert isinstance(payload, DetectionFile)
        assert payload.image_width is None
        assert len(payload.detections) == 1

    def test_object_with_image(self, tmp_path) -> None:
        path = tmp_path / "detections.json"
        write_json(
            path,
            {
                "image": {"width": 800, "height": 600},
                "detections": [detection_payload()],
            },
        )
        payload = load_detection_file(path)
        assert (payload.image_width, payload.image_height) == (800, 600)

    def test_object_without_image(self, tmp_path) -> None:
        path = tmp_path / "detections.json"
        write_json(path, {"detections": [detection_payload()]})
        payload = load_detection_file(path)
        assert payload.image_width is None
        assert payload.image_height is None
        assert len(payload.detections) == 1

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(PolicyError, match="could not read"):
            load_detection_file(tmp_path / "missing.json")

    def test_invalid_json(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(PolicyError, match="not valid JSON"):
            load_detection_file(path)

    def test_top_level_scalar(self, tmp_path) -> None:
        path = tmp_path / "scalar.json"
        write_json(path, 42)
        with pytest.raises(PolicyError, match="array or object"):
            load_detection_file(path)

    def test_object_unknown_key(self, tmp_path) -> None:
        path = tmp_path / "unknown.json"
        write_json(path, {"detections": [], "camera": "front"})
        with pytest.raises(PolicyError, match="unknown key"):
            load_detection_file(path)

    def test_object_missing_detections(self, tmp_path) -> None:
        path = tmp_path / "missing.json"
        write_json(path, {"image": {"width": 1, "height": 1}})
        with pytest.raises(PolicyError, match="must contain a 'detections' array"):
            load_detection_file(path)

    def test_detections_not_array(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        write_json(path, {"detections": {}})
        with pytest.raises(PolicyError, match="'detections' must be"):
            load_detection_file(path)

    def test_image_not_object(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        write_json(path, {"detections": [], "image": 800})
        with pytest.raises(PolicyError, match="'image' must be"):
            load_detection_file(path)

    def test_image_unknown_key(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        write_json(path, {"detections": [], "image": {"width": 1, "height": 1, "depth": 3}})
        with pytest.raises(PolicyError, match="unknown key"):
            load_detection_file(path)

    def test_image_missing_dimension(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        write_json(path, {"detections": [], "image": {"width": 1}})
        with pytest.raises(PolicyError, match="both width and height"):
            load_detection_file(path)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "800"])
    def test_image_bad_width(self, tmp_path, value: object) -> None:
        path = tmp_path / "bad.json"
        write_json(path, {"detections": [], "image": {"width": value, "height": 1}})
        with pytest.raises(PolicyError, match="image.width"):
            load_detection_file(path)

    def test_malformed_detection_propagates(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        write_json(path, [{"label": "scratch"}])
        with pytest.raises(PolicyError, match="confidence is required"):
            load_detection_file(path)


class TestCli:
    def test_help(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        assert "Offline vehicle-inspection" in capsys.readouterr().out

    def test_version(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_parser_is_reusable(self) -> None:
        assert build_parser().prog == "python -m vehicle_inspection"

    def test_demo_markdown(self, capsys) -> None:
        assert main(["--demo"]) == 0
        output = capsys.readouterr().out
        assert "**Verdict:** `fail`" in output
        assert "Aggregate risk" in output

    def test_demo_json(self, capsys) -> None:
        assert main(["--demo", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["inspection"]["verdict"] == "fail"
        assert report["suppressed_count"] == 2
        assert report["source"] == "demo"

    def test_detection_file_with_image(self, tmp_path, capsys) -> None:
        path = tmp_path / "detections.json"
        write_json(
            path,
            {
                "image": {"width": 800, "height": 600},
                "detections": [
                    {
                        "label": "crack",
                        "confidence": 0.9,
                        "box": [100, 100, 200, 200],
                    }
                ],
            },
        )
        assert main([str(path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["inspection"]["image"] == {"width": 800, "height": 600}
        assert report["model"] is None
        assert report["inspection"]["verdict"] == "fail"

    def test_detection_file_without_image_uses_flags(self, tmp_path, capsys) -> None:
        path = tmp_path / "detections.json"
        write_json(path, [detection_payload(confidence=0.1)])
        assert main([str(path), "--image-width", "320", "--image-height", "240", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["inspection"]["image"] == {"width": 320, "height": 240}

    def test_manifest_is_applied(self, tmp_path, capsys) -> None:
        manifest_path = tmp_path / "manifest.json"
        write_json(manifest_path, manifest_payload())
        detections_path = tmp_path / "detections.json"
        write_json(
            detections_path,
            [
                {
                    "label": "crack",
                    "confidence": 0.82,
                    "box": [10, 10, 110, 110],
                }
            ],
        )
        assert main([str(detections_path), "--manifest", str(manifest_path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["model"]["name"] == "vehicle-damage"
        assert report["findings"][0]["confidence"] == pytest.approx(0.780, abs=1e-3)

    def test_manifest_label_mismatch(self, tmp_path, capsys) -> None:
        manifest_path = tmp_path / "manifest.json"
        write_json(manifest_path, manifest_payload())
        detections_path = tmp_path / "detections.json"
        write_json(detections_path, [detection_payload(label="rust")])
        assert main([str(detections_path), "--manifest", str(manifest_path)]) == 2
        assert "not declared in manifest" in capsys.readouterr().err

    def test_manifest_score_floor_filters_everything(self, tmp_path, capsys) -> None:
        payload = manifest_payload()
        payload["thresholds"]["score"] = 0.95
        payload["calibration"] = {}
        manifest_path = tmp_path / "manifest.json"
        write_json(manifest_path, payload)
        detections_path = tmp_path / "detections.json"
        write_json(detections_path, [detection_payload(confidence=0.9)])
        assert main([str(detections_path), "--manifest", str(manifest_path), "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["findings"] == []
        assert report["suppressed_count"] == 1
        assert report["inspection"]["verdict"] == "pass"

    def test_manifest_without_calibration(self, tmp_path, capsys) -> None:
        payload = manifest_payload()
        payload["calibration"] = {}
        manifest_path = tmp_path / "manifest.json"
        write_json(manifest_path, payload)
        detections_path = tmp_path / "detections.json"
        write_json(detections_path, [detection_payload(confidence=0.8)])
        assert main([str(detections_path), "--manifest", str(manifest_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["findings"][0]["raw_confidence"] == 0.8

    def test_output_file(self, tmp_path, capsys) -> None:
        output = tmp_path / "report.md"
        assert main(["--demo", "--output", str(output)]) == 0
        assert output.read_text(encoding="utf-8").startswith("# Vehicle Inspection Report")
        assert f"Report written to {output}" in capsys.readouterr().out

    def test_output_directory_fails(self, tmp_path, capsys) -> None:
        assert main(["--demo", "--output", str(tmp_path)]) == 2
        assert "could not write report" in capsys.readouterr().err

    def test_both_inputs_rejected(self, tmp_path, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["detections.json", "--demo"])
        assert exc.value.code == 2
        assert "not both" in capsys.readouterr().err

    def test_no_input_rejected(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2
        assert "required unless --demo" in capsys.readouterr().err

    def test_domain_error_returns_two(self, tmp_path, capsys) -> None:
        path = tmp_path / "bad.json"
        write_json(path, [{"label": "scratch"}])
        assert main([str(path)]) == 2
        assert "error:" in capsys.readouterr().err

    def test_missing_manifest_returns_two(self, tmp_path, capsys) -> None:
        detections_path = tmp_path / "detections.json"
        write_json(detections_path, [detection_payload()])
        assert main([str(detections_path), "--manifest", str(tmp_path / "missing.json")]) == 2
        assert "could not read manifest" in capsys.readouterr().err

    def test_invalid_image_flag_returns_two(self, tmp_path, capsys) -> None:
        detections_path = tmp_path / "detections.json"
        write_json(detections_path, [detection_payload()])
        assert main([str(detections_path), "--image-width", "0"]) == 2
        assert "--image-width" in capsys.readouterr().err

    def test_module_entry_point(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["vehicle_inspection", "--demo", "--json"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("vehicle_inspection", run_name="__main__", alter_sys=True)
        assert exc.value.code == 0
        assert json.loads(capsys.readouterr().out)["inspection"]["verdict"] == "fail"

    def test_module_import_does_not_exit(self) -> None:
        module = importlib.import_module("vehicle_inspection.__main__")
        assert module.main is main


def test_public_error_hierarchy() -> None:
    for error in (GeometryError, ManifestError, CalibrationError, PolicyError, ReportError):
        assert issubclass(error, VehicleInspectionError)
