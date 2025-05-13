"""Exception hierarchy for :mod:`vehicle_inspection`.

Every error raised deliberately by the package derives from
:class:`VehicleInspectionError`, which lets callers (and the CLI) handle
domain failures without catching unrelated programming errors.
"""

from __future__ import annotations


class VehicleInspectionError(Exception):
    """Base class for all errors raised by this package."""


class GeometryError(VehicleInspectionError):
    """Raised for invalid boxes, image sizes, or geometry parameters."""


class ManifestError(VehicleInspectionError):
    """Raised when a model manifest is missing, unreadable, or invalid."""


class CalibrationError(VehicleInspectionError):
    """Raised for invalid probabilities or calibration parameters."""


class PolicyError(VehicleInspectionError):
    """Raised when detections or policy configuration violate the contract."""


class ReportError(VehicleInspectionError):
    """Raised when a report cannot be built or rendered."""
