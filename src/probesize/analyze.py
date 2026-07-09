"""End-to-end edge-resolution analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .edges import EdgePoint, detect_edge_points
from .fitting import EdgeFitResult, fit_edge_profile, resolution_from_sigma
from .metadata import InstrumentMetadata, read_metadata
from .particles import ParticleEdgePoint, detect_particle_edge_points
from .profile import extract_profile_averaged, profile_in_bounds


@dataclass
class AnalysisParams:
    criterion_lo: float = 0.25
    criterion_hi: float = 0.75
    detection_mode: str = "edge"  # "edge" or "particles"

    # generic Canny-edge detection (detection_mode="edge")
    min_spacing_px: float = 10.0
    canny_sigma: float = 3.0
    min_gradient_snr: float = 3.0

    # round-particle detection (detection_mode="particles")
    min_radius_nm: float = 1.5
    max_radius_nm: float = 100.0
    background_radius_nm: float = 30.0
    min_solidity: float = 0.85
    min_circularity: float = 0.7
    contour_spacing_px: float = 4.0

    profile_length_px: float = 60.0
    tangential_span_px: float = 16.0
    n_tangential_lines: int = 13
    r_squared_min: float = 0.85
    snr_min: float = 3.0
    pixel_size_nm: Optional[float] = None  # overrides embedded metadata if set
    # when neither the image metadata nor pixel_size_nm provides a
    # calibration, fall back to measuring in pixels (result.units == "px")
    # instead of raising; set False to require a real calibration
    allow_pixel_units: bool = True
    # manual calibration applied ONLY to uncalibrated images (used instead
    # of the 1 px fallback); unlike pixel_size_nm it can never override a
    # real embedded calibration
    fallback_pixel_size_nm: Optional[float] = None
    # restrict measurements to a rectangular region of interest, given as
    # (row_min, row_max, col_min, col_max) in image pixels; None = whole
    # image. Applied as a post-hoc acceptance check, so changing the region
    # only needs refilter_result, not a re-analysis.
    region: Optional[tuple[float, float, float, float]] = None


def tangential_params_for(point: EdgePoint, params: AnalysisParams) -> tuple[float, int]:
    """Cap the tangential averaging window for curved (particle) boundaries
    so the locally-straight-line assumption behind the averaging in
    :func:`probesize.profile.extract_profile_averaged` stays valid -- a full
    span comparable to a small particle's own radius would sample points
    where the true boundary has curved away from the perpendicular line,
    which inflates the apparent edge width."""
    if not isinstance(point, ParticleEdgePoint):
        return params.tangential_span_px, params.n_tangential_lines

    span = min(params.tangential_span_px, max(2.0, point.particle_radius_px * 0.6))
    if span < 2.0:
        return span, 1
    n_lines = max(3, min(params.n_tangential_lines, int(span) + 1))
    return span, n_lines


@dataclass
class ProfileResult:
    point: EdgePoint
    fit: EdgeFitResult
    resolution_nm: float
    accepted: bool
    reject_reason: Optional[str] = None


def evaluate_acceptance(point: EdgePoint, fit: EdgeFitResult, params: AnalysisParams) -> tuple[bool, Optional[str]]:
    """Decide whether a fitted profile passes the current quality thresholds.

    Pure post-hoc function of metrics recorded at detection/fit time (shape
    metrics on the point, R-squared and S/N on the fit), so acceptance can
    be re-evaluated for new thresholds without re-detecting or re-fitting --
    see :func:`refilter_result`. NaN metrics (points from a detector that
    doesn't record them) never fail a threshold check.
    """
    if params.region is not None:
        row_min, row_max, col_min, col_max = params.region
        if not (row_min <= point.row <= row_max and col_min <= point.col <= col_max):
            return False, "outside region"

    if isinstance(point, ParticleEdgePoint):
        if point.circularity < params.min_circularity:
            return False, f"circularity={point.circularity:.2f}<{params.min_circularity:g}"
        if point.solidity < params.min_solidity:
            return False, f"solidity={point.solidity:.2f}<{params.min_solidity:g}"
    elif point.gradient_snr is not None and point.gradient_snr < params.min_gradient_snr:
        return False, f"gradient_snr={point.gradient_snr:.1f}<{params.min_gradient_snr:g}"

    if not fit.success:
        return False, fit.failure_reason
    if fit.r_squared < params.r_squared_min:
        return False, f"r_squared={fit.r_squared:.2f}<{params.r_squared_min:g}"
    if fit.snr < params.snr_min:
        return False, f"snr={fit.snr:.1f}<{params.snr_min:g}"
    return True, None


@dataclass
class AnalysisResult:
    """Aggregate output of :func:`analyze_image`.

    ``units`` is ``"nm"`` when a real calibration was available and ``"px"``
    when the analysis fell back to pixel units (uncalibrated image with
    ``AnalysisParams.allow_pixel_units``). In the ``"px"`` case the
    ``*_nm``-suffixed fields hold values in **pixels** (the pipeline ran
    with a pixel size of exactly 1); the suffix is kept for API stability.

    ``calibration`` records where the pixel size came from: ``"metadata"``
    (embedded instrument tag), ``"user"`` (explicit pixel_size_nm override),
    ``"fallback"`` (manual calibration of an uncalibrated image, adjustable
    via :func:`calibrate_result`), or ``"uncalibrated"`` (pixel units).
    """

    image_path: Path
    pixel_size_nm: float
    resolution_mean_nm: float
    resolution_std_nm: float
    resolution_median_nm: float
    resolution_mad_nm: float
    n_profiles_analyzed: int
    n_edge_points_found: int
    snr_mean: float
    asymmetry_mean: float
    units: str = "nm"
    calibration: str = "metadata"
    # region of interest the acceptance was evaluated under, as
    # (row_min, row_max, col_min, col_max); None = whole image
    region: Optional[tuple[float, float, float, float]] = None
    profiles: list[ProfileResult] = field(default_factory=list)
    image_gray: Optional[np.ndarray] = None
    metadata: Optional[InstrumentMetadata] = None


# PIL modes that carry more than 8 bits per sample; converting these with
# convert("L") would truncate to 8-bit (e.g. FEI images are 16-bit "I;16"),
# so read them natively and keep the full dynamic range as float instead.
_HIGH_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def load_gray_image(path: Path | str) -> tuple[np.ndarray, InstrumentMetadata]:
    meta = read_metadata(path)
    with Image.open(path) as img:
        if img.mode in _HIGH_DEPTH_MODES:
            arr = np.asarray(img, dtype=float)
        else:
            arr = np.asarray(img.convert("L"), dtype=float)
    if meta.scan_height_px and meta.scan_width_px:
        arr = arr[: meta.scan_height_px, : meta.scan_width_px]
    return arr, meta


def analyze_image(path: Path | str, params: Optional[AnalysisParams] = None) -> AnalysisResult:
    if params is None:
        params = AnalysisParams()
    path = Path(path)
    image, meta = load_gray_image(path)

    units = "nm"
    if params.pixel_size_nm is not None:
        pixel_size_nm = params.pixel_size_nm
        calibration = "user"
    elif meta.pixel_size_nm is not None:
        pixel_size_nm = meta.pixel_size_nm
        calibration = "metadata"
    elif params.fallback_pixel_size_nm is not None:
        # manual calibration for images that carry none of their own
        pixel_size_nm = params.fallback_pixel_size_nm
        calibration = "fallback"
    elif params.allow_pixel_units:
        # uncalibrated fallback: run the whole pipeline at 1 px per "px",
        # so every reported length is in pixels
        pixel_size_nm = 1.0
        units = "px"
        calibration = "uncalibrated"
    else:
        raise ValueError(
            f"{path}: no pixel size found in image metadata; pass params.pixel_size_nm explicitly"
        )
    if pixel_size_nm <= 0:
        raise ValueError(f"{path}: pixel size must be positive, got {pixel_size_nm} nm/px")

    if params.detection_mode == "particles":
        edge_points: list[EdgePoint] = detect_particle_edge_points(
            image,
            pixel_size_nm=pixel_size_nm,
            min_radius_nm=params.min_radius_nm,
            max_radius_nm=params.max_radius_nm,
            background_radius_nm=params.background_radius_nm,
            min_solidity=params.min_solidity,
            min_circularity=params.min_circularity,
            contour_spacing_px=params.contour_spacing_px,
        )
    elif params.detection_mode == "edge":
        edge_points = detect_edge_points(
            image,
            min_spacing_px=params.min_spacing_px,
            canny_sigma=params.canny_sigma,
            min_gradient_snr=params.min_gradient_snr,
        )
    else:
        raise ValueError(f"unknown detection_mode {params.detection_mode!r}; use 'edge' or 'particles'")

    bounds_margin = params.profile_length_px + params.tangential_span_px

    profile_results: list[ProfileResult] = []
    for point in edge_points:
        if not profile_in_bounds(image.shape, point.row, point.col, point.angle, bounds_margin):
            continue

        span_px, n_lines = tangential_params_for(point, params)
        _, intensity = extract_profile_averaged(
            image,
            point.row,
            point.col,
            point.angle,
            length_px=params.profile_length_px,
            tangential_span_px=span_px,
            n_lines=n_lines,
        )
        distances_nm = (
            np.linspace(-params.profile_length_px / 2, params.profile_length_px / 2, intensity.size)
            * pixel_size_nm
        )

        fit = fit_edge_profile(distances_nm, intensity)
        resolution_nm = (
            resolution_from_sigma(fit.sigma, params.criterion_lo, params.criterion_hi)
            if fit.success
            else float("nan")
        )
        accepted, reject_reason = evaluate_acceptance(point, fit, params)
        profile_results.append(ProfileResult(point, fit, resolution_nm, accepted, reject_reason))

    return _assemble_result(
        path, pixel_size_nm, units, calibration, params.region, len(edge_points), profile_results, image, meta
    )


def refilter_result(result: AnalysisResult, params: AnalysisParams) -> AnalysisResult:
    """Re-apply the acceptance thresholds and edge-width criterion of
    `params` to an existing result without re-detecting or re-fitting.

    This is what makes threshold changes (e.g. the GUI sensitivity slider)
    effectively instant: detection and fitting are done once at the most
    lenient thresholds, and stricter settings only need this pure
    recomputation. Structural parameters (detection mode, radii, spacing,
    profile geometry) are NOT re-applied -- changing those requires a fresh
    :func:`analyze_image`.

    In particles mode the refiltered result is exactly what a direct
    analysis at `params` would produce (shape filtering is per-particle and
    order-independent). In edge mode it is approximately equivalent: the
    spacing-based thinning ran on the lenient candidate set, so the
    surviving representatives can differ slightly from a direct strict run
    even though the resolution statistics agree within their uncertainty.
    """
    profile_results = []
    for profile in result.profiles:
        fit = profile.fit
        resolution_nm = (
            resolution_from_sigma(fit.sigma, params.criterion_lo, params.criterion_hi)
            if fit.success
            else float("nan")
        )
        accepted, reject_reason = evaluate_acceptance(profile.point, fit, params)
        profile_results.append(ProfileResult(profile.point, fit, resolution_nm, accepted, reject_reason))

    return _assemble_result(
        result.image_path,
        result.pixel_size_nm,
        result.units,
        result.calibration,
        params.region,
        result.n_edge_points_found,
        profile_results,
        result.image_gray,
        result.metadata,
    )


def calibrate_result(result: AnalysisResult, pixel_size_nm: float) -> AnalysisResult:
    """Apply (or adjust) a manual pixel-size calibration on a result whose
    lengths did not come from a real embedded calibration.

    Every fitted length is linear in the pixel size, so this is a pure
    rescale of the stored fits -- numerically identical to re-running
    :func:`analyze_image` with ``fallback_pixel_size_nm=pixel_size_nm``,
    but instant. Works both on ``"uncalibrated"`` (px) results and on
    ``"fallback"`` results whose manual value is being corrected (the
    rescale factor is the ratio of new to current pixel size).
    """
    if result.calibration not in ("uncalibrated", "fallback"):
        raise ValueError(
            f"refusing to override a {result.calibration!r} calibration; "
            "manual calibration is only for images without one"
        )
    if pixel_size_nm <= 0:
        raise ValueError(f"pixel size must be positive, got {pixel_size_nm}")

    factor = pixel_size_nm / result.pixel_size_nm
    profile_results = []
    for profile in result.profiles:
        fit = profile.fit
        if fit.success:
            fit = replace(fit, sigma=fit.sigma * factor, x0=fit.x0 * factor)
        resolution = profile.resolution_nm * factor if np.isfinite(profile.resolution_nm) else profile.resolution_nm
        profile_results.append(ProfileResult(profile.point, fit, resolution, profile.accepted, profile.reject_reason))

    return _assemble_result(
        result.image_path,
        pixel_size_nm,
        "nm",
        "fallback",
        result.region,
        result.n_edge_points_found,
        profile_results,
        result.image_gray,
        result.metadata,
    )


def _assemble_result(
    path: Path,
    pixel_size_nm: float,
    units: str,
    calibration: str,
    region: Optional[tuple[float, float, float, float]],
    n_edge_points: int,
    profile_results: list[ProfileResult],
    image: Optional[np.ndarray],
    meta: Optional[InstrumentMetadata],
) -> AnalysisResult:
    accepted = [p for p in profile_results if p.accepted]
    resolutions = np.array([p.resolution_nm for p in accepted])
    snrs = np.array([p.fit.snr for p in accepted])
    asymmetries = np.array([p.fit.asymmetry for p in accepted])

    if resolutions.size:
        median_nm = float(np.median(resolutions))
        mad_nm = float(1.4826 * np.median(np.abs(resolutions - median_nm)))
    else:
        median_nm = mad_nm = float("nan")

    return AnalysisResult(
        image_path=path,
        pixel_size_nm=pixel_size_nm,
        resolution_mean_nm=float(np.mean(resolutions)) if resolutions.size else float("nan"),
        resolution_std_nm=float(np.std(resolutions)) if resolutions.size else float("nan"),
        resolution_median_nm=median_nm,
        resolution_mad_nm=mad_nm,
        n_profiles_analyzed=len(accepted),
        n_edge_points_found=n_edge_points,
        snr_mean=float(np.nanmean(snrs)) if snrs.size else float("nan"),
        asymmetry_mean=float(np.nanmean(asymmetries)) if asymmetries.size else float("nan"),
        units=units,
        calibration=calibration,
        region=region,
        profiles=profile_results,
        image_gray=image,
        metadata=meta,
    )
