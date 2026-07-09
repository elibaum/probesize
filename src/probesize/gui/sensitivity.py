"""Maps a single 'sensitivity' slider value onto the handful of shape and
fit-quality thresholds that most often cause a real (imperfect, noisy)
gold-on-carbon or single-edge image to yield too few -- or zero -- detected
profiles: particle circularity/solidity, edge gradient significance, and
the fit acceptance thresholds (R-squared, S/N).

Strict (low) values favor precision: only very clean, unambiguous
particles/edges are kept. Lenient (high) values favor recall: more
marginal candidates are accepted, at some risk of noisier profiles or
substrate artifacts slipping through -- trading cleanliness for coverage
on images where the strict defaults detect nothing.
"""

from __future__ import annotations

import copy

from ..analyze import AnalysisParams

DEFAULT_SENSITIVITY = 50


def _lerp(strict: float, lenient: float, t: float) -> float:
    return strict + (lenient - strict) * t


def apply_sensitivity(base: AnalysisParams, sensitivity_pct: int) -> AnalysisParams:
    """Return a copy of `base` with quality/shape thresholds adjusted
    according to `sensitivity_pct` (0=strict, 100=lenient). Leaves
    everything else (mode, criterion, size bounds, spacing, ...) untouched.
    """
    t = max(0, min(100, sensitivity_pct)) / 100.0
    params = copy.copy(base)
    params.r_squared_min = _lerp(0.90, 0.45, t)
    params.snr_min = _lerp(5.0, 1.2, t)
    if params.detection_mode == "particles":
        params.min_circularity = _lerp(0.85, 0.35, t)
        params.min_solidity = _lerp(0.95, 0.55, t)
    else:
        params.min_gradient_snr = _lerp(5.0, 1.5, t)
    return params


def describe_sensitivity(params: AnalysisParams) -> str:
    """Short human-readable summary of the thresholds a given sensitivity
    setting currently maps to, for display next to the slider."""
    if params.detection_mode == "particles":
        return (
            f"circularity ≥ {params.min_circularity:.2f}, "
            f"solidity ≥ {params.min_solidity:.2f}, "
            f"R² ≥ {params.r_squared_min:.2f}, S/N ≥ {params.snr_min:.1f}"
        )
    return (
        f"gradient S/N ≥ {params.min_gradient_snr:.1f}, "
        f"R² ≥ {params.r_squared_min:.2f}, S/N ≥ {params.snr_min:.1f}"
    )
