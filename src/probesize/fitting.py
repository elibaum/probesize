"""Edge-spread-function fitting for the edge-resolution criterion.

The classic way to quantify the resolution of a charged-particle microscope
from a real specimen (rather than from a calibration standard) is to find a
sharp material edge in the image, extract an intensity profile
perpendicular to it, and fit the profile with the error function -- the
edge-spread function of a system whose point-spread function is
approximately Gaussian:

    I(x) = lo + (hi - lo) / 2 * (1 + erf((x - x0) / (sqrt(2) * sigma)))

The "resolution" is then reported as the distance between two chosen
percentile crossings of the transition (e.g. 25%/75%, or 20%/80%), which is
a standard, widely published convention in electron/ion microscopy and beam
metrology (sometimes called the knife-edge or edge-width criterion). This module
implements that fit and derived metrics from scratch against the public
formula above -- it is not derived from any vendor's source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv


def edge_spread_function(x: np.ndarray, x0: float, sigma: float, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) * 0.5 * (1.0 + erf((x - x0) / (np.sqrt(2.0) * sigma)))


def _edge_spread_jacobian(x: np.ndarray, x0: float, sigma: float, lo: float, hi: float) -> np.ndarray:
    """Analytical Jacobian of :func:`edge_spread_function` w.r.t. (x0, sigma,
    lo, hi). Passed to ``curve_fit`` so it doesn't fall back to numerical
    (finite-difference) differentiation, which was the dominant cost of
    fitting -- each fit needs ~4x more model evaluations per iteration
    without it."""
    z = (x - x0) / (np.sqrt(2.0) * sigma)
    exp_term = np.exp(-(z**2))
    common = (hi - lo) * exp_term
    d_x0 = -common / (sigma * np.sqrt(2 * np.pi))
    d_sigma = -common * z / (sigma * np.sqrt(np.pi))
    d_lo = 0.5 * (1 - erf(z))
    d_hi = 0.5 * (1 + erf(z))
    return np.column_stack([d_x0, d_sigma, d_lo, d_hi])


@dataclass
class EdgeFitResult:
    success: bool
    x0: float = np.nan
    sigma: float = np.nan
    lo: float = np.nan
    hi: float = np.nan
    r_squared: float = np.nan
    snr: float = np.nan
    asymmetry: float = np.nan
    failure_reason: Optional[str] = None


def _initial_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    lo_guess = np.percentile(y, 5)
    hi_guess = np.percentile(y, 95)
    mid = (lo_guess + hi_guess) / 2.0
    # index closest to the midline crossing, scanned from the profile center
    crossing_idx = np.argmin(np.abs(y - mid))
    x0_guess = x[crossing_idx]
    sigma_guess = max((x.max() - x.min()) / 8.0, 1e-6)
    if hi_guess < lo_guess:
        lo_guess, hi_guess = hi_guess, lo_guess
    return x0_guess, sigma_guess, lo_guess, hi_guess


def fit_edge_profile(x: np.ndarray, y: np.ndarray) -> EdgeFitResult:
    """Fit an error-function edge model to a 1-D intensity profile.

    ``x`` is the sample position along the profile (any consistent unit,
    e.g. pixels or nm) and ``y`` the interpolated intensity at each position.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 6:
        return EdgeFitResult(success=False, failure_reason="profile too short")

    if np.ptp(y) < 1e-9:
        return EdgeFitResult(success=False, failure_reason="flat profile (no edge)")

    x0_g, sigma_g, lo_g, hi_g = _initial_guess(x, y)
    span = x.max() - x.min()

    try:
        popt, _ = curve_fit(
            edge_spread_function,
            x,
            y,
            p0=[x0_g, sigma_g, lo_g, hi_g],
            jac=_edge_spread_jacobian,
            bounds=(
                [x.min() - span, 1e-6, -np.inf, -np.inf],
                [x.max() + span, span, np.inf, np.inf],
            ),
            # A well-behaved 4-parameter fit converges in well under 100
            # evaluations; this just bounds the worst case for pathological
            # (flat/noisy) profiles that would otherwise burn iterations
            # before still failing the r_squared/snr acceptance checks.
            maxfev=300,
        )
    except RuntimeError as exc:
        return EdgeFitResult(success=False, failure_reason=f"fit did not converge ({exc})")

    x0, sigma, lo, hi = popt
    if hi == lo:
        return EdgeFitResult(success=False, failure_reason="degenerate fit (no contrast)")
    if sigma > 0.9 * span:
        # The optimizer pinned sigma against its upper bound -- the profile
        # window doesn't contain a full transition (e.g. a real edge wider
        # than the sampled length, or a monotonic ramp with no plateau), so
        # the fitted width is a boundary artifact, not a measurement.
        return EdgeFitResult(success=False, failure_reason="sigma at fit bound (profile window too short)")

    y_model = edge_spread_function(x, *popt)
    residuals = y - y_model
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    far_mask = np.abs(x - x0) > 2 * sigma
    noise_sample = residuals[far_mask] if np.count_nonzero(far_mask) >= 4 else residuals
    noise_std = float(np.std(noise_sample))
    snr = abs(hi - lo) / noise_std if noise_std > 1e-12 else np.inf

    asymmetry = _empirical_asymmetry(x, y, x0)

    return EdgeFitResult(
        success=True,
        x0=float(x0),
        sigma=float(abs(sigma)),
        lo=float(lo),
        hi=float(hi),
        r_squared=float(r_squared),
        snr=float(snr),
        asymmetry=asymmetry,
    )


def _empirical_asymmetry(x: np.ndarray, y: np.ndarray, x0: float, frac: float = 0.3) -> float:
    """Symmetry check computed from the raw data rather than the (necessarily
    symmetric) erf fit: compares the distance from the fitted center to the
    interpolated `frac` and `1-frac` crossing points on either side.

    Returns 0 for a perfectly symmetric transition; positive/negative values
    indicate the transition is wider on the high-x / low-x side.
    """
    lo, hi = np.percentile(y, 5), np.percentile(y, 95)
    if hi == lo:
        return float("nan")
    level_lo = lo + frac * (hi - lo)
    level_hi = lo + (1 - frac) * (hi - lo)

    order = np.argsort(x)
    xs, ys = x[order], y[order]

    x_lo = _interp_crossing_nearest(xs, ys, level_lo, x0)
    x_hi = _interp_crossing_nearest(xs, ys, level_hi, x0)
    if x_lo is None or x_hi is None:
        return float("nan")

    left_half = x0 - min(x_lo, x_hi)
    right_half = max(x_lo, x_hi) - x0
    denom = left_half + right_half
    if denom <= 0:
        return float("nan")
    return float((right_half - left_half) / denom)


def _interp_crossing_nearest(xs: np.ndarray, ys: np.ndarray, level: float, near: float) -> Optional[float]:
    """Linear-interpolated x closest to `near` where the sorted profile
    crosses `level`. Real (noisy) profiles can cross a given intensity level
    more than once away from the true transition; picking the crossing
    nearest the fitted center is far more robust than taking the first one
    scanning from an arbitrary end of the profile."""
    above = ys >= level
    crossing_idx = np.where(np.diff(above.astype(int)) != 0)[0]
    if crossing_idx.size == 0:
        return None

    candidates = []
    for i in crossing_idx:
        x0_, x1_ = xs[i], xs[i + 1]
        y0_, y1_ = ys[i], ys[i + 1]
        t = 0.5 if y1_ == y0_ else (level - y0_) / (y1_ - y0_)
        candidates.append(x0_ + t * (x1_ - x0_))

    candidates = np.array(candidates)
    return float(candidates[np.argmin(np.abs(candidates - near))])



def resolution_from_sigma(sigma: float, p_lo: float = 0.25, p_hi: float = 0.75) -> float:
    """Edge width between two intensity-fraction crossings of the fitted erf,
    in the same units as the sigma passed in. Default is the 25%/75%
    criterion (a common convention in SEM/FIB edge-width metrology; see
    module docstring)."""
    if not (0 < p_lo < p_hi < 1):
        raise ValueError("require 0 < p_lo < p_hi < 1")
    z_lo = erfinv(2 * p_lo - 1)
    z_hi = erfinv(2 * p_hi - 1)
    return float(np.sqrt(2.0) * sigma * (z_hi - z_lo))
