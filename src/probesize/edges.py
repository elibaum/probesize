"""Locate candidate edge points and their local normal direction.

Rather than assume a specific test-pattern geometry (a single straight
knife edge, a field of round particles, etc.), this detector works
generically: it runs a Canny edge map over the image, estimates the local
gradient direction (the edge normal) at each edge pixel from a Sobel
filter, and then greedily thins the edge-pixel set so that sample points
are spaced at least ``min_spacing_px`` apart. That keeps the number of
profiles proportional to edge *length* rather than pixel count, and works
the same whether the image contains one long diagonal edge or hundreds of
small particle boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import canny


@dataclass
class EdgePoint:
    row: float
    col: float
    angle: float  # radians, direction of steepest intensity change (edge normal)
    # gradient magnitude at this point in units of the estimated pixel-noise
    # floor; lets acceptance be re-evaluated later without re-detecting
    gradient_snr: Optional[float] = None


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Robust estimate of the per-pixel noise level, using the median
    absolute deviation of the image's Laplacian (a standard wavelet/Laplacian
    noise estimator: it responds to uncorrelated pixel noise but is largely
    insensitive to smooth structure or a single sharp edge)."""
    laplacian = ndimage.laplace(image.astype(float))
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    return float(mad / 0.6745 / 6.0)


def detect_edge_points(
    image: np.ndarray,
    min_spacing_px: float = 8.0,
    canny_sigma: float = 1.5,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    min_gradient_snr: float = 5.0,
) -> list[EdgePoint]:
    """Find candidate edge points, reject ones whose gradient magnitude is
    not well above the estimated pixel noise floor (``min_gradient_snr``
    times :func:`estimate_noise_sigma`), and thin the remainder so sample
    points are spaced at least ``min_spacing_px`` apart.
    """
    image = image.astype(float)

    edge_map = canny(
        image,
        sigma=canny_sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    rows, cols = np.nonzero(edge_map)
    if rows.size == 0:
        return []

    smoothed = ndimage.gaussian_filter(image, sigma=canny_sigma)
    gy = ndimage.sobel(smoothed, axis=0)
    gx = ndimage.sobel(smoothed, axis=1)
    magnitude = np.hypot(gy[rows, cols], gx[rows, cols])
    angles = np.arctan2(gy[rows, cols], gx[rows, cols])

    noise_sigma = estimate_noise_sigma(image)
    if noise_sigma > 0:
        snr_values = magnitude / noise_sigma
        strong = snr_values > min_gradient_snr
        rows, cols, angles, snr_values = rows[strong], cols[strong], angles[strong], snr_values[strong]
    else:
        snr_values = np.full(rows.shape, np.inf)
    if rows.size == 0:
        return []

    order = _spaced_subset(rows, cols, min_spacing_px)

    return [
        EdgePoint(
            row=float(rows[i]),
            col=float(cols[i]),
            angle=float(angles[i]),
            gradient_snr=float(snr_values[i]),
        )
        for i in order
    ]


def _spaced_subset(rows: np.ndarray, cols: np.ndarray, min_spacing_px: float) -> list[int]:
    """Greedily pick indices so that no two selected points are closer than
    ``min_spacing_px``, using a KD-tree for neighbor lookups."""
    points = np.column_stack([rows, cols]).astype(float)
    tree = cKDTree(points)
    remaining = set(range(len(points)))
    selected: list[int] = []

    # Deterministic order (row-major) keeps results reproducible.
    for i in range(len(points)):
        if i not in remaining:
            continue
        selected.append(i)
        nearby = tree.query_ball_point(points[i], r=min_spacing_px)
        remaining.difference_update(nearby)

    return selected
