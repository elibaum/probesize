"""Perpendicular intensity-profile extraction at an edge point."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def extract_profile(
    image: np.ndarray,
    row: float,
    col: float,
    angle: float,
    length_px: float = 24.0,
    samples_per_px: float = 4.0,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample image intensity along a line through (row, col) in direction
    `angle` (radians, as returned by :func:`probesize.edges.detect_edge_points`).

    Returns ``(distance_px, intensity)`` where ``distance_px`` is centered at
    0 at the input point and spans ``[-length_px/2, length_px/2]``.

    ``order`` is the spline order passed to ``map_coordinates``. The default
    of 1 (bilinear) is what the measurement pipeline uses. Pass ``order=0``
    for nearest-neighbour sampling, which returns the **actual pixel values**
    the line passes over rather than interpolated ones -- used by the profile
    inspector to show real data next to the interpolated curve.
    """
    n_samples = max(int(length_px * samples_per_px), 8)
    distances = np.linspace(-length_px / 2, length_px / 2, n_samples)

    d_row = np.sin(angle)
    d_col = np.cos(angle)
    sample_rows = row + distances * d_row
    sample_cols = col + distances * d_col

    intensity = map_coordinates(
        np.asarray(image, dtype=float),
        [sample_rows, sample_cols],
        order=order,
        mode="nearest",
    )
    return distances, intensity


def extract_profile_averaged(
    image: np.ndarray,
    row: float,
    col: float,
    angle: float,
    length_px: float = 24.0,
    samples_per_px: float = 4.0,
    tangential_span_px: float = 10.0,
    n_lines: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`extract_profile`, but averages several parallel profile
    lines offset along the edge tangent to reduce per-pixel shot noise
    (the same idea as oversampled/slanted-edge response measurements: a
    locally straight edge segment is sampled repeatedly along its length
    and the perpendicular profiles are stacked and averaged).
    """
    n_samples = max(int(length_px * samples_per_px), 8)
    distances = np.linspace(-length_px / 2, length_px / 2, n_samples)

    d_row, d_col = np.sin(angle), np.cos(angle)
    t_row, t_col = -d_col, d_row  # tangent, perpendicular to the profile direction

    n_lines = max(1, n_lines)
    if n_lines == 1:
        # np.linspace(-a, a, 1) returns [-a], which would put the single
        # line half a span off the edge point instead of through it.
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-tangential_span_px / 2, tangential_span_px / 2, n_lines)
    image_float = np.asarray(image, dtype=float)

    stack = np.empty((n_lines, n_samples))
    for i, off in enumerate(offsets):
        base_row = row + off * t_row
        base_col = col + off * t_col
        sample_rows = base_row + distances * d_row
        sample_cols = base_col + distances * d_col
        stack[i] = map_coordinates(
            image_float,
            [sample_rows, sample_cols],
            order=1,
            mode="nearest",
        )

    return distances, stack.mean(axis=0)


def profile_in_bounds(image_shape: tuple[int, int], row: float, col: float, angle: float, length_px: float) -> bool:
    """Check whether the full profile line stays within the image bounds."""
    half = length_px / 2
    d_row, d_col = np.sin(angle), np.cos(angle)
    r0, r1 = row - half * d_row, row + half * d_row
    c0, c1 = col - half * d_col, col + half * d_col
    h, w = image_shape
    return 0 <= r0 < h and 0 <= r1 < h and 0 <= c0 < w and 0 <= c1 < w
