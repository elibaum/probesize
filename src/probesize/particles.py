"""Detect round particle boundaries (e.g. a gold-on-carbon nanoparticle
resolution test sample) and generate edge-profile sample points around
each accepted particle's perimeter.

Unlike the generic Canny-based detector in :mod:`probesize.edges` -- which
treats every strong gradient in the image as a candidate edge, including
irregular substrate cracks and fold lines that are not meaningful
resolution-test features -- this module specifically segments compact,
round, convex blobs and rejects everything else by shape. It is a
from-scratch, standard image-processing recipe (background flattening via
a difference-of-Gaussians estimate, Otsu thresholding, and region-property
filtering by size/circularity/solidity) and is not derived from any
vendor's blob-detection implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import closing, disk, opening

from .edges import EdgePoint


@dataclass
class ParticleEdgePoint(EdgePoint):
    particle_id: int = -1
    particle_radius_px: float = float("nan")
    # shape metrics of the parent particle, recorded so acceptance can be
    # re-evaluated at stricter thresholds without re-running segmentation
    circularity: float = float("nan")
    solidity: float = float("nan")


def _segment_particles(
    image: np.ndarray,
    background_sigma_px: float,
    smoothing_sigma: float,
) -> np.ndarray:
    """Flatten large-scale background variation (uneven substrate
    brightness, illumination gradients) with a difference-of-Gaussians
    background estimate, then threshold the residual for bright blobs.
    This is much cheaper than grayscale morphology with a large structuring
    element and gives an equivalent result for compact round features."""
    smoothed = ndimage.gaussian_filter(image, sigma=smoothing_sigma)
    background = ndimage.gaussian_filter(image, sigma=background_sigma_px)
    flattened = smoothed - background

    if np.ptp(flattened) < 1e-9:
        return np.zeros(image.shape, dtype=bool)

    try:
        thresh = threshold_otsu(flattened)
    except ValueError:
        return np.zeros(image.shape, dtype=bool)
    thresh = max(thresh, 0.0)

    mask = flattened > thresh
    mask = opening(mask, footprint=disk(2))
    mask = closing(mask, footprint=disk(2))
    mask = ndimage.binary_fill_holes(mask)
    return mask


def _resample_contour(contour: np.ndarray, spacing_px: float) -> np.ndarray:
    """Resample a closed (row, col) contour to ~uniform arc-length spacing."""
    deltas = np.diff(contour, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    arc_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = arc_length[-1]
    if total_length < spacing_px:
        return contour[:1]

    n_points = max(int(total_length / spacing_px), 3)
    sample_positions = np.linspace(0, total_length, n_points, endpoint=False)
    rows = np.interp(sample_positions, arc_length, contour[:, 0])
    cols = np.interp(sample_positions, arc_length, contour[:, 1])
    return np.column_stack([rows, cols])


def _contour_normals(points: np.ndarray, centroid: tuple[float, float]) -> np.ndarray:
    """Outward-pointing normal angle (radians) at each point of a closed,
    ~uniformly spaced contour, via a central-difference tangent estimate."""
    next_pts = np.roll(points, -1, axis=0)
    prev_pts = np.roll(points, 1, axis=0)
    tangent = next_pts - prev_pts
    # perpendicular to the tangent, in (row, col)
    normal_row = tangent[:, 1]
    normal_col = -tangent[:, 0]

    to_point_row = points[:, 0] - centroid[0]
    to_point_col = points[:, 1] - centroid[1]
    dot = normal_row * to_point_row + normal_col * to_point_col
    flip = dot < 0
    normal_row[flip] *= -1
    normal_col[flip] *= -1

    return np.arctan2(normal_row, normal_col)


def detect_particle_edge_points(
    image: np.ndarray,
    pixel_size_nm: float,
    min_radius_nm: float = 1.5,
    max_radius_nm: float = 100.0,
    background_radius_nm: float = 30.0,
    min_solidity: float = 0.85,
    min_circularity: float = 0.7,
    contour_spacing_px: float = 4.0,
    smoothing_sigma: float = 1.5,
) -> list[ParticleEdgePoint]:
    """Segment round particles and return contour sample points suitable for
    :func:`probesize.profile.extract_profile` / ``extract_profile_averaged``.

    ``background_radius_nm`` sets the scale of the background-flattening
    filter (see :func:`_segment_particles`) and should be a few times the
    typical particle radius in the image -- it is independent of
    ``max_radius_nm``, which is only a shape-filter cutoff and can be left
    generous without a performance cost.
    """
    min_radius_px = max(min_radius_nm / pixel_size_nm, 1.0)
    max_radius_px = max(max_radius_nm / pixel_size_nm, min_radius_px + 1.0)

    background_sigma_px = float(np.clip(background_radius_nm / pixel_size_nm, 3, 150))
    mask = _segment_particles(image, background_sigma_px, smoothing_sigma)
    if not mask.any():
        return []

    labeled = label(mask)
    points: list[ParticleEdgePoint] = []

    for region in regionprops(labeled):
        radius_px = np.sqrt(region.area / np.pi)
        if not (min_radius_px <= radius_px <= max_radius_px):
            continue
        if region.solidity < min_solidity:
            continue
        perimeter = region.perimeter if region.perimeter > 0 else np.inf
        circularity = 4 * np.pi * region.area / (perimeter**2)
        if circularity < min_circularity:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        if min_row == 0 or min_col == 0 or max_row >= image.shape[0] or max_col >= image.shape[1]:
            continue  # skip particles touching the image border (incomplete boundary)

        pad = 2
        sub_mask = np.zeros((max_row - min_row + 2 * pad, max_col - min_col + 2 * pad), dtype=float)
        sub_mask[pad:-pad, pad:-pad] = region.image

        contours = find_contours(sub_mask, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)  # outer boundary is the longest contour
        contour = contour + [min_row - pad, min_col - pad]  # back to full-image coordinates

        sampled = _resample_contour(contour, contour_spacing_px)
        if len(sampled) < 3:
            continue
        angles = _contour_normals(sampled, region.centroid)

        for (row, col), angle in zip(sampled, angles):
            points.append(
                ParticleEdgePoint(
                    row=float(row),
                    col=float(col),
                    angle=float(angle),
                    particle_id=region.label,
                    particle_radius_px=float(radius_px),
                    circularity=float(circularity),
                    solidity=float(region.solidity),
                )
            )

    return points
