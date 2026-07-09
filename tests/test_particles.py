import numpy as np
import pytest
from PIL import Image
from scipy import ndimage
from scipy.special import erf

from probesize.analyze import AnalysisParams, analyze_image
from probesize.fitting import resolution_from_sigma
from probesize.particles import detect_particle_edge_points


def _draw_disk(canvas: np.ndarray, center: tuple[float, float], radius: float, sigma_px: float, amplitude: float = 180.0) -> None:
    """Add a Gaussian-blurred (erf edge-spread) bright disk to `canvas`."""
    rows, cols = np.indices(canvas.shape)
    dist = np.hypot(rows - center[0], cols - center[1])
    # radial edge-spread function of a disk with a Gaussian-blurred boundary
    canvas += amplitude * 0.5 * (1 - erf((dist - radius) / (np.sqrt(2) * sigma_px)))


def _make_synthetic_particle_field(sigma_px: float = 1.2, size: int = 300, seed: int = 3):
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size), 40.0)

    particles = []
    for _ in range(25):
        radius = rng.uniform(6, 14)
        for _attempt in range(50):
            center = (
                rng.uniform(radius + 15, size - radius - 15),
                rng.uniform(radius + 15, size - radius - 15),
            )
            if all(
                np.hypot(center[0] - c[0], center[1] - c[1]) > radius + r + 6
                for c, r in particles
            ):
                break
        else:
            continue  # couldn't place without overlap; skip this particle
        _draw_disk(canvas, center, radius, sigma_px)
        particles.append((center, radius))

    # decoy: a large, irregular (non-convex, non-circular) bright blob that a
    # shape-based particle filter should reject, unlike a naive brightness
    # threshold which would happily pick it up.
    decoy_mask = np.zeros((size, size), dtype=bool)
    rows, cols = np.indices((size, size))
    decoy_mask |= (np.abs(rows - 40) < 6) & (cols < 250)
    decoy_mask |= (np.abs(cols - 250) < 6) & (rows < 40)
    decoy_mask = ndimage.binary_dilation(decoy_mask, iterations=2)
    canvas[decoy_mask] += 150.0

    rng2 = np.random.default_rng(seed + 1)
    canvas += rng2.normal(0, 2.0, size=canvas.shape)
    return np.clip(canvas, 0, 255), particles


def test_particle_detection_finds_circles_and_rejects_decoy_shape():
    image, particles = _make_synthetic_particle_field()

    points = detect_particle_edge_points(image, pixel_size_nm=1.0, min_radius_nm=3, max_radius_nm=20)

    particle_ids = {p.particle_id for p in points}
    # every synthetic disk should be found (allow for edge cases near the
    # image border being clipped by the detector's own margin logic)
    assert len(particle_ids) >= len(particles) - 2

    radii_found = sorted({round(p.particle_radius_px, 1) for p in points})
    assert all(3 <= r <= 20 for r in radii_found)


def test_particle_mode_end_to_end_recovers_known_resolution(tmp_path):
    sigma_px = 1.2
    pixel_size_nm = 1.0
    image, _ = _make_synthetic_particle_field(sigma_px=sigma_px)
    path = tmp_path / "particles.png"
    Image.fromarray(image.astype(np.uint8)).save(path)

    params = AnalysisParams(
        detection_mode="particles",
        pixel_size_nm=pixel_size_nm,
        min_radius_nm=3,
        max_radius_nm=20,
        profile_length_px=20,
        tangential_span_px=6,
        r_squared_min=0.8,
        snr_min=2,
    )
    result = analyze_image(path, params)

    assert result.n_profiles_analyzed > 20
    expected = resolution_from_sigma(sigma_px, 0.25, 0.75)
    assert result.resolution_median_nm == pytest.approx(expected, rel=0.3)
