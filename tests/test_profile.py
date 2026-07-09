import numpy as np

from probesize.profile import extract_profile, extract_profile_averaged


def _gradient_image(size: int = 64) -> np.ndarray:
    # intensity varies along rows only, so any tangential (row) offset of a
    # horizontal profile line changes the sampled values detectably
    return np.tile(np.arange(size, dtype=float)[:, None], (1, size))


def test_single_line_profile_passes_through_the_edge_point():
    # regression: with n_lines=1, the tangential offsets were computed as
    # np.linspace(-span/2, span/2, 1) == [-span/2], so the "averaged"
    # profile was extracted half a span away from the requested point.
    image = _gradient_image()
    row, col = 32.0, 32.0

    _, single = extract_profile_averaged(
        image, row, col, angle=0.0, length_px=16, tangential_span_px=10.0, n_lines=1
    )
    _, reference = extract_profile(image, row, col, angle=0.0, length_px=16)

    np.testing.assert_allclose(single, reference)


def test_zero_or_negative_n_lines_is_clamped_to_one():
    image = _gradient_image()

    _, profile = extract_profile_averaged(
        image, 32.0, 32.0, angle=0.0, length_px=16, tangential_span_px=10.0, n_lines=0
    )

    assert np.all(np.isfinite(profile))
