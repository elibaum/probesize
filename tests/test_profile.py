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


def test_order_zero_returns_true_pixel_values_order_one_interpolates():
    # a binary image: nearest-neighbour sampling can only ever return values
    # that exist in the image, while bilinear interpolation manufactures
    # intermediate levels between them. This is the distinction the profile
    # inspector now shows explicitly.
    image = np.zeros((40, 40))
    image[:, 20:] = 255.0

    _, raw = extract_profile(image, 20.0, 20.0, angle=0.0, length_px=20, samples_per_px=1.0, order=0)
    _, interpolated = extract_profile(image, 20.0, 20.0, angle=0.0, length_px=20, samples_per_px=4.0)

    assert set(np.unique(raw)) <= {0.0, 255.0}  # only real pixel values
    intermediate = interpolated[(interpolated > 0.5) & (interpolated < 254.5)]
    assert intermediate.size > 0  # interpolation invents in-between levels


def test_zero_or_negative_n_lines_is_clamped_to_one():
    image = _gradient_image()

    _, profile = extract_profile_averaged(
        image, 32.0, 32.0, angle=0.0, length_px=16, tangential_span_px=10.0, n_lines=0
    )

    assert np.all(np.isfinite(profile))
