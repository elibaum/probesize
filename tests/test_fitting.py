import numpy as np
import pytest

from probesize.fitting import edge_spread_function, fit_edge_profile, resolution_from_sigma


@pytest.mark.parametrize("true_sigma", [0.5, 1.5, 4.0])
def test_recovers_known_sigma_from_clean_profile(true_sigma):
    x = np.linspace(-20, 20, 300)
    y = edge_spread_function(x, 0.2, true_sigma, 50, 200)

    result = fit_edge_profile(x, y)

    assert result.success
    assert result.sigma == pytest.approx(true_sigma, rel=0.02)
    assert result.r_squared > 0.999


def test_recovers_known_sigma_with_noise():
    rng = np.random.default_rng(42)
    x = np.linspace(-20, 20, 300)
    true_sigma = 2.0
    y = edge_spread_function(x, 0.0, true_sigma, 50, 200) + rng.normal(0, 1.5, size=x.size)

    result = fit_edge_profile(x, y)

    assert result.success
    assert result.sigma == pytest.approx(true_sigma, rel=0.1)
    assert result.snr > 1


def test_flat_profile_is_rejected():
    x = np.linspace(-20, 20, 100)
    y = np.full_like(x, 128.0)

    result = fit_edge_profile(x, y)

    assert not result.success


def test_resolution_from_sigma_matches_known_multiplier():
    # For the 16%/84% criterion the edge width is exactly 2*sigma, since
    # those percentiles correspond to +/- 1 standard deviation of the
    # underlying Gaussian point-spread function.
    sigma = 3.0
    width = resolution_from_sigma(sigma, p_lo=0.1586552539, p_hi=0.8413447461)
    assert width == pytest.approx(2 * sigma, rel=1e-6)


def test_resolution_from_sigma_rejects_bad_criterion():
    with pytest.raises(ValueError):
        resolution_from_sigma(1.0, p_lo=0.8, p_hi=0.2)
