"""Sampling-limit exclusion.

An edge-spread function can only be measured if the transition is sampled by
several pixels. Below about a pixel the fit describes the sampling grid and
the bilinear interpolation used to extract profiles, not the specimen -- and
crucially R-squared/SN cannot detect it (an erf fits a smooth interpolation
ramp almost perfectly). Such profiles are excluded from the statistics.
"""

import numpy as np
import pytest
from PIL import Image
from scipy.special import erf

from probesize.analyze import (
    AnalysisParams,
    analyze_image,
    calibrate_result,
    refilter_result,
)


def _write_edge(path, sigma_px, size=200):
    """Vertical edge with a known Gaussian blur; sigma_px=0 is a hard step."""
    cols = np.arange(size)
    if sigma_px <= 0:
        profile = np.where(cols >= size // 2, 255.0, 0.0)
    else:
        profile = 127.5 * (1 + erf((cols - size // 2) / (np.sqrt(2) * sigma_px)))
    Image.fromarray(np.tile(profile, (size, 1)).astype(np.uint8)).save(path)


def _write_mixed_edges(path, size=200):
    """One well-sampled edge (sigma 3 px) and one perfectly hard edge, so a
    single image yields both resolvable and unresolvable profiles."""
    cols = np.arange(size)
    profile = 127.5 * (1 + erf((cols - 60) / (np.sqrt(2) * 3.0)))   # resolvable
    profile[140:] = 0.0                                             # hard step
    Image.fromarray(np.tile(profile, (size, 1)).astype(np.uint8)).save(path)


def test_perfect_step_is_excluded_from_statistics(tmp_path):
    path = tmp_path / "step.png"
    _write_edge(path, sigma_px=0)

    result = analyze_image(path, AnalysisParams(canny_sigma=2.0))

    # an unresolvable edge yields no measurement at all, rather than a
    # confident figure derived from the interpolation kernel
    assert result.n_sampling_limited > 0
    assert result.n_profiles_analyzed == 0
    assert result.sampling_limited_dominant
    assert np.isnan(result.resolution_median_nm)


def test_excluded_profiles_do_not_affect_mean_or_median(tmp_path):
    # the point of the change: sub-limit fits must not drag the statistics
    path = tmp_path / "mixed.png"
    _write_mixed_edges(path)

    excluded = analyze_image(path, AnalysisParams(canny_sigma=2.0))
    included = analyze_image(path, AnalysisParams(canny_sigma=2.0, sampling_limit_px=0.0))

    # the image contains both kinds of edge, so both sets are non-empty
    assert excluded.n_sampling_limited > 0
    assert 0 < excluded.n_profiles_analyzed < included.n_profiles_analyzed

    # statistics come from the surviving, resolvable subset only
    surviving = [p.resolution_nm for p in excluded.profiles if p.accepted]
    assert excluded.resolution_median_nm == pytest.approx(float(np.median(surviving)))
    assert excluded.resolution_mean_nm == pytest.approx(float(np.mean(surviving)))
    # and the unresolvable edge was dragging the figure down before
    assert excluded.resolution_median_nm > included.resolution_median_nm


def test_well_sampled_edge_is_not_flagged(tmp_path):
    path = tmp_path / "blurred.png"
    _write_edge(path, sigma_px=3.0)

    result = analyze_image(path, AnalysisParams(canny_sigma=2.0))

    assert result.n_profiles_analyzed > 0
    assert not result.sampling_limited_dominant
    assert result.n_sampling_limited == 0


def test_sigma_px_recorded_and_matches_pixel_size(tmp_path):
    path = tmp_path / "blurred.png"
    _write_edge(path, sigma_px=3.0)

    result = analyze_image(path, AnalysisParams(canny_sigma=2.0, pixel_size_nm=0.5))

    fit = next(p.fit for p in result.profiles if p.accepted)
    assert fit.sigma_px == pytest.approx(fit.sigma / 0.5)
    assert fit.sigma_px == pytest.approx(3.0, rel=0.15)  # recovers the true width


def test_flags_survive_refilter_and_calibration(tmp_path):
    path = tmp_path / "step.png"
    _write_edge(path, sigma_px=0)
    raw = analyze_image(path, AnalysisParams(canny_sigma=2.0))
    assert raw.sampling_limited_dominant

    refiltered = refilter_result(raw, AnalysisParams(canny_sigma=2.0, r_squared_min=0.5))
    assert refiltered.sampling_limited_dominant

    # sigma_px is a property of the pixel grid: invariant under calibration,
    # even though sigma itself is rescaled
    calibrated = calibrate_result(raw, 0.25)
    assert calibrated.sampling_limited_dominant
    first_raw = next(p for p in raw.profiles if p.fit.success)
    first_cal = next(p for p in calibrated.profiles if p.fit.success)
    assert first_cal.fit.sigma_px == pytest.approx(first_raw.fit.sigma_px)
    assert first_cal.fit.sigma == pytest.approx(0.25 * first_raw.fit.sigma)


def test_threshold_is_configurable(tmp_path):
    path = tmp_path / "blurred.png"
    _write_edge(path, sigma_px=3.0)

    lenient = analyze_image(path, AnalysisParams(canny_sigma=2.0, sampling_limit_px=1.0))
    strict = analyze_image(path, AnalysisParams(canny_sigma=2.0, sampling_limit_px=10.0))

    assert not lenient.sampling_limited_dominant
    assert strict.sampling_limited_dominant  # same data, stricter definition
    # a stricter limit excludes more, so fewer profiles survive
    assert strict.n_profiles_analyzed < lenient.n_profiles_analyzed


def test_sensitivity_slider_does_not_change_the_sampling_limit():
    # the limit is physical, not a quality preference: "Lenient" must not
    # reopen the interpolation-artifact regime
    from probesize.gui.sensitivity import apply_sensitivity

    base = AnalysisParams()
    for pct in (0, 50, 100):
        assert apply_sensitivity(base, pct).sampling_limit_px == base.sampling_limit_px


def test_reports_carry_the_flags(tmp_path):
    import csv
    import json

    from probesize.report import write_csv_summary, write_json_report, write_text_report

    path = tmp_path / "step.png"
    _write_edge(path, sigma_px=0)
    result = analyze_image(path, AnalysisParams(canny_sigma=2.0))

    json_path = tmp_path / "r.json"
    write_json_report(result, json_path)
    data = json.loads(json_path.read_text())
    assert data["sampling_limited_dominant"] is True
    assert data["sampling_limited_excluded"] == result.n_sampling_limited

    txt_path = tmp_path / "r.txt"
    write_text_report(result, txt_path)
    assert "sampling-limited" in txt_path.read_text().lower()

    csv_path = tmp_path / "s.csv"
    write_csv_summary([result], csv_path)
    with open(csv_path, newline="") as f:
        row = next(iter(csv.DictReader(f)))
    assert row["sampling_limited_dominant"] == "True"
