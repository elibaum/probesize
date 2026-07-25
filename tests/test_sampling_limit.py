"""Sampling-limit diagnostics.

An edge-spread function can only be measured if the transition is sampled by
several pixels. Below about a pixel the fit describes the sampling grid and
the bilinear interpolation used to extract profiles, not the specimen -- and
crucially R-squared/SN cannot detect it (an erf fits a smooth interpolation
ramp almost perfectly). These profiles are still reported; they are flagged.
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


def test_perfect_step_is_flagged_but_still_reported(tmp_path):
    path = tmp_path / "step.png"
    _write_edge(path, sigma_px=0)

    result = analyze_image(path, AnalysisParams(canny_sigma=2.0))

    # nothing is excluded -- the measurements are reported, just flagged
    assert result.n_profiles_analyzed > 0
    assert np.isfinite(result.resolution_median_nm)
    assert result.median_sampling_limited
    assert result.n_sampling_limited == result.n_profiles_analyzed


def test_well_sampled_edge_is_not_flagged(tmp_path):
    path = tmp_path / "blurred.png"
    _write_edge(path, sigma_px=3.0)

    result = analyze_image(path, AnalysisParams(canny_sigma=2.0))

    assert result.n_profiles_analyzed > 0
    assert not result.median_sampling_limited
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
    assert raw.median_sampling_limited

    refiltered = refilter_result(raw, AnalysisParams(canny_sigma=2.0, r_squared_min=0.5))
    assert refiltered.median_sampling_limited

    # sigma_px is a property of the pixel grid: invariant under calibration,
    # even though sigma itself is rescaled
    calibrated = calibrate_result(raw, 0.25)
    assert calibrated.median_sampling_limited
    first_raw = next(p for p in raw.profiles if p.accepted)
    first_cal = next(p for p in calibrated.profiles if p.accepted)
    assert first_cal.fit.sigma_px == pytest.approx(first_raw.fit.sigma_px)
    assert first_cal.fit.sigma == pytest.approx(0.25 * first_raw.fit.sigma)


def test_threshold_is_configurable(tmp_path):
    path = tmp_path / "blurred.png"
    _write_edge(path, sigma_px=3.0)

    lenient = analyze_image(path, AnalysisParams(canny_sigma=2.0, sampling_limit_px=1.0))
    strict = analyze_image(path, AnalysisParams(canny_sigma=2.0, sampling_limit_px=10.0))

    assert not lenient.median_sampling_limited
    assert strict.median_sampling_limited  # same data, stricter definition
    # the reported numbers are identical either way -- this is diagnostic only
    assert strict.resolution_median_nm == pytest.approx(lenient.resolution_median_nm)
    assert strict.n_profiles_analyzed == lenient.n_profiles_analyzed


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
    assert data["median_sampling_limited"] is True
    assert data["sampling_limited_profiles"] == result.n_sampling_limited

    txt_path = tmp_path / "r.txt"
    write_text_report(result, txt_path)
    assert "sampling limit" in txt_path.read_text().lower()

    csv_path = tmp_path / "s.csv"
    write_csv_summary([result], csv_path)
    with open(csv_path, newline="") as f:
        row = next(iter(csv.DictReader(f)))
    assert row["median_sampling_limited"] == "True"
