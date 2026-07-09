import numpy as np
import pytest
from PIL import Image

from probesize.analyze import AnalysisParams, analyze_image, evaluate_acceptance, refilter_result
from probesize.edges import EdgePoint
from probesize.fitting import EdgeFitResult
from test_particles import _make_synthetic_particle_field


@pytest.fixture(scope="module")
def particle_image(tmp_path_factory):
    image, _ = _make_synthetic_particle_field()
    path = tmp_path_factory.mktemp("refilter") / "particles.png"
    Image.fromarray(image.astype(np.uint8)).save(path)
    return path


def _params(**overrides) -> AnalysisParams:
    base = dict(
        detection_mode="particles",
        pixel_size_nm=1.0,
        min_radius_nm=3,
        max_radius_nm=20,
        profile_length_px=20,
        tangential_span_px=6,
    )
    base.update(overrides)
    return AnalysisParams(**base)


def test_refilter_of_lenient_result_matches_direct_strict_analysis(particle_image):
    lenient = _params(min_circularity=0.35, min_solidity=0.55, r_squared_min=0.45, snr_min=1.2)
    strict = _params(min_circularity=0.75, min_solidity=0.9, r_squared_min=0.85, snr_min=3.0)

    raw = analyze_image(particle_image, lenient)
    direct = analyze_image(particle_image, strict)

    refiltered = refilter_result(raw, strict)

    assert refiltered.n_profiles_analyzed == direct.n_profiles_analyzed
    if direct.n_profiles_analyzed:
        assert refiltered.resolution_median_nm == pytest.approx(direct.resolution_median_nm)
        assert refiltered.resolution_mean_nm == pytest.approx(direct.resolution_mean_nm)


def test_refilter_applies_new_criterion_to_stored_fits(particle_image):
    params = _params()
    raw = analyze_image(particle_image, params)
    assert raw.n_profiles_analyzed > 0

    wider = refilter_result(raw, _params(criterion_lo=0.1, criterion_hi=0.9))

    # same accepted profiles, but every resolution value re-derived for the
    # wider criterion, which is a fixed multiple of the 25/75 width
    assert wider.n_profiles_analyzed == raw.n_profiles_analyzed
    assert wider.resolution_median_nm > raw.resolution_median_nm


def test_refilter_records_shape_reject_reasons(particle_image):
    raw = analyze_image(particle_image, _params(min_circularity=0.35, min_solidity=0.55))

    impossible = refilter_result(raw, _params(min_circularity=0.999, min_solidity=0.999))

    assert impossible.n_profiles_analyzed == 0
    reasons = {p.reject_reason.split("=")[0] for p in impossible.profiles if p.reject_reason}
    assert reasons <= {"circularity", "solidity"}
    assert reasons  # at least one shape rejection recorded


def test_points_without_recorded_metrics_never_fail_shape_checks():
    # e.g. an EdgePoint constructed by external code without gradient_snr
    point = EdgePoint(row=10, col=10, angle=0.0)
    fit = EdgeFitResult(success=True, sigma=1.0, lo=0, hi=100, r_squared=0.99, snr=50.0)

    accepted, reason = evaluate_acceptance(point, fit, AnalysisParams(min_gradient_snr=100.0))

    assert accepted and reason is None


def test_region_restricts_accepted_points_and_is_refilterable(particle_image):
    raw = analyze_image(particle_image, _params())
    assert raw.n_profiles_analyzed > 0
    assert raw.region is None

    half = _params(region=(0, 150, 0, 150))  # top-left quadrant of the 300px field
    restricted = refilter_result(raw, half)

    assert 0 < restricted.n_profiles_analyzed < raw.n_profiles_analyzed
    assert restricted.region == (0, 150, 0, 150)
    for p in restricted.profiles:
        if p.accepted:
            assert 0 <= p.point.row <= 150 and 0 <= p.point.col <= 150
    assert any(p.reject_reason == "outside region" for p in restricted.profiles)

    # removing the region restores the full set
    assert refilter_result(restricted, _params()).n_profiles_analyzed == raw.n_profiles_analyzed


def test_region_matches_direct_analysis(particle_image):
    region = (0, 150, 0, 150)
    raw = analyze_image(particle_image, _params())
    direct = analyze_image(particle_image, _params(region=region))

    refiltered = refilter_result(raw, _params(region=region))

    assert refiltered.n_profiles_analyzed == direct.n_profiles_analyzed
    if direct.n_profiles_analyzed:
        assert refiltered.resolution_median_nm == pytest.approx(direct.resolution_median_nm)


def test_gradient_snr_is_recorded_and_filterable(particle_image):
    lenient = AnalysisParams(detection_mode="edge", pixel_size_nm=1.0, min_gradient_snr=1.5)
    raw = analyze_image(particle_image, lenient)
    recorded = [p.point.gradient_snr for p in raw.profiles]
    assert all(g is not None for g in recorded)

    strict = AnalysisParams(detection_mode="edge", pixel_size_nm=1.0, min_gradient_snr=1e9)
    refiltered = refilter_result(raw, strict)

    assert refiltered.n_profiles_analyzed == 0
    assert any("gradient_snr" in (p.reject_reason or "") for p in refiltered.profiles)
