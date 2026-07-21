from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image
from scipy.special import erf

from probesize.analyze import AnalysisParams, analyze_image, load_gray_image

EXAMPLES = Path(__file__).resolve().parent.parent / "example_images"
FEI_EXAMPLES = Path(__file__).resolve().parent.parent / "example_images_fei"


def _make_synthetic_edge_image(path: Path, sigma_px: float = 2.0, size: int = 240) -> None:
    rng = np.random.default_rng(7)
    cols = np.arange(size)
    # vertical edge at x=120, blurred by a Gaussian PSF of the given sigma
    profile = 40 + 180 * 0.5 * (1 + erf((cols - 120) / (np.sqrt(2) * sigma_px)))
    image = np.tile(profile, (size, 1))
    image = image + rng.normal(0, 3.0, size=image.shape)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def test_end_to_end_on_synthetic_edge(tmp_path):
    sigma_px = 2.0
    pixel_size_nm = 1.0
    image_path = tmp_path / "synthetic_edge.png"
    _make_synthetic_edge_image(image_path, sigma_px=sigma_px)

    params = AnalysisParams(pixel_size_nm=pixel_size_nm, canny_sigma=2.0, min_spacing_px=10, r_squared_min=0.9)
    result = analyze_image(image_path, params)

    assert result.n_profiles_analyzed > 5
    # default 25/75 criterion width for sigma_px=2nm/px is a fixed, known multiple of sigma
    from probesize.fitting import resolution_from_sigma

    expected = resolution_from_sigma(sigma_px, 0.25, 0.75)
    assert result.resolution_mean_nm == pytest.approx(expected, rel=0.15)


def test_missing_pixel_size_raises_when_calibration_required(tmp_path):
    image_path = tmp_path / "plain.png"
    Image.new("L", (64, 64)).save(image_path)

    with pytest.raises(ValueError):
        analyze_image(image_path, AnalysisParams(allow_pixel_units=False))


def test_uncalibrated_image_falls_back_to_pixel_units(tmp_path):
    # same synthetic edge as the calibrated test, but as a PNG with no
    # pixel size anywhere: the analysis must run with units == "px" and,
    # since the fallback pixel size is exactly 1, produce the same numbers
    # as an explicit pixel_size_nm=1.0 run
    sigma_px = 2.0
    image_path = tmp_path / "uncalibrated_edge.png"
    _make_synthetic_edge_image(image_path, sigma_px=sigma_px)

    fallback = analyze_image(image_path, AnalysisParams(canny_sigma=2.0, min_spacing_px=10, r_squared_min=0.9))
    explicit = analyze_image(
        image_path, AnalysisParams(pixel_size_nm=1.0, canny_sigma=2.0, min_spacing_px=10, r_squared_min=0.9)
    )

    assert fallback.units == "px"
    assert fallback.pixel_size_nm == 1.0
    assert explicit.units == "nm"
    assert fallback.n_profiles_analyzed == explicit.n_profiles_analyzed
    assert fallback.resolution_mean_nm == pytest.approx(explicit.resolution_mean_nm)


def test_bootstrap_ci_properties():
    from probesize.analyze import bootstrap_median_ci

    assert all(np.isnan(v) for v in bootstrap_median_ci(np.array([])))  # empty -> (nan, nan)
    assert bootstrap_median_ci(np.array([3.0])) == (3.0, 3.0)  # single -> zero width

    values = np.arange(1.0, 101.0)  # median 50.5
    lo, hi = bootstrap_median_ci(values)
    assert lo <= np.median(values) <= hi
    assert values.min() <= lo < hi <= values.max()
    # deterministic (fixed seed) so reports are reproducible
    assert bootstrap_median_ci(values) == bootstrap_median_ci(values)


def test_ci_on_result_brackets_median_and_is_tighter_than_mad(tmp_path):
    image_path = tmp_path / "edge.png"
    _make_synthetic_edge_image(image_path)
    result = analyze_image(image_path, AnalysisParams(canny_sigma=2.0))

    assert result.n_profiles_analyzed > 20
    assert result.resolution_ci_low_nm <= result.resolution_median_nm <= result.resolution_ci_high_nm
    # the CI (uncertainty of the estimate) is narrower than the MAD (spread
    # of individual measurements) once many profiles contribute
    assert (result.resolution_ci_high_nm - result.resolution_ci_low_nm) < 2 * result.resolution_mad_nm


def test_manual_calibration_rescales_ci(tmp_path):
    from probesize.analyze import calibrate_result

    image_path = tmp_path / "edge.png"
    _make_synthetic_edge_image(image_path)
    px = analyze_image(image_path, AnalysisParams(canny_sigma=2.0))

    nm = calibrate_result(px, 0.5)

    assert nm.resolution_ci_low_nm == pytest.approx(0.5 * px.resolution_ci_low_nm)
    assert nm.resolution_ci_high_nm == pytest.approx(0.5 * px.resolution_ci_high_nm)


def test_refilter_preserves_pixel_units(tmp_path):
    from probesize.analyze import refilter_result

    image_path = tmp_path / "uncalibrated_edge.png"
    _make_synthetic_edge_image(image_path)
    result = analyze_image(image_path, AnalysisParams(canny_sigma=2.0))
    assert result.units == "px"
    assert result.calibration == "uncalibrated"

    refiltered = refilter_result(result, AnalysisParams(r_squared_min=0.5))

    assert refiltered.units == "px"
    assert refiltered.calibration == "uncalibrated"


def test_manual_calibration_rescale_matches_direct_analysis(tmp_path):
    from probesize.analyze import calibrate_result

    image_path = tmp_path / "uncalibrated_edge.png"
    _make_synthetic_edge_image(image_path)
    px_result = analyze_image(image_path, AnalysisParams(canny_sigma=2.0))

    rescaled = calibrate_result(px_result, 0.75)
    direct = analyze_image(image_path, AnalysisParams(canny_sigma=2.0, fallback_pixel_size_nm=0.75))

    assert rescaled.units == direct.units == "nm"
    assert rescaled.calibration == direct.calibration == "fallback"
    assert rescaled.n_profiles_analyzed == direct.n_profiles_analyzed
    assert rescaled.resolution_median_nm == pytest.approx(direct.resolution_median_nm, abs=1e-9)


def test_manual_calibration_is_adjustable_by_ratio(tmp_path):
    from probesize.analyze import calibrate_result

    image_path = tmp_path / "uncalibrated_edge.png"
    _make_synthetic_edge_image(image_path)
    px_result = analyze_image(image_path, AnalysisParams(canny_sigma=2.0))

    first = calibrate_result(px_result, 0.5)
    corrected = calibrate_result(first, 1.0)  # fix a typo: 0.5 -> 1.0

    assert corrected.pixel_size_nm == 1.0
    assert corrected.resolution_median_nm == pytest.approx(2 * first.resolution_median_nm)
    # 1.0 nm/px is numerically the px values again
    assert corrected.resolution_median_nm == pytest.approx(px_result.resolution_median_nm)


def test_manual_calibration_refuses_calibrated_results():
    from probesize.analyze import calibrate_result

    path = EXAMPLES / "2.tif"
    if not path.exists():
        pytest.skip("sample file not present")
    real = analyze_image(path, AnalysisParams())
    assert real.calibration == "metadata"

    with pytest.raises(ValueError, match="refusing"):
        calibrate_result(real, 0.5)


def test_fallback_pixel_size_never_overrides_embedded_calibration():
    path = EXAMPLES / "2.tif"
    if not path.exists():
        pytest.skip("sample file not present")

    result = analyze_image(path, AnalysisParams(fallback_pixel_size_nm=99.0))

    assert result.calibration == "metadata"
    assert result.pixel_size_nm == pytest.approx(0.3906, rel=1e-3)


def test_zero_pixel_size_raises_rather_than_being_ignored(tmp_path):
    # regression: pixel_size_nm=0.0 used to be treated as "unset" via a
    # truthiness check and silently fell back to embedded metadata.
    image_path = tmp_path / "plain.png"
    Image.new("L", (64, 64)).save(image_path)

    with pytest.raises(ValueError, match="positive"):
        analyze_image(image_path, AnalysisParams(pixel_size_nm=0.0))


def test_default_params_are_not_shared_between_calls(tmp_path):
    # regression: analyze_image had a mutable AnalysisParams() default
    # created once at import time and shared by every call.
    import inspect

    from probesize.analyze import analyze_image as fn

    assert inspect.signature(fn).parameters["params"].default is None


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example_images not present")
def test_runs_on_real_example_image():
    path = EXAMPLES / "2.tif"
    if not path.exists():
        pytest.skip("sample file not present")

    result = analyze_image(path, AnalysisParams())

    assert result.pixel_size_nm > 0
    assert result.n_edge_points_found > 0
    # end-to-end smoke test on real instrument data: just check the pipeline
    # produces a sane, finite, positive resolution rather than a specific
    # value (which depends on detector/threshold choices).
    assert np.isfinite(result.resolution_mean_nm)
    assert result.resolution_mean_nm > 0


def test_load_gray_image_preserves_16bit_and_crops_footer(tmp_path):
    # a 16-bit FEI-style image: full dynamic range must survive loading
    # (not be truncated to 8-bit by convert("L")), and the data bar cropped
    ini = (
        "[Scan]\r\nPixelWidth=2.0e-009\r\nPixelHeight=2.0e-009\r\n\r\n"
        "[Image]\r\nResolutionX=128\r\nResolutionY=100\r\n\r\n"
        "[System]\r\nSystemType=Helios\r\n\r\n"
        "[PrivateFei]\r\nDatabarHeight=20\r\n"
    )
    arr = np.zeros((120, 128), dtype=np.uint16)
    arr[:100, :] = 50000  # scan region well above the 8-bit range
    path = tmp_path / "fei16.tif"
    tifffile.imwrite(path, arr, extratags=[(34682, "s", 0, ini, True)])

    image, meta = load_gray_image(path)

    assert meta.pixel_size_nm == pytest.approx(2.0, rel=1e-6)
    assert image.shape == (100, 128)  # data bar (20 rows) cropped off
    assert image.max() == pytest.approx(50000)  # 16-bit range preserved


@pytest.mark.skipif(not FEI_EXAMPLES.exists(), reason="example_images_fei not present")
def test_analyzes_real_fei_image():
    path = FEI_EXAMPLES / "0_000_000.tif"
    if not path.exists():
        pytest.skip("sample file not present")

    result = analyze_image(path, AnalysisParams(detection_mode="edge"))

    assert result.pixel_size_nm == pytest.approx(0.9615, rel=1e-3)
    assert result.n_profiles_analyzed > 0
    assert np.isfinite(result.resolution_median_nm)
    assert result.resolution_median_nm > 0
