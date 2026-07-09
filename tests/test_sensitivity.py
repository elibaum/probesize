import pytest

from probesize.analyze import AnalysisParams
from probesize.gui.sensitivity import apply_sensitivity, describe_sensitivity


def test_lenient_thresholds_are_never_stricter_than_strict():
    strict = apply_sensitivity(AnalysisParams(detection_mode="particles"), 0)
    lenient = apply_sensitivity(AnalysisParams(detection_mode="particles"), 100)

    assert lenient.min_circularity < strict.min_circularity
    assert lenient.min_solidity < strict.min_solidity
    assert lenient.r_squared_min < strict.r_squared_min
    assert lenient.snr_min < strict.snr_min


def test_edge_mode_adjusts_gradient_snr_not_particle_shape_fields():
    base = AnalysisParams(detection_mode="edge")
    strict = apply_sensitivity(base, 0)
    lenient = apply_sensitivity(base, 100)

    assert lenient.min_gradient_snr < strict.min_gradient_snr
    # untouched fields carry over unchanged
    assert lenient.min_circularity == base.min_circularity
    assert lenient.min_solidity == base.min_solidity


def test_apply_sensitivity_does_not_mutate_input():
    base = AnalysisParams(detection_mode="particles")
    original_circularity = base.min_circularity

    apply_sensitivity(base, 100)

    assert base.min_circularity == original_circularity


def test_apply_sensitivity_preserves_unrelated_fields():
    base = AnalysisParams(detection_mode="particles", min_radius_nm=7.5, criterion_lo=0.3)

    result = apply_sensitivity(base, 60)

    assert result.min_radius_nm == 7.5
    assert result.criterion_lo == 0.3
    assert result.detection_mode == "particles"


@pytest.mark.parametrize("pct", [-10, 0, 50, 100, 150])
def test_apply_sensitivity_clamps_out_of_range_values(pct):
    result = apply_sensitivity(AnalysisParams(detection_mode="particles"), pct)
    assert 0.0 <= result.min_circularity <= 1.0
    assert 0.0 <= result.min_solidity <= 1.0


def test_describe_sensitivity_mentions_mode_specific_fields():
    particles = describe_sensitivity(AnalysisParams(detection_mode="particles"))
    edge = describe_sensitivity(AnalysisParams(detection_mode="edge"))

    assert "circularity" in particles
    assert "gradient" in edge
