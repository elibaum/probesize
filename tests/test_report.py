import json
from pathlib import Path

from probesize.analyze import AnalysisResult
from probesize.report import write_json_report, write_text_report


def _empty_result(units: str = "nm") -> AnalysisResult:
    # an image where nothing was detected: all statistics are NaN
    return AnalysisResult(
        image_path=Path("nothing_detected.tif"),
        pixel_size_nm=0.1 if units == "nm" else 1.0,
        resolution_mean_nm=float("nan"),
        resolution_std_nm=float("nan"),
        resolution_median_nm=float("nan"),
        resolution_mad_nm=float("nan"),
        n_profiles_analyzed=0,
        n_edge_points_found=0,
        snr_mean=float("nan"),
        asymmetry_mean=float("nan"),
        units=units,
    )


def test_json_report_is_strictly_valid_when_stats_are_nan(tmp_path):
    # regression: NaN statistics used to serialize as bare NaN tokens,
    # which strict JSON parsers (jq, browsers, other languages) reject.
    out = tmp_path / "result.json"

    write_json_report(_empty_result(), out)

    text = out.read_text()
    assert "NaN" not in text

    def reject_constants(name):
        raise ValueError(f"non-strict JSON constant {name}")

    data = json.loads(text, parse_constant=reject_constants)
    assert data["resolution_nm_median"] is None
    assert data["profiles_analyzed"] == 0


def test_pixel_units_reported_in_json_and_text(tmp_path):
    result = _empty_result(units="px")

    json_path = tmp_path / "result.json"
    write_json_report(result, json_path)
    data = json.loads(json_path.read_text())
    assert data["units"] == "px"
    assert data["pixel_size_nm"] is None  # no real calibration to report

    txt_path = tmp_path / "result.txt"
    write_text_report(result, txt_path)
    text = txt_path.read_text()
    assert "uncalibrated" in text
    assert "px (mean +/- std)" in text


def test_nm_units_reported_by_default(tmp_path):
    json_path = tmp_path / "result.json"
    write_json_report(_empty_result(), json_path)
    data = json.loads(json_path.read_text())
    assert data["units"] == "nm"
    assert data["pixel_size_nm"] == 0.1
