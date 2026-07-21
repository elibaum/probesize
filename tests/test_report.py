import csv
import json
from pathlib import Path

from probesize.analyze import AnalysisResult
from probesize.report import write_csv_summary, write_json_report, write_text_report


def _result(name: str, median: float, ci: tuple[float, float]) -> AnalysisResult:
    return AnalysisResult(
        image_path=Path(name),
        pixel_size_nm=0.5,
        resolution_mean_nm=median,
        resolution_std_nm=0.2,
        resolution_median_nm=median,
        resolution_mad_nm=0.15,
        resolution_ci_low_nm=ci[0],
        resolution_ci_high_nm=ci[1],
        n_profiles_analyzed=100,
        n_edge_points_found=120,
        snr_mean=20.0,
        asymmetry_mean=0.01,
    )


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


def test_ci_in_json_and_text_reports(tmp_path):
    result = _result("img.tif", median=1.5, ci=(1.4, 1.6))

    json_path = tmp_path / "r.json"
    write_json_report(result, json_path)
    data = json.loads(json_path.read_text())
    assert data["resolution_nm_ci95_low"] == 1.4
    assert data["resolution_nm_ci95_high"] == 1.6

    txt_path = tmp_path / "r.txt"
    write_text_report(result, txt_path)
    assert "95% CI = [1.40, 1.60] nm" in txt_path.read_text()


def test_csv_summary_has_one_row_per_result_with_ci(tmp_path):
    results = [
        _result("a.tif", median=1.0, ci=(0.9, 1.1)),
        _result("b.tif", median=2.0, ci=(1.8, 2.2)),
    ]
    out = tmp_path / "summary.csv"

    write_csv_summary(results, out)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["image"] for r in rows] == ["a.tif", "b.tif"]
    assert float(rows[0]["resolution_nm_median"]) == 1.0
    assert float(rows[1]["resolution_nm_ci95_low"]) == 1.8
    assert float(rows[1]["resolution_nm_ci95_high"]) == 2.2
    # image column is the file name only, not a full path
    assert "/" not in rows[0]["image"]
