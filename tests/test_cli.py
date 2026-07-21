import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import erf

from probesize.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "example_images"


def _write_uncalibrated_edge_png(path: Path, sigma_px: float = 2.0, size: int = 240) -> None:
    rng = np.random.default_rng(7)
    cols = np.arange(size)
    profile = 40 + 180 * 0.5 * (1 + erf((cols - 120) / (np.sqrt(2) * sigma_px)))
    image = np.tile(profile, (size, 1)) + rng.normal(0, 3.0, size=(size, size))
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def test_batch_continues_past_corrupt_file(tmp_path, capsys):
    # regression: one unreadable file used to raise UnidentifiedImageError
    # (an OSError, which _process_one didn't catch) and abort the whole
    # --batch run instead of reporting the file and moving on.
    good = EXAMPLES / "2.tif"
    if good.exists():
        shutil.copy(good, tmp_path / "a_good.tif")
    (tmp_path / "b_corrupt.tif").write_bytes(b"this is not a tiff at all")

    rc = main(["--batch", str(tmp_path), "--no-plots", "-s"])

    captured = capsys.readouterr()
    assert rc == 1  # the corrupt file is reported as a failure...
    assert "b_corrupt.tif" in captured.err
    if good.exists():
        # ...but the good file was still fully processed
        assert "Resolution =" in captured.out
        assert (tmp_path / "results" / "a_good_result.json").exists()


def test_uncalibrated_image_analyzed_in_pixel_units(tmp_path, capsys):
    image = tmp_path / "edge.png"
    _write_uncalibrated_edge_png(image)

    rc = main([str(image), "--no-plots", "-s"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "px (median)" in captured.out
    assert "[uncalibrated: pixel units]" in captured.out


def test_require_calibration_rejects_uncalibrated_image(tmp_path, capsys):
    image = tmp_path / "edge.png"
    _write_uncalibrated_edge_png(image)

    rc = main([str(image), "--no-plots", "--require-calibration"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "no pixel size" in captured.err


def test_fallback_pixel_size_flag_gives_nm_results(tmp_path, capsys):
    image = tmp_path / "edge.png"
    _write_uncalibrated_edge_png(image)

    rc = main([str(image), "--no-plots", "-s", "--fallback-pixel-size-nm", "0.5"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "nm (median)" in captured.out
    assert "uncalibrated" not in captured.out


def test_region_flag_restricts_measurements(tmp_path, capsys):
    # the synthetic edge is vertical at x=120: a region entirely to its
    # left contains no transition, so nothing is measured there, while a
    # region spanning the edge measures normally
    image = tmp_path / "edge.png"
    _write_uncalibrated_edge_png(image)

    rc = main([str(image), "--no-plots", "--region", "80,40,160,200"])
    out_with_edge = capsys.readouterr().out
    assert rc == 0
    assert "profiles analyzed = 0" not in out_with_edge

    rc = main([str(image), "--no-plots", "--region", "0,0,60,200"])
    out_without_edge = capsys.readouterr().out
    assert "profiles analyzed = 0" in out_without_edge


def test_region_flag_validation(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["ignored.png", "--region", "100,100,50,200"])  # X1 < X0
    assert "X1>X0" in capsys.readouterr().err


def test_batch_writes_summary_csv(tmp_path, capsys):
    import csv as _csv

    for name in ("a.png", "b.png"):
        _write_uncalibrated_edge_png(tmp_path / name)
    out = tmp_path / "results"

    rc = main(["--batch", str(tmp_path), "--out", str(out), "--no-plots"])

    assert rc == 0
    summary = out / "summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert sorted(r["image"] for r in rows) == ["a.png", "b.png"]
    assert all(r["resolution_nm_ci95_low"] for r in rows)


def test_single_image_output_includes_ci(tmp_path, capsys):
    image = tmp_path / "edge.png"
    _write_uncalibrated_edge_png(image)

    main([str(image), "--no-plots", "-s"])

    assert "95% CI" in capsys.readouterr().out
