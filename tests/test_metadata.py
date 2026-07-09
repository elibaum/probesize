from pathlib import Path

import numpy as np
import pytest
import tifffile

from probesize.metadata import read_metadata

EXAMPLES = Path(__file__).resolve().parent.parent / "example_images"
FEI_EXAMPLES = Path(__file__).resolve().parent.parent / "example_images_fei"


def _write_fei_tiff(path: Path, pixel_width_m: float, scan_h: int, file_h: int, width: int = 512) -> None:
    """Write a minimal TIFF carrying an FEI-style INI block in tag 34682."""
    ini = (
        "[User]\r\nDate=01/01/2026\r\n\r\n"
        f"[Scan]\r\nPixelWidth={pixel_width_m}\r\nPixelHeight={pixel_width_m}\r\n\r\n"
        f"[Image]\r\nResolutionX={width}\r\nResolutionY={scan_h}\r\n\r\n"
        "[System]\r\nSystemType=Helios\r\n\r\n"
        f"[PrivateFei]\r\nDatabarHeight={file_h - scan_h}\r\n"
    )
    arr = (np.random.default_rng(0).random((file_h, width)) * 65535).astype(np.uint16)
    tifffile.imwrite(path, arr, extratags=[(34682, "s", 0, ini, True)])


# -- Zeiss (real files) -----------------------------------------------------


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example_images not present")
def test_reads_pixel_size_and_footer_from_real_zeiss_tif():
    path = EXAMPLES / "site_2_10um0.3pA_try1.tif"
    if not path.exists():
        pytest.skip("sample file not present")

    meta = read_metadata(path)

    assert meta.vendor == "Zeiss (ImageTags)"
    assert meta.pixel_size_nm == pytest.approx(0.1953, rel=1e-3)
    assert meta.scan_width_px == 512
    assert meta.scan_height_px == 512
    assert meta.footer_height_px == 48


# -- FEI / Thermo -----------------------------------------------------------


def test_parses_synthetic_fei_ini_tag():
    import tempfile

    path = Path(tempfile.mktemp(suffix=".tif"))
    try:
        _write_fei_tiff(path, pixel_width_m=2.5e-9, scan_h=442, file_h=512)
        meta = read_metadata(path)
    finally:
        path.unlink(missing_ok=True)

    assert meta.vendor is not None and meta.vendor.startswith("FEI/Thermo")
    assert meta.pixel_size_nm == pytest.approx(2.5, rel=1e-6)
    assert meta.scan_height_px == 442
    assert meta.footer_height_px == 512 - 442


@pytest.mark.skipif(not FEI_EXAMPLES.exists(), reason="example_images_fei not present")
def test_reads_real_fei_tif():
    path = FEI_EXAMPLES / "0_000_000.tif"
    if not path.exists():
        pytest.skip("sample file not present")

    meta = read_metadata(path)

    assert meta.vendor is not None and meta.vendor.startswith("FEI/Thermo")
    assert meta.pixel_size_nm == pytest.approx(0.9615, rel=1e-3)
    assert meta.scan_width_px == 2048
    assert meta.scan_height_px == 1768
    assert meta.footer_height_px == 119


# -- generic + fallbacks ----------------------------------------------------


def test_generic_tiff_resolution_used_as_last_resort(tmp_path):
    # a plain calibrated TIFF: 1000 px/cm -> 10 um/px -> 10000 nm/px
    path = tmp_path / "calibrated.tif"
    arr = np.zeros((16, 16), dtype=np.uint8)
    tifffile.imwrite(path, arr, resolution=(1000, 1000), resolutionunit=3)  # unit 3 = centimeter

    meta = read_metadata(path)

    assert meta.vendor == "Generic TIFF"
    assert meta.pixel_size_nm == pytest.approx(10000.0, rel=1e-6)


@pytest.mark.parametrize("dpi", [72, 96, 300])
def test_standard_dpi_defaults_are_not_treated_as_calibration(tmp_path, dpi):
    # regression: image software writes 72/96/300 dpi as an uncalibrated
    # default; treating it as real would silently claim ~0.1-0.35 mm/px
    # and produce resolution numbers off by orders of magnitude.
    path = tmp_path / "dtp_default.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8), resolution=(dpi, dpi), resolutionunit=2)

    meta = read_metadata(path)

    assert meta.pixel_size_nm is None
    assert meta.vendor is None


def test_implausibly_coarse_resolution_rejected(tmp_path):
    # 5 px/cm -> 2 mm per pixel: not a microscope, reject
    path = tmp_path / "coarse.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8), resolution=(5, 5), resolutionunit=3)

    meta = read_metadata(path)

    assert meta.pixel_size_nm is None


def test_missing_tag_returns_none_pixel_size(tmp_path):
    from PIL import Image

    path = tmp_path / "plain.png"
    Image.new("L", (32, 32)).save(path)

    meta = read_metadata(path)

    assert meta.pixel_size_nm is None
    assert meta.vendor is None
    # dimensions are still reported so callers know the image size
    assert meta.scan_width_px == 32
    assert meta.scan_height_px == 32


def test_foreign_non_text_private_tag_degrades_gracefully(tmp_path):
    # regression: another instrument's tag 65000 holding a non-text payload
    # (here a tuple of ints) used to raise from the XML parser instead of
    # being treated as "no usable metadata".
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    path = tmp_path / "foreign.tif"
    ifd = ImageFileDirectory_v2()
    ifd[65000] = (1, 2, 3)
    ifd.tagtype[65000] = 3  # SHORT
    Image.new("L", (32, 32)).save(path, tiffinfo=ifd)

    meta = read_metadata(path)

    assert meta.pixel_size_nm is None


def test_non_xml_text_private_tag_degrades_gracefully(tmp_path):
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    path = tmp_path / "foreign_text.tif"
    ifd = ImageFileDirectory_v2()
    ifd[65000] = "not xml at all"
    ifd.tagtype[65000] = 2  # ASCII
    Image.new("L", (32, 32)).save(path, tiffinfo=ifd)

    meta = read_metadata(path)

    assert meta.pixel_size_nm is None
