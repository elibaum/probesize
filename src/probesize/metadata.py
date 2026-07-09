"""Instrument metadata extraction for charged-particle microscope images.

Different SEM/FIB/HIM vendors record the acquisition pixel size (and the
height of any burnt-in info/data bar) in their own way. This module reads
them behind a small registry of per-vendor parsers: :func:`read_metadata`
tries each parser in turn and returns the first one that recognizes the
file. New vendors are added by writing another parser and appending it to
``_PARSERS``.

Currently recognized:

* **Zeiss ImageTags** -- an XML blob in private TIFF tag 65000 (SmartSEM /
  GeminiSEM / ORION NanoFab HIM), with pixel size in ``ScalingX`` (metres)
  and the raw scan dimensions in ``ImageWidth``/``ImageHeight``.
* **FEI / Thermo Fisher** -- an INI-style text block in tag 34682/34683
  (Helios, Nova, Quanta, Verios); pixel size in ``[Scan] PixelWidth``
  (metres), scan height in ``[Image] ResolutionY`` and the data bar height
  in ``[PrivateFei] DatabarHeight``. Parsed via :mod:`tifffile`.
* **Zeiss CZ_SEM** -- the structured binary tag 34118 written by SmartSEM,
  as a fallback for Zeiss files without the richer ImageTags XML. Parsed
  via :mod:`tifffile`.
* **Generic TIFF resolution** -- ``XResolution``/``ResolutionUnit`` as a
  last resort for any calibrated TIFF.

On-disk images are usually taller than the scan because the instrument
appends an information/data bar; each parser reports ``footer_height_px``
so that band can be cropped before analysis.

This module reads only standard/vendor-documented TIFF structures. It does
not contain, and is not derived from, any vendor's source code.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import tifffile
from PIL import Image

_ZEISS_IMAGETAGS_TAG = 65000


@dataclass
class InstrumentMetadata:
    pixel_size_nm: Optional[float] = None
    scan_width_px: Optional[int] = None
    scan_height_px: Optional[int] = None
    footer_height_px: int = 0
    vendor: Optional[str] = None
    raw_fields: Optional[dict] = None


# -- Zeiss ImageTags (tag 65000, XML) ---------------------------------------


def _parse_zeiss_imagetags_xml(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    fields = {}
    for child in root:
        value_el = child.find("Value")
        if value_el is None or value_el.text is None:
            continue
        fields[child.tag] = value_el.text
    return fields


def _parse_zeiss_imagetags(tif: tifffile.TiffFile, height: int, width: int) -> Optional[InstrumentMetadata]:
    tag = tif.pages[0].tags.get(_ZEISS_IMAGETAGS_TAG)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, str):
        value = value.encode()
    if not isinstance(value, bytes):
        return None  # some other instrument's private tag 65000, non-text payload
    try:
        fields = _parse_zeiss_imagetags_xml(value)
    except ET.ParseError:
        return None
    if "ScalingX" not in fields:
        return None

    try:
        pixel_size_nm = float(fields["ScalingX"]) * 1e9
    except ValueError:
        return None

    scan_width = _to_int(fields.get("ImageWidth")) or width
    scan_height = _to_int(fields.get("ImageHeight"))
    footer_height = max(0, height - scan_height) if scan_height else 0

    return InstrumentMetadata(
        pixel_size_nm=pixel_size_nm,
        scan_width_px=scan_width,
        scan_height_px=scan_height,
        footer_height_px=footer_height,
        vendor="Zeiss (ImageTags)",
        raw_fields=fields,
    )


# -- FEI / Thermo Fisher (tag 34682/34683, INI) -----------------------------


def _parse_fei(tif: tifffile.TiffFile, height: int, width: int) -> Optional[InstrumentMetadata]:
    fei = tif.fei_metadata
    if not fei:
        return None

    # pixel size lives under [Scan] (some columns write it under [EScan])
    pixel_width = None
    for section in ("Scan", "EScan"):
        value = fei.get(section, {}).get("PixelWidth")
        if value:
            pixel_width = value
            break
    if not pixel_width:
        return None
    try:
        pixel_size_nm = float(pixel_width) * 1e9
    except (TypeError, ValueError):
        return None

    image = fei.get("Image", {})
    scan_width = _to_int(image.get("ResolutionX")) or width
    scan_height = _to_int(image.get("ResolutionY"))
    # prefer scan-height math; fall back to the explicit data bar height
    if scan_height:
        footer_height = max(0, height - scan_height)
    else:
        footer_height = _to_int(fei.get("PrivateFei", {}).get("DatabarHeight")) or 0
        scan_height = height - footer_height

    system = fei.get("System", {})
    vendor = "FEI/Thermo"
    if system.get("SystemType"):
        vendor = f"FEI/Thermo ({system['SystemType']})"

    return InstrumentMetadata(
        pixel_size_nm=pixel_size_nm,
        scan_width_px=scan_width,
        scan_height_px=scan_height,
        footer_height_px=footer_height,
        vendor=vendor,
        raw_fields=fei,
    )


# -- Zeiss CZ_SEM (tag 34118, structured) -----------------------------------

# CZ_SEM keys that hold the pixel size, in order of preference; the value is
# a (value, unit) pair where unit is a metric prefix like "nm", "um", "m".
_CZ_SEM_PIXEL_KEYS = ("ap_image_pixel_size", "ap_pixel_size")
_UNIT_TO_NM = {"pm": 1e-3, "nm": 1.0, "µm": 1e3, "um": 1e3, "mm": 1e6, "m": 1e9}


def _parse_zeiss_czsem(tif: tifffile.TiffFile, height: int, width: int) -> Optional[InstrumentMetadata]:
    sem = tif.sem_metadata
    if not sem:
        return None
    for key in _CZ_SEM_PIXEL_KEYS:
        entry = sem.get(key)
        if not entry:
            continue
        pixel_size_nm = _cz_sem_value_to_nm(entry)
        if pixel_size_nm is not None:
            return InstrumentMetadata(
                pixel_size_nm=pixel_size_nm,
                scan_width_px=width,
                scan_height_px=None,
                footer_height_px=0,
                vendor="Zeiss (CZ_SEM)",
                raw_fields=sem,
            )
    return None


def _cz_sem_value_to_nm(entry) -> Optional[float]:
    # entry is typically (label, value, unit) or (value, unit)
    if not isinstance(entry, (tuple, list)):
        return None
    numbers = [e for e in entry if isinstance(e, (int, float))]
    units = [e for e in entry if isinstance(e, str)]
    if not numbers:
        return None
    value = float(numbers[-1])
    factor = _UNIT_TO_NM.get(units[-1].strip(), None) if units else 1e9  # bare metres if unitless
    if factor is None:
        return None
    result = value * factor
    return result if result > 0 else None


# -- Generic calibrated TIFF (XResolution / ResolutionUnit) ------------------

_RESOLUTION_UNIT_TO_NM_PER = {2: 25_400_000.0, 3: 10_000_000.0}  # inch, cm -> nm
# Standard desktop-publishing dpi values that image software writes as an
# uncalibrated default -- a file carrying exactly one of these in inches is
# virtually never a real instrument calibration.
_STANDARD_DPI_DEFAULTS = {72.0, 96.0, 150.0, 300.0, 600.0, 1200.0}
# Reject implied pixel sizes coarser than 0.1 mm/px outright: no microscope
# (or even flatbed scanner of interest here) produces that, so it is almost
# certainly a bogus default rather than a calibration.
_MAX_PLAUSIBLE_PIXEL_NM = 1e5


def _parse_generic_resolution(tif: tifffile.TiffFile, height: int, width: int) -> Optional[InstrumentMetadata]:
    tags = tif.pages[0].tags
    x_res = tags.get("XResolution")
    unit = tags.get("ResolutionUnit")
    if x_res is None or unit is None:
        return None
    value = x_res.value
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        pixels_per_unit = value[0] / value[1]
    else:
        try:
            pixels_per_unit = float(value)
        except (TypeError, ValueError):
            return None
    nm_per_unit = _RESOLUTION_UNIT_TO_NM_PER.get(int(unit.value))
    if not nm_per_unit or pixels_per_unit <= 0:
        return None
    if int(unit.value) == 2 and pixels_per_unit in _STANDARD_DPI_DEFAULTS:
        return None  # a DTP default (e.g. 96 dpi), not an instrument calibration
    pixel_size_nm = nm_per_unit / pixels_per_unit
    if pixel_size_nm > _MAX_PLAUSIBLE_PIXEL_NM:
        return None
    return InstrumentMetadata(
        pixel_size_nm=pixel_size_nm,
        scan_width_px=width,
        scan_height_px=None,
        footer_height_px=0,
        vendor="Generic TIFF",
    )


_PARSERS: tuple[Callable[[tifffile.TiffFile, int, int], Optional[InstrumentMetadata]], ...] = (
    _parse_zeiss_imagetags,
    _parse_fei,
    _parse_zeiss_czsem,
    _parse_generic_resolution,
)


def read_metadata(path: Path | str) -> InstrumentMetadata:
    """Best-effort extraction of pixel size and footer height from a TIFF.

    Tries each vendor parser in :data:`_PARSERS` and returns the first that
    recognizes the file. Returns an :class:`InstrumentMetadata` with
    ``pixel_size_nm`` left as ``None`` when no parser matches -- callers
    should fall back to a user-supplied pixel size in that case.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            height, width = int(page.shape[0]), int(page.shape[1])
            for parser in _PARSERS:
                try:
                    meta = parser(tif, height, width)
                except Exception:  # noqa: BLE001 - a malformed vendor block must not abort the others
                    meta = None
                if meta is not None:
                    return meta
    except (tifffile.TiffFileError, ValueError, OSError):
        # not a TIFF (e.g. PNG/JPG test images) -- fall through to no metadata
        return _read_dimensions_only(path)
    return _read_dimensions_only(path)


def _read_dimensions_only(path: Path | str) -> InstrumentMetadata:
    try:
        with Image.open(path) as img:
            return InstrumentMetadata(scan_width_px=img.width, scan_height_px=img.height)
    except OSError:
        return InstrumentMetadata()


def _to_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
