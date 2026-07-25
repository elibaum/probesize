"""Command-line interface for probesize.

Examples
--------
Single image, printed summary + annotated output next to the input:

    probesize image.tif

Batch a folder, writing reports into ./results:

    probesize --batch ./images --out ./results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import AnalysisParams, analyze_image
from .report import (
    save_annotated_image,
    save_histogram,
    save_polar_plot,
    write_csv_summary,
    write_json_report,
    write_text_report,
)

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="probesize", description="Edge-resolution analysis for charged-particle microscope images")
    p.add_argument("input", nargs="?", help="path to a single image")
    p.add_argument("--batch", metavar="DIR", help="process every image in DIR")
    p.add_argument("--out", metavar="DIR", default=None, help="output directory for reports (default: alongside input)")
    p.add_argument("--pixel-size-nm", type=float, default=None, help="override the pixel size (nm/px) for ALL images, including calibrated ones")
    p.add_argument(
        "--fallback-pixel-size-nm",
        type=float,
        default=None,
        help="pixel size (nm/px) used ONLY for images without an embedded calibration; calibrated images keep their own",
    )
    p.add_argument("--criterion", default="0.25,0.75", help="intensity-fraction pair for the edge-width criterion, e.g. 0.25,0.75 (default) or 0.2,0.8")
    p.add_argument("--mode", choices=["edge", "particles"], default="edge", help="'edge': generic Canny edge detection (single/long edges); 'particles': segment round particles (e.g. gold-on-carbon test samples) and sample their perimeters")
    p.add_argument("--min-spacing-px", type=float, default=10.0, help="[edge mode] minimum spacing between sampled edge points, in pixels")
    p.add_argument("--canny-sigma", type=float, default=3.0, help="[edge mode] Gaussian smoothing used before edge detection; raise for noisy images")
    p.add_argument("--min-gradient-snr", type=float, default=3.0, help="[edge mode] reject candidate edge points whose gradient is not at least this many multiples of the estimated pixel noise floor")
    p.add_argument("--min-radius-nm", type=float, default=1.5, help="[particles mode] reject segmented blobs smaller than this radius")
    p.add_argument("--max-radius-nm", type=float, default=100.0, help="[particles mode] reject segmented blobs larger than this radius")
    p.add_argument("--background-radius-nm", type=float, default=30.0, help="[particles mode] scale of the background-flattening filter; a few times the typical particle radius")
    p.add_argument("--min-solidity", type=float, default=0.85, help="[particles mode] reject blobs less convex than this (area / convex-hull area)")
    p.add_argument("--min-circularity", type=float, default=0.7, help="[particles mode] reject blobs less round than this (4*pi*area/perimeter^2)")
    p.add_argument("--contour-spacing-px", type=float, default=4.0, help="[particles mode] spacing between sampled points around each particle's perimeter")
    p.add_argument(
        "--sampling-limit-px",
        type=float,
        default=1.0,
        help="warn when fitted edge widths fall below this many pixels (diagnostic only; nothing is excluded)",
    )
    p.add_argument("--r-squared-min", type=float, default=0.85, help="minimum fit quality to accept a profile")
    p.add_argument("--snr-min", type=float, default=3.0, help="minimum signal-to-noise ratio to accept a profile")
    p.add_argument(
        "--region",
        default=None,
        metavar="X0,Y0,X1,Y1",
        help="restrict measurements to a rectangle, in pixels (x = column, y = row, origin top-left)",
    )
    p.add_argument("--no-plots", action="store_true", help="skip annotated image / histogram / polar plot outputs")
    p.add_argument(
        "--require-calibration",
        action="store_true",
        help="fail on images without an embedded/supplied pixel size instead of falling back to pixel units",
    )
    p.add_argument("-s", dest="script_mode", action="store_true", help="print only the final resolution line (for scripting)")
    return p


def _iter_images(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _parse_criterion(text: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("--criterion expects 'lo,hi', e.g. 0.25,0.75")
    lo, hi = float(parts[0]), float(parts[1])
    return lo, hi


def _parse_region(text: str) -> tuple[float, float, float, float]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("--region expects 'X0,Y0,X1,Y1' in pixels")
    x0, y0, x1, y1 = (float(p) for p in parts)
    if not (x1 > x0 and y1 > y0):
        raise ValueError("--region needs X1>X0 and Y1>Y0")
    # convert screen-style x/y to the (row_min, row_max, col_min, col_max)
    # convention used internally
    return (y0, y1, x0, x1)


def _process_one(path: Path, out_dir: Path, params: AnalysisParams, make_plots: bool, script_mode: bool):
    """Analyze one image and write its reports. Returns the AnalysisResult,
    or None if the file could not be read (already reported to stderr)."""
    try:
        result = analyze_image(path, params)
    except (ValueError, OSError) as exc:
        # OSError covers unreadable/corrupt image files (PIL raises
        # UnidentifiedImageError, an OSError subclass) -- report and keep
        # going so one bad file doesn't abort a whole --batch run
        print(f"{path}: {exc}", file=sys.stderr)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    write_json_report(result, out_dir / f"{stem}_result.json")
    write_text_report(result, out_dir / f"{stem}_result.txt")
    if make_plots:
        save_annotated_image(result, out_dir / f"{stem}_annotated.jpg")
        save_histogram(result, out_dir / f"{stem}_histogram.png")
        save_polar_plot(result, out_dir / f"{stem}_polar.png")

    units = result.units
    uncalibrated_note = "" if units == "nm" else " [uncalibrated: pixel units]"
    ci = f"[95% CI {result.resolution_ci_low_nm:.2f}-{result.resolution_ci_high_nm:.2f} {units}]"
    if script_mode:
        print(
            f"Resolution = {result.resolution_median_nm:.2f} +/- {result.resolution_mad_nm:.2f} {units} "
            f"(median) {ci}{uncalibrated_note}"
        )
    else:
        print(
            f"{path.name}: resolution = {result.resolution_median_nm:.2f} +/- {result.resolution_mad_nm:.2f} {units} (median) "
            f"{ci} [{result.resolution_mean_nm:.2f} +/- {result.resolution_std_nm:.2f} {units} mean] "
            f"(profiles analyzed = {result.n_profiles_analyzed}, "
            f"S/N = {result.snr_mean:.1f}, asymmetry = {result.asymmetry_mean:.3f})"
            f"{uncalibrated_note}"
        )

    # sampling-limit diagnostics go to stderr so `-s` stdout stays scriptable
    if result.median_sampling_limited:
        print(
            f"{path.name}: WARNING resolution is at or below the pixel sampling limit "
            f"({result.n_sampling_limited}/{result.n_profiles_analyzed} profiles have an edge width "
            f"< {result.sampling_limit_px:g} px). The fit reflects the pixel grid, not the "
            f"instrument -- increase magnification for a valid measurement.",
            file=sys.stderr,
        )
    elif result.n_profiles_analyzed and result.n_sampling_limited / result.n_profiles_analyzed >= 0.2:
        share = result.n_sampling_limited / result.n_profiles_analyzed
        print(
            f"{path.name}: note {share:.0%} of profiles are below the pixel sampling limit "
            f"({result.sampling_limit_px:g} px); the median is still above it.",
            file=sys.stderr,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input and not args.batch:
        build_parser().error("provide an image path or --batch DIR")

    try:
        criterion_lo, criterion_hi = _parse_criterion(args.criterion)
        region = _parse_region(args.region) if args.region else None
    except ValueError as exc:
        build_parser().error(str(exc))

    params = AnalysisParams(
        criterion_lo=criterion_lo,
        criterion_hi=criterion_hi,
        detection_mode=args.mode,
        pixel_size_nm=args.pixel_size_nm,
        min_spacing_px=args.min_spacing_px,
        canny_sigma=args.canny_sigma,
        min_gradient_snr=args.min_gradient_snr,
        min_radius_nm=args.min_radius_nm,
        max_radius_nm=args.max_radius_nm,
        background_radius_nm=args.background_radius_nm,
        min_solidity=args.min_solidity,
        min_circularity=args.min_circularity,
        contour_spacing_px=args.contour_spacing_px,
        r_squared_min=args.r_squared_min,
        snr_min=args.snr_min,
        allow_pixel_units=not args.require_calibration,
        fallback_pixel_size_nm=args.fallback_pixel_size_nm,
        region=region,
        sampling_limit_px=args.sampling_limit_px,
    )

    status = 0
    if args.batch:
        batch_dir = Path(args.batch)
        out_dir = Path(args.out) if args.out else batch_dir / "results"
        results = []
        for image_path in _iter_images(batch_dir):
            result = _process_one(image_path, out_dir, params, not args.no_plots, args.script_mode)
            if result is None:
                status = 1
            else:
                results.append(result)
        if results:
            csv_path = out_dir / "summary.csv"
            write_csv_summary(results, csv_path)
            if not args.script_mode:
                print(f"Wrote summary of {len(results)} image(s) to {csv_path}")
    else:
        input_path = Path(args.input)
        out_dir = Path(args.out) if args.out else input_path.parent
        status = 0 if _process_one(input_path, out_dir, params, not args.no_plots, args.script_mode) else 1

    return status


if __name__ == "__main__":
    raise SystemExit(main())
