"""Rendering and serialization of analysis results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze import AnalysisResult


def summary_dict(result: AnalysisResult) -> dict:
    # NOTE: when units == "px" (uncalibrated fallback) the *_nm keys hold
    # values in pixels; key names are kept stable for downstream parsers.
    return {
        "image": str(result.image_path),
        "units": result.units,
        "calibration": result.calibration,
        "region_row_col": list(result.region) if result.region is not None else None,
        "pixel_size_nm": result.pixel_size_nm if result.units == "nm" else None,
        "resolution_nm_mean": result.resolution_mean_nm,
        "resolution_nm_std": result.resolution_std_nm,
        "resolution_nm_median": result.resolution_median_nm,
        "resolution_nm_mad": result.resolution_mad_nm,
        "profiles_analyzed": result.n_profiles_analyzed,
        "edge_points_found": result.n_edge_points_found,
        "snr_mean": result.snr_mean,
        "asymmetry_mean": result.asymmetry_mean,
    }


def write_json_report(result: AnalysisResult, out_path: Path | str) -> None:
    # Non-finite floats (an image with zero accepted profiles has NaN
    # statistics) serialize as bare NaN/Infinity tokens, which are not
    # valid JSON -- emit null instead so strict parsers can read the file.
    payload = {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in summary_dict(result).items()
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_text_report(result: AnalysisResult, out_path: Path | str) -> None:
    d = summary_dict(result)
    units = result.units
    pixel_size_line = (
        f"Pixel size = {d['pixel_size_nm']:.4f} nm"
        if units == "nm"
        else "Pixel size = uncalibrated (all measurements in pixels)"
    )
    lines = [
        f"Image: {d['image']}",
        pixel_size_line,
        f"Resolution = {d['resolution_nm_mean']:.2f} +/- {d['resolution_nm_std']:.2f} {units} (mean +/- std)",
        f"Resolution = {d['resolution_nm_median']:.2f} +/- {d['resolution_nm_mad']:.2f} {units} (median +/- robust MAD-based std)",
        f"Profiles analyzed = {d['profiles_analyzed']}",
        f"Edge points found = {d['edge_points_found']}",
        f"S/N ratio (mean) = {d['snr_mean']:.1f}",
        f"Asymmetry (mean) = {d['asymmetry_mean']:.3f}",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def save_annotated_image(result: AnalysisResult, out_path: Path | str) -> None:
    """Save the analyzed image with accepted edge-profile locations marked,
    colored by resolution value."""
    image = result.image_gray
    accepted = [p for p in result.profiles if p.accepted]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image, cmap="gray")

    if accepted:
        rows = [p.point.row for p in accepted]
        cols = [p.point.col for p in accepted]
        values = [p.resolution_nm for p in accepted]
        sc = ax.scatter(cols, rows, c=values, cmap="jet", s=6, linewidths=0)
        fig.colorbar(sc, ax=ax, label=f"resolution ({result.units})", fraction=0.046, pad=0.04)

    if result.region is not None:
        row_min, row_max, col_min, col_max = result.region
        ax.add_patch(
            plt.Rectangle(
                (col_min, row_min), col_max - col_min, row_max - row_min,
                fill=False, edgecolor="tab:blue", linewidth=1.5,
            )
        )

    ax.set_title(
        f"{Path(result.image_path).name}\n"
        f"resolution (median) = {result.resolution_median_nm:.2f} +/- {result.resolution_mad_nm:.2f} {result.units} "
        f"(n={result.n_profiles_analyzed})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_histogram(result: AnalysisResult, out_path: Path | str) -> None:
    accepted = [p.resolution_nm for p in result.profiles if p.accepted]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    if accepted:
        ax.hist(accepted, bins=30, color="tab:green", edgecolor="black")
        ax.axvline(result.resolution_mean_nm, color="tab:red", label="mean")
        ax.axvline(result.resolution_median_nm, color="tab:blue", label="median")
        ax.legend()
    ax.set_xlabel(f"resolution ({result.units})")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_polar_plot(result: AnalysisResult, out_path: Path | str) -> None:
    """Angular distribution of resolution values -- anisotropic effects like
    astigmatism or coma show up as a non-circular distribution."""
    accepted = [p for p in result.profiles if p.accepted]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150, subplot_kw={"projection": "polar"})
    # clockwise from East so the angular orientation matches the image (the
    # edge-normal angle uses a downward-pointing row axis); see PolarDialog
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title(f"resolution ({result.units}) vs edge-normal angle", fontsize=9)
    if accepted:
        angles = [p.point.angle for p in accepted]
        values = [p.resolution_nm for p in accepted]
        ax.scatter(angles, values, s=6, c="tab:blue", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
