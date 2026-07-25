"""Single-profile inspector: shows the raw intensity data and erf fit for
one clicked edge/particle point, recomputed on demand from the same
parameters used for the full analysis."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from ..analyze import AnalysisParams, AnalysisResult, tangential_params_for
from ..fitting import edge_spread_function
from ..profile import extract_profile_averaged
from .canvas import MplCanvas


class ProfileDialog(QDialog):
    def __init__(self, result: AnalysisResult, params: AnalysisParams, profile_index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profile inspector")
        self.resize(520, 480)

        profile = result.profiles[profile_index]
        point = profile.point
        span_px, n_lines = tangential_params_for(point, params)
        _, intensity = extract_profile_averaged(
            result.image_gray,
            point.row,
            point.col,
            point.angle,
            length_px=params.profile_length_px,
            tangential_span_px=span_px,
            n_lines=n_lines,
        )
        distances_nm = (
            np.linspace(-params.profile_length_px / 2, params.profile_length_px / 2, intensity.size)
            * result.pixel_size_nm
        )

        units = result.units
        layout = QVBoxLayout(self)
        canvas = MplCanvas(figsize=(5, 4))
        canvas.axes.plot(distances_nm, intensity, ".", color="black", markersize=3, label="data")
        fit = profile.fit
        if fit.success:
            model = edge_spread_function(distances_nm, fit.x0, fit.sigma, fit.lo, fit.hi)
            canvas.axes.plot(distances_nm, model, "-", color="tab:red", label="erf fit")
        canvas.axes.set_xlabel(f"distance ({units})")
        canvas.axes.set_ylabel("intensity")
        canvas.axes.legend()
        layout.addWidget(canvas)

        status = "accepted" if profile.accepted else f"rejected ({profile.reject_reason})"
        lines = [
            f"Location: row={point.row:.1f}, col={point.col:.1f}, angle={np.degrees(point.angle):.1f} deg",
            f"Status: {status}",
        ]
        if fit.success:
            lines += [
                f"Resolution = {profile.resolution_nm:.3f} {units}",
                f"sigma = {fit.sigma:.3f} {units}, R-squared = {fit.r_squared:.3f}",
                f"S/N = {fit.snr:.1f}, asymmetry = {fit.asymmetry:.3f}",
            ]
            limit = result.sampling_limit_px
            if np.isfinite(fit.sigma_px) and fit.sigma_px < limit:
                lines.append(
                    f"⚠ SAMPLING-LIMITED: edge width {fit.sigma_px:.2f} px < {limit:g} px. "
                    "This transition is contained within about a pixel, so the fit describes "
                    "the sampling grid and profile interpolation rather than the instrument. "
                    "Note R-squared and S/N look excellent here precisely because an erf fits "
                    "an interpolation ramp almost perfectly."
                )
        else:
            lines.append(f"Fit failed: {fit.failure_reason}")
        info = QLabel("\n".join(lines))
        info.setWordWrap(True)
        layout.addWidget(info)
