"""Summary-statistics panel shown alongside the image canvas."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel

from ..analyze import AnalysisResult


class ResultsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Results")
        layout = QFormLayout(self)
        self._labels = {
            "resolution_median": QLabel("--"),
            "resolution_ci": QLabel("--"),
            "resolution_mean": QLabel("--"),
            "profiles_analyzed": QLabel("--"),
            "edge_points_found": QLabel("--"),
            "snr_mean": QLabel("--"),
            "asymmetry_mean": QLabel("--"),
            "pixel_size": QLabel("--"),
            "instrument": QLabel("--"),
        }
        # the median is the headline; make it visually so
        median_label = self._labels["resolution_median"]
        median_label.setStyleSheet("font-weight: bold;")
        layout.addRow("Resolution (median +/- MAD):", median_label)
        layout.addRow("95% confidence interval:", self._labels["resolution_ci"])
        layout.addRow("Resolution (mean +/- std):", self._labels["resolution_mean"])
        layout.addRow("Profiles analyzed:", self._labels["profiles_analyzed"])
        layout.addRow("Edge points found:", self._labels["edge_points_found"])
        layout.addRow("S/N ratio (mean):", self._labels["snr_mean"])
        layout.addRow("Asymmetry (mean):", self._labels["asymmetry_mean"])
        layout.addRow("Pixel size:", self._labels["pixel_size"])
        layout.addRow("Instrument:", self._labels["instrument"])

    def clear(self) -> None:
        for label in self._labels.values():
            label.setText("--")

    def update_from_result(self, result: AnalysisResult) -> None:
        units = result.units
        self._labels["resolution_median"].setText(f"{result.resolution_median_nm:.2f} +/- {result.resolution_mad_nm:.2f} {units}")
        self._labels["resolution_ci"].setText(
            f"[{result.resolution_ci_low_nm:.2f}, {result.resolution_ci_high_nm:.2f}] {units}"
        )
        self._labels["resolution_mean"].setText(f"{result.resolution_mean_nm:.2f} +/- {result.resolution_std_nm:.2f} {units}")
        self._labels["profiles_analyzed"].setText(str(result.n_profiles_analyzed))
        self._labels["edge_points_found"].setText(str(result.n_edge_points_found))
        self._labels["snr_mean"].setText(f"{result.snr_mean:.1f}")
        self._labels["asymmetry_mean"].setText(f"{result.asymmetry_mean:.3f}")
        if units == "px":
            self._labels["pixel_size"].setText("uncalibrated — measuring in pixels")
            self._labels["instrument"].setText(
                (result.metadata.vendor if result.metadata else None) or "unknown (no calibration found)"
            )
        else:
            suffix = " (manual)" if result.calibration == "fallback" else ""
            self._labels["pixel_size"].setText(f"{result.pixel_size_nm:.4f} nm/px{suffix}")
            vendor = result.metadata.vendor if result.metadata else None
            self._labels["instrument"].setText(vendor or "unknown (pixel size supplied)")
