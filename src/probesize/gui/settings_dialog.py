"""Dialog exposing the full :class:`AnalysisParams` as editable fields."""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from ..analyze import AnalysisParams


def _spinbox(minimum, maximum, value, decimals=3, step=None) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setValue(value)
    if step is not None:
        box.setSingleStep(step)
    return box


class SettingsDialog(QDialog):
    def __init__(self, params: AnalysisParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis settings")
        self._params = copy.copy(params)

        layout = QVBoxLayout(self)

        general = QGroupBox("General")
        form = QFormLayout(general)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["edge", "particles"])
        self.mode_combo.setCurrentText(params.detection_mode)
        self.mode_combo.currentTextChanged.connect(self._update_group_visibility)
        form.addRow("Detection mode:", self.mode_combo)

        self.criterion_lo = _spinbox(0.01, 0.49, params.criterion_lo, decimals=2, step=0.05)
        self.criterion_hi = _spinbox(0.51, 0.99, params.criterion_hi, decimals=2, step=0.05)
        form.addRow("Criterion low fraction:", self.criterion_lo)
        form.addRow("Criterion high fraction:", self.criterion_hi)

        # minimum above zero: a checked override of 0.0 nm/px is never valid
        self.pixel_size_override = _spinbox(0.00001, 1000.0, params.pixel_size_nm or 0.1, decimals=5, step=0.01)
        self.pixel_size_enabled = QCheckBox("Override pixel size (nm/px)")
        self.pixel_size_enabled.setChecked(params.pixel_size_nm is not None)
        form.addRow(self.pixel_size_enabled, self.pixel_size_override)

        self.r_squared_min = _spinbox(0.0, 1.0, params.r_squared_min, decimals=2, step=0.05)
        self.snr_min = _spinbox(0.0, 100.0, params.snr_min, decimals=1, step=0.5)
        form.addRow("Min R-squared:", self.r_squared_min)
        form.addRow("Min S/N ratio:", self.snr_min)
        layout.addWidget(general)

        self.edge_group = QGroupBox("Edge mode")
        edge_form = QFormLayout(self.edge_group)
        self.min_spacing_px = _spinbox(1, 500, params.min_spacing_px, decimals=1, step=1)
        self.canny_sigma = _spinbox(0.1, 20, params.canny_sigma, decimals=1, step=0.5)
        self.min_gradient_snr = _spinbox(0.1, 50, params.min_gradient_snr, decimals=1, step=0.5)
        edge_form.addRow("Min point spacing (px):", self.min_spacing_px)
        edge_form.addRow("Canny sigma:", self.canny_sigma)
        edge_form.addRow("Min gradient S/N:", self.min_gradient_snr)
        layout.addWidget(self.edge_group)

        self.particles_group = QGroupBox("Particles mode")
        particle_form = QFormLayout(self.particles_group)
        self.min_radius_nm = _spinbox(0.1, 10000, params.min_radius_nm, decimals=2, step=0.5)
        self.max_radius_nm = _spinbox(0.1, 10000, params.max_radius_nm, decimals=2, step=1)
        self.background_radius_nm = _spinbox(0.1, 10000, params.background_radius_nm, decimals=2, step=1)
        self.min_solidity = _spinbox(0.0, 1.0, params.min_solidity, decimals=2, step=0.05)
        self.min_circularity = _spinbox(0.0, 1.0, params.min_circularity, decimals=2, step=0.05)
        self.contour_spacing_px = _spinbox(0.5, 100, params.contour_spacing_px, decimals=1, step=0.5)
        particle_form.addRow("Min particle radius (nm):", self.min_radius_nm)
        particle_form.addRow("Max particle radius (nm):", self.max_radius_nm)
        particle_form.addRow("Background flatten radius (nm):", self.background_radius_nm)
        particle_form.addRow("Min solidity:", self.min_solidity)
        particle_form.addRow("Min circularity:", self.min_circularity)
        particle_form.addRow("Contour point spacing (px):", self.contour_spacing_px)
        layout.addWidget(self.particles_group)

        note = QLabel("Profile length / tangential averaging use built-in defaults tuned against real instrument data.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_group_visibility(self.mode_combo.currentText())

    def _update_group_visibility(self, mode: str) -> None:
        self.edge_group.setVisible(mode == "edge")
        self.particles_group.setVisible(mode == "particles")

    def get_params(self) -> AnalysisParams:
        p = self._params
        p.detection_mode = self.mode_combo.currentText()
        p.criterion_lo = self.criterion_lo.value()
        p.criterion_hi = self.criterion_hi.value()
        p.pixel_size_nm = self.pixel_size_override.value() if self.pixel_size_enabled.isChecked() else None
        p.r_squared_min = self.r_squared_min.value()
        p.snr_min = self.snr_min.value()
        p.min_spacing_px = self.min_spacing_px.value()
        p.canny_sigma = self.canny_sigma.value()
        p.min_gradient_snr = self.min_gradient_snr.value()
        p.min_radius_nm = self.min_radius_nm.value()
        p.max_radius_nm = self.max_radius_nm.value()
        p.background_radius_nm = self.background_radius_nm.value()
        p.min_solidity = self.min_solidity.value()
        p.min_circularity = self.min_circularity.value()
        p.contour_spacing_px = self.contour_spacing_px.value()
        return p
