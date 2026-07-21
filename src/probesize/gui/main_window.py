"""Main application window."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analyze import AnalysisParams, AnalysisResult, calibrate_result, refilter_result
from ..report import (
    save_annotated_image,
    save_histogram,
    save_polar_plot,
    write_csv_summary,
    write_json_report,
    write_text_report,
)
from .about_dialog import AboutDialog
from .canvas import ImageCanvas
from .plot_dialogs import HistogramDialog, PolarDialog
from .profile_dialog import ProfileDialog
from .results_panel import ResultsPanel
from .sensitivity import DEFAULT_SENSITIVITY, apply_sensitivity, describe_sensitivity
from .settings_dialog import SettingsDialog
from .workers import AnalysisWorker, BatchAnalysisWorker, run_in_thread

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Detection and fitting always run at the slider's most-lenient position;
# the raw result is cached and stricter settings are applied as an instant
# post-hoc refilter (see analyze.refilter_result).
_LENIENT_FLOOR = 100

# Edge-width criterion presets offered in the main-screen dropdown.
_CRITERION_PRESETS = (
    ("25–75%", 0.25, 0.75),
    ("20–80%", 0.20, 0.80),
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("probesize")
        self.resize(1100, 700)

        self.params = apply_sensitivity(AnalysisParams(), DEFAULT_SENSITIVITY)
        self.current_result: Optional[AnalysisResult] = None
        # analysis output at the lenient floor; refiltered for display
        self._raw_result: Optional[AnalysisResult] = None
        self.batch_results: dict[str, AnalysisResult] = {}  # raw (lenient-floor) results
        self._current_batch_name: Optional[str] = None
        self._restore_batch_selection: Optional[str] = None
        self._thread = None
        self._worker = None
        self._progress_dialog: Optional[QProgressDialog] = None
        self._histogram_dialog: Optional[HistogramDialog] = None
        self._polar_dialog: Optional[PolarDialog] = None
        self._last_rendered_path: Optional[Path] = None

        self.image_canvas = ImageCanvas()
        self.image_canvas.point_clicked.connect(self._on_point_clicked)
        self.results_panel = ResultsPanel()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["edge", "particles"])
        self.mode_combo.setCurrentText(self.params.detection_mode)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        # edge-width criterion presets: (label, lo, hi). Switching is instant
        # (the resolution is recomputed from each stored fit's sigma).
        self.criterion_combo = QComboBox()
        for label, lo, hi in _CRITERION_PRESETS:
            self.criterion_combo.addItem(label, (lo, hi))
        self._sync_criterion_combo()
        self.criterion_combo.currentIndexChanged.connect(self._on_criterion_changed)

        mode_group = QGroupBox("Detection mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_combo)
        mode_row.addWidget(QLabel("edge = generic edges · particles = round particles"))
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)
        criterion_row = QHBoxLayout()
        criterion_row.addWidget(QLabel("Edge-width criterion:"))
        criterion_row.addWidget(self.criterion_combo)
        criterion_row.addStretch()
        mode_layout.addLayout(criterion_row)

        # shown only when the displayed image has no embedded calibration:
        # lets the user supply a pixel size manually (applied instantly as a
        # rescale of the stored fits -- see analyze.calibrate_result)
        self.calibration_spin = QDoubleSpinBox()
        self.calibration_spin.setRange(0.00001, 100000.0)
        self.calibration_spin.setDecimals(5)
        self.calibration_spin.setValue(1.0)
        self.calibration_spin.setSuffix(" nm/px")
        apply_calibration = QPushButton("Apply")
        apply_calibration.clicked.connect(self._on_manual_calibration_applied)
        self.calibration_group = QGroupBox("Manual calibration (image has none)")
        calibration_layout = QVBoxLayout(self.calibration_group)
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(QLabel("Pixel size:"))
        calibration_row.addWidget(self.calibration_spin)
        calibration_row.addWidget(apply_calibration)
        calibration_row.addStretch()
        calibration_layout.addLayout(calibration_row)
        calibration_note = QLabel(
            "Converts this image's pixel measurements to nm instantly and applies "
            "to future uncalibrated images. Images with their own embedded "
            "calibration are never affected."
        )
        calibration_note.setWordWrap(True)
        calibration_note.setStyleSheet("color: gray; font-size: 10px;")
        calibration_layout.addWidget(calibration_note)
        self.calibration_group.setVisible(False)

        # region of interest: restrict measurements to a user-drawn rectangle
        self.region_edit_checkbox = QCheckBox("Edit region (drag on image)")
        self.region_edit_checkbox.toggled.connect(self._on_region_edit_toggled)
        self.region_clear_button = QPushButton("Clear")
        self.region_clear_button.clicked.connect(self._on_region_cleared)
        self.region_status = QLabel("Whole image")
        self.region_status.setStyleSheet("color: gray; font-size: 10px;")
        region_group = QGroupBox("Region of interest")
        region_layout = QVBoxLayout(region_group)
        region_row = QHBoxLayout()
        region_row.addWidget(self.region_edit_checkbox)
        region_row.addWidget(self.region_clear_button)
        region_row.addStretch()
        region_layout.addLayout(region_row)
        region_layout.addWidget(self.region_status)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(DEFAULT_SENSITIVITY)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.sensitivity_slider.sliderReleased.connect(self._on_sensitivity_released)
        self.sensitivity_description = QLabel(describe_sensitivity(self.params))
        self.sensitivity_description.setWordWrap(True)
        self.sensitivity_description.setStyleSheet("color: gray; font-size: 10px;")

        sensitivity_group = QGroupBox("Detection sensitivity")
        sensitivity_layout = QVBoxLayout(sensitivity_group)
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Strict"))
        slider_row.addWidget(self.sensitivity_slider)
        slider_row.addWidget(QLabel("Lenient"))
        sensitivity_layout.addLayout(slider_row)
        sensitivity_layout.addWidget(self.sensitivity_description)
        sensitivity_note = QLabel(
            "Raise this if a real (noisy/imperfect) image detects too few or zero "
            "particles/edges at the default (strict) settings."
        )
        sensitivity_note.setWordWrap(True)
        sensitivity_note.setStyleSheet("color: gray; font-size: 10px;")
        sensitivity_layout.addWidget(sensitivity_note)

        self.show_rejected_checkbox = QCheckBox("Show rejected points (grey; click one to see why)")
        self.show_rejected_checkbox.toggled.connect(lambda _checked: self._render_current_result())
        sensitivity_layout.addWidget(self.show_rejected_checkbox)

        self.batch_table = QTableWidget(0, 3)
        self.batch_table.setHorizontalHeaderLabels(["Image", "Resolution (median)", "Profiles"])
        self.batch_table.itemSelectionChanged.connect(self._on_batch_row_selected)
        self.batch_table.setMinimumWidth(280)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.resizeColumnsToContents()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(mode_group)
        right_layout.addWidget(self.calibration_group)
        right_layout.addWidget(region_group)
        right_layout.addWidget(sensitivity_group)
        right_layout.addWidget(self.results_panel)
        right_layout.addWidget(QLabel("Batch results (select a row to view):"))
        right_layout.addWidget(self.batch_table)

        self.image_toolbar = NavigationToolbar2QT(self.image_canvas, self)
        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.image_toolbar)
        canvas_layout.addWidget(self.image_canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(canvas_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._build_menus()
        self.statusBar().showMessage("Open an image to begin.")

    # -- menu construction ------------------------------------------------

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Open Image...", self.open_image)
        file_menu.addAction("Open Batch Folder...", self.open_batch_folder)
        file_menu.addSeparator()
        file_menu.addAction("Save Results...", self.save_results)
        file_menu.addAction("Export Batch Summary (CSV)...", self.export_batch_csv)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction("Settings...", self.open_settings)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction("Histogram", self.show_histogram)
        tools_menu.addAction("Polar Plot", self.show_polar_plot)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("About", lambda: AboutDialog(self).exec())

    # -- settings -----------------------------------------------------------

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.params, self)
        if dialog.exec():
            new_params = dialog.get_params()
            if new_params == self.params:
                return  # nothing changed; keep all cached results
            self.params = new_params
            self.sensitivity_description.setText(describe_sensitivity(self.params))
            # keep the main-screen selectors in sync; _on_mode_changed
            # no-ops because params.detection_mode is already updated
            self.mode_combo.setCurrentText(self.params.detection_mode)
            self._sync_criterion_combo()
            self._reanalyze_after_structural_change()

    def _invalidate_cached_results(self) -> None:
        """Drop everything computed under now-stale structural parameters:
        the cached lenient-floor result, the batch cache, AND the batch
        table rows (otherwise the table keeps showing stale numbers whose
        rows silently do nothing when clicked)."""
        self._raw_result = None
        self.batch_results.clear()
        self.batch_table.setRowCount(0)
        self._current_batch_name = None

    def _reanalyze_after_structural_change(self) -> None:
        """Structural parameters (detection mode, radii, spacing, ...)
        changed: cached lenient-floor results no longer apply. Re-analyze
        whatever is loaded -- the whole batch if one is loaded (repopulating
        the table rather than losing it), otherwise the current image."""
        batch_paths = [Path(raw.image_path) for raw in self.batch_results.values()]
        selected = self._current_batch_name
        self._invalidate_cached_results()
        if batch_paths:
            self._restore_batch_selection = selected
            self._start_batch(batch_paths)
        elif self.current_result is not None:
            self._run_analysis(Path(self.current_result.image_path))

    # -- detection mode (main-screen selector) --------------------------------

    def _on_mode_changed(self, mode: str) -> None:
        if mode == self.params.detection_mode:
            return  # no-op (e.g. programmatic sync after the settings dialog)
        self.params.detection_mode = mode
        self.sensitivity_description.setText(
            describe_sensitivity(apply_sensitivity(self.params, self.sensitivity_slider.value()))
        )
        self._reanalyze_after_structural_change()

    # -- edge-width criterion --------------------------------------------------

    def _sync_criterion_combo(self) -> None:
        """Reflect params.criterion_lo/hi in the dropdown without triggering
        the change handler; values matching no preset get a Custom entry
        (e.g. after being set in the Settings dialog)."""
        current = (self.params.criterion_lo, self.params.criterion_hi)
        self.criterion_combo.blockSignals(True)
        try:
            while self.criterion_combo.count() > len(_CRITERION_PRESETS):
                self.criterion_combo.removeItem(self.criterion_combo.count() - 1)
            for i in range(self.criterion_combo.count()):
                if self.criterion_combo.itemData(i) == current:
                    self.criterion_combo.setCurrentIndex(i)
                    return
            self.criterion_combo.addItem(
                f"custom ({current[0] * 100:g}–{current[1] * 100:g}%)", current
            )
            self.criterion_combo.setCurrentIndex(self.criterion_combo.count() - 1)
        finally:
            self.criterion_combo.blockSignals(False)

    def _on_criterion_changed(self, index: int) -> None:
        data = self.criterion_combo.itemData(index)
        if data is None:
            return
        lo, hi = data
        if (lo, hi) == (self.params.criterion_lo, self.params.criterion_hi):
            return
        self.params.criterion_lo, self.params.criterion_hi = lo, hi
        if self._raw_result is not None:
            # the criterion only rescales each fit's reported width, so this
            # is a pure refilter of the cached fits -- no re-analysis
            self._display_refiltered(self._raw_result)
            self._refresh_batch_rows()
            self.statusBar().showMessage(
                f"Edge-width criterion set to {self.criterion_combo.currentText()} — recomputed instantly."
            )

    # -- region of interest ----------------------------------------------------

    def _on_region_edit_toggled(self, checked: bool) -> None:
        if checked:
            if self.current_result is None:
                self.region_edit_checkbox.setChecked(False)
                return
            self.image_canvas.start_region_edit(self._on_region_drawn, initial=self.params.region)
            self.statusBar().showMessage("Drag on the image to place the region; drag its edges/corners to adjust.")
        else:
            final = self.image_canvas.stop_region_edit()
            if final is not None:
                self.params.region = final
            self.image_canvas.show_region(self.params.region)

    def _on_region_drawn(self, region: tuple[float, float, float, float]) -> None:
        self.params.region = region
        if self._raw_result is not None:
            self._display_refiltered(self._raw_result)
            self._refresh_batch_rows()
        row_min, row_max, col_min, col_max = region
        self.region_status.setText(
            f"rows {row_min:.0f}-{row_max:.0f}, cols {col_min:.0f}-{col_max:.0f} "
            f"({self.current_result.n_profiles_analyzed} profiles inside)"
        )

    def _on_region_cleared(self) -> None:
        self.params.region = None
        self.region_edit_checkbox.setChecked(False)
        self.image_canvas.stop_region_edit()
        self.image_canvas.show_region(None)
        self.region_status.setText("Whole image")
        if self._raw_result is not None:
            self._display_refiltered(self._raw_result)
            self._refresh_batch_rows()

    # -- sensitivity slider ---------------------------------------------------

    def _on_sensitivity_changed(self, value: int) -> None:
        # live preview of the thresholds this position maps to; the refilter
        # is applied once the user releases the slider.
        preview = apply_sensitivity(self.params, value)
        self.sensitivity_description.setText(describe_sensitivity(preview))

    def _on_sensitivity_released(self) -> None:
        self.params = apply_sensitivity(self.params, self.sensitivity_slider.value())
        self.sensitivity_description.setText(describe_sensitivity(self.params))
        if self._raw_result is not None:
            # detection + fitting were already done at the lenient floor;
            # new thresholds are a pure post-hoc filter -- no re-run needed
            self._display_refiltered(self._raw_result)
            self._refresh_batch_rows()
            self.statusBar().showMessage(
                f"Refiltered instantly: {self.current_result.n_profiles_analyzed} profiles accepted."
            )
        elif self.current_result is not None:
            self._run_analysis(Path(self.current_result.image_path))

    def _display_refiltered(self, raw: AnalysisResult) -> None:
        self.current_result = refilter_result(self._apply_manual_calibration(raw), self.params)
        self.results_panel.update_from_result(self.current_result)
        self._render_current_result()
        self._update_open_tool_dialogs()
        self._update_calibration_group()

    def _apply_manual_calibration(self, raw: AnalysisResult) -> AnalysisResult:
        """Rescale an uncalibrated (or previously manually-calibrated) raw
        result to the user-entered pixel size, if one is set. Results with
        a real embedded/explicit calibration pass through untouched."""
        target = self.params.fallback_pixel_size_nm
        if (
            target
            and raw.calibration in ("uncalibrated", "fallback")
            and raw.pixel_size_nm != target
        ):
            return calibrate_result(raw, target)
        return raw

    def _update_calibration_group(self) -> None:
        result = self.current_result
        show = result is not None and result.calibration in ("uncalibrated", "fallback")
        self.calibration_group.setVisible(show)
        if show and result.calibration == "fallback":
            self.calibration_spin.setValue(result.pixel_size_nm)

    def _on_manual_calibration_applied(self) -> None:
        self.params.fallback_pixel_size_nm = self.calibration_spin.value()
        if self._raw_result is not None:
            self._display_refiltered(self._raw_result)
            self._refresh_batch_rows()
            self.statusBar().showMessage(
                f"Calibrated: {self.calibration_spin.value():g} nm/px applied instantly "
                "(uncalibrated images only)."
            )

    def _update_open_tool_dialogs(self) -> None:
        # keep any open Histogram/Polar tool windows in sync with the
        # currently displayed (refiltered) result
        if self.current_result is None:
            return
        for dialog in (self._histogram_dialog, self._polar_dialog):
            if dialog is not None and dialog.isVisible():
                dialog.update_result(self.current_result)

    def _refresh_batch_rows(self) -> None:
        for name, raw in self.batch_results.items():
            self._set_batch_row(name, refilter_result(self._apply_manual_calibration(raw), self.params))

    # -- single image analysis --------------------------------------------

    def open_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
        )
        if not path_str:
            return
        self._current_batch_name = None
        self._run_analysis(Path(path_str))

    def _run_analysis(self, path: Path) -> None:
        self.statusBar().showMessage(f"Analyzing {path.name} (once, at full sensitivity)...")
        self.setEnabled(False)
        self._worker = AnalysisWorker(path, apply_sensitivity(self.params, _LENIENT_FLOOR))
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.failed.connect(self._on_analysis_failed)
        self._thread = run_in_thread(self._worker, self)

    def _on_analysis_finished(self, raw: AnalysisResult) -> None:
        self.setEnabled(True)
        self._raw_result = raw
        self._display_refiltered(raw)
        if self._current_batch_name is not None:
            self.batch_results[self._current_batch_name] = raw
            self._set_batch_row(self._current_batch_name, self.current_result)
        note = (
            "" if raw.units == "nm"
            else " No calibration found: measurements are in PIXELS (set a pixel size in Settings for nm)."
        )
        self.statusBar().showMessage(
            f"{Path(raw.image_path).name}: done. Sensitivity changes now apply instantly.{note}"
        )

    def _on_analysis_failed(self, message: str) -> None:
        self.setEnabled(True)
        self.statusBar().showMessage("Analysis failed.")
        QMessageBox.critical(self, "Analysis failed", message)

    def _render_current_result(self) -> None:
        result = self.current_result
        if result is None:
            return
        # re-rendering the SAME image (sensitivity refilter, rejected-points
        # toggle) must not throw away the user's pan/zoom; only a genuinely
        # new image resets the view
        same_image = self._last_rendered_path == result.image_path
        if same_image:
            prev_xlim = self.image_canvas.axes.get_xlim()
            prev_ylim = self.image_canvas.axes.get_ylim()

        uncalibrated_note = "" if result.units == "nm" else "  [uncalibrated: pixel units]"
        title = (
            f"{Path(result.image_path).name}\n"
            f"resolution (median) = {result.resolution_median_nm:.2f} +/- {result.resolution_mad_nm:.2f} {result.units} "
            f"(n={result.n_profiles_analyzed}){uncalibrated_note}"
        )
        self.image_canvas.set_image(result.image_gray, title=title)

        accepted = [(i, p) for i, p in enumerate(result.profiles) if p.accepted]
        self.image_canvas.set_points(
            [p.point.row for _, p in accepted],
            [p.point.col for _, p in accepted],
            [p.resolution_nm for _, p in accepted],
            profile_indices=[i for i, _ in accepted],
            units=result.units,
        )
        if self.show_rejected_checkbox.isChecked():
            rejected = [(i, p) for i, p in enumerate(result.profiles) if not p.accepted]
            self.image_canvas.set_rejected_points(
                [p.point.row for _, p in rejected],
                [p.point.col for _, p in rejected],
                profile_indices=[i for i, _ in rejected],
            )

        # restore the locked-region outline (set_image cleared the axes; the
        # interactive selector, when editing, is re-created by set_image itself)
        if not self.region_edit_checkbox.isChecked():
            self.image_canvas.show_region(self.params.region)

        if same_image:
            self.image_canvas.axes.set_xlim(prev_xlim)
            self.image_canvas.axes.set_ylim(prev_ylim)
            self.image_canvas.draw_idle()
        else:
            # reset the pan/zoom history for the freshly rendered view and
            # seed it so the toolbar's Home button always restores this full
            # view (an empty nav stack makes home() a silent no-op)
            self.image_toolbar.update()
            self.image_toolbar.push_current()
        self._last_rendered_path = result.image_path

    def _on_point_clicked(self, profile_index: int) -> None:
        if self.current_result is None:
            return
        if not (0 <= profile_index < len(self.current_result.profiles)):
            return
        profile = self.current_result.profiles[profile_index]
        self.image_canvas.highlight_point(profile.point.row, profile.point.col)
        ProfileDialog(self.current_result, self.params, profile_index, self).exec()

    # -- batch analysis -----------------------------------------------------

    def open_batch_folder(self) -> None:
        folder_str = QFileDialog.getExistingDirectory(self, "Open batch folder")
        if not folder_str:
            return
        folder = Path(folder_str)
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            QMessageBox.information(self, "No images found", f"No supported images found in {folder}")
            return
        self._invalidate_cached_results()
        self._start_batch(paths)

    def _start_batch(self, paths: list[Path]) -> None:
        self._progress_dialog = QProgressDialog("Analyzing batch...", "Cancel", 0, len(paths), self)
        self._progress_dialog.setWindowModality(Qt.WindowModal)

        self._worker = BatchAnalysisWorker(paths, apply_sensitivity(self.params, _LENIENT_FLOOR))
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.file_done.connect(self._on_batch_file_done)
        self._worker.file_failed.connect(self._on_batch_file_failed)
        self._worker.finished.connect(self._on_batch_finished)
        self._progress_dialog.canceled.connect(self._worker.stop)
        self._thread = run_in_thread(self._worker, self)

    def _on_batch_progress(self, done: int, total: int, current_name: str) -> None:
        if self._progress_dialog is None:
            return
        self._progress_dialog.setMaximum(total)
        self._progress_dialog.setValue(done)
        if current_name:
            self._progress_dialog.setLabelText(f"Analyzing {current_name}...")

    def _on_batch_file_done(self, name: str, raw: AnalysisResult) -> None:
        self.batch_results[name] = raw
        self._set_batch_row(name, refilter_result(self._apply_manual_calibration(raw), self.params))

    def _set_batch_row(self, name: str, result: AnalysisResult) -> None:
        row = next(
            (r for r in range(self.batch_table.rowCount()) if self.batch_table.item(r, 0).text() == name),
            None,
        )
        if row is None:
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
        self.batch_table.setItem(row, 0, QTableWidgetItem(name))
        self.batch_table.setItem(row, 1, QTableWidgetItem(f"{result.resolution_median_nm:.2f} {result.units}"))
        self.batch_table.setItem(row, 2, QTableWidgetItem(str(result.n_profiles_analyzed)))
        self.batch_table.resizeColumnsToContents()

    def _on_batch_file_failed(self, name: str, message: str) -> None:
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        self.batch_table.setItem(row, 0, QTableWidgetItem(name))
        self.batch_table.setItem(row, 1, QTableWidgetItem("error"))
        self.batch_table.setItem(row, 2, QTableWidgetItem(message))

    def _on_batch_finished(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        self.statusBar().showMessage(f"Batch complete: {len(self.batch_results)} image(s) analyzed.")
        # after a re-analysis triggered by a mode/settings change, put the
        # user back on the image they were viewing
        if self._restore_batch_selection:
            name = self._restore_batch_selection
            self._restore_batch_selection = None
            for row in range(self.batch_table.rowCount()):
                if self.batch_table.item(row, 0).text() == name:
                    self.batch_table.selectRow(row)  # triggers _on_batch_row_selected
                    break

    def _on_batch_row_selected(self) -> None:
        rows = self.batch_table.selectionModel().selectedRows()
        if not rows:
            return
        name = self.batch_table.item(rows[0].row(), 0).text()
        raw = self.batch_results.get(name)
        if raw is None:
            return
        self._current_batch_name = name
        self._raw_result = raw
        self._display_refiltered(raw)

    # -- tools --------------------------------------------------------------

    def show_histogram(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        # close any previous instance first: a replaced-but-open dialog
        # would keep showing stale data (only the tracked one gets updates)
        if self._histogram_dialog is not None:
            self._histogram_dialog.close()
            self._histogram_dialog.deleteLater()
        # non-modal so the main window stays usable alongside the plot
        self._histogram_dialog = HistogramDialog(self.current_result, self)
        self._histogram_dialog.show()

    def show_polar_plot(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        if self._polar_dialog is not None:
            self._polar_dialog.close()
            self._polar_dialog.deleteLater()
        # non-modal: clicking a polar point highlights it on the main image,
        # which the user needs to be able to see (and pan/zoom) while the
        # plot is open
        self._polar_dialog = PolarDialog(self.current_result, self)
        self._polar_dialog.point_selected.connect(self.image_canvas.highlight_point)
        self._polar_dialog.show()

    # -- saving ---------------------------------------------------------------

    def save_results(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        out_dir_str = QFileDialog.getExistingDirectory(self, "Save results to folder")
        if not out_dir_str:
            return
        out_dir = Path(out_dir_str)
        stem = Path(self.current_result.image_path).stem
        write_json_report(self.current_result, out_dir / f"{stem}_result.json")
        write_text_report(self.current_result, out_dir / f"{stem}_result.txt")
        save_annotated_image(self.current_result, out_dir / f"{stem}_annotated.jpg")
        save_histogram(self.current_result, out_dir / f"{stem}_histogram.png")
        save_polar_plot(self.current_result, out_dir / f"{stem}_polar.png")
        self.statusBar().showMessage(f"Saved results to {out_dir}")

    def export_batch_csv(self) -> None:
        if not self.batch_results:
            QMessageBox.information(
                self, "No batch loaded", "Open a batch folder first (File > Open Batch Folder...)."
            )
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export batch summary", "summary.csv", "CSV files (*.csv)"
        )
        if not path_str:
            return
        # export the results as currently displayed: refiltered at the active
        # thresholds/criterion/region and manual calibration, in table order
        rows = []
        for row in range(self.batch_table.rowCount()):
            name = self.batch_table.item(row, 0).text()
            raw = self.batch_results.get(name)
            if raw is not None:
                rows.append(refilter_result(self._apply_manual_calibration(raw), self.params))
        write_csv_summary(rows, path_str)
        self.statusBar().showMessage(f"Exported {len(rows)} image(s) to {path_str}")

    # -- shutdown -------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Destroying a QThread while its worker is still running aborts the
        # whole process. Ask a batch worker to stop after the current file,
        # then wait for whichever worker thread is active to finish before
        # letting the window (and the thread parented to it) be destroyed.
        if self._worker is not None and hasattr(self._worker, "stop"):
            self._worker.stop()
        if self._thread is not None and self._thread.isRunning():
            self.statusBar().showMessage("Waiting for the running analysis to finish...")
            self._thread.quit()
            self._thread.wait()
        event.accept()
