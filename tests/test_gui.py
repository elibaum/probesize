import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication

from probesize.analyze import AnalysisParams, analyze_image
from probesize.gui.main_window import MainWindow
from probesize.gui.plot_dialogs import HistogramDialog, PolarDialog
from probesize.gui.profile_dialog import ProfileDialog
from probesize.gui.sensitivity import DEFAULT_SENSITIVITY
from probesize.gui.settings_dialog import SettingsDialog

EXAMPLES = Path(__file__).resolve().parent.parent / "example_images"
pytestmark = pytest.mark.skipif(not EXAMPLES.exists(), reason="example_images not present")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def sample_result():
    path = EXAMPLES / "2.tif"
    if not path.exists():
        pytest.skip("sample file not present")
    return analyze_image(path, AnalysisParams(detection_mode="particles"))


def _result_with_path(result, name):
    """A shallow copy of an AnalysisResult under a different filename, so
    several distinct batch entries can share one analyzed sample."""
    import dataclasses

    return dataclasses.replace(result, image_path=Path(name))


class _StubDialog:
    """Stand-in for SettingsDialog that returns a preset params object."""

    def __init__(self, params):
        self._params = params

    def exec(self):
        return True

    def get_params(self):
        return self._params


def test_main_window_renders_result(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    # the incoming result is cached raw; the displayed result is a refilter
    # of it at the window's current thresholds
    assert win._raw_result is sample_result
    assert win.current_result is not sample_result
    assert win.image_canvas._rows.size == win.current_result.n_profiles_analyzed


def test_opening_multiple_images_leaves_a_single_colorbar(qapp):
    # regression test: set_image() used to reset the _colorbar reference
    # without removing the actual colorbar axes from the figure, so each
    # newly opened image left the previous colorbar stacked on the canvas.
    canvas = MainWindow().image_canvas
    for name in ("1.tif", "2.tif", "3.tif"):
        path = EXAMPLES / name
        if not path.exists():
            pytest.skip("sample file not present")
        result = analyze_image(path, AnalysisParams(detection_mode="particles"))
        canvas.set_image(result.image_gray, title=name)
        canvas.set_points(
            [p.point.row for p in result.profiles if p.accepted],
            [p.point.col for p in result.profiles if p.accepted],
            [p.resolution_nm for p in result.profiles if p.accepted],
        )

    extra_axes = [a for a in canvas.figure.axes if a is not canvas.axes]
    assert len(extra_axes) == 1


def test_point_click_maps_to_correct_profile(qapp, sample_result):
    import types

    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    result = win.current_result
    first_accepted_index = next(i for i, p in enumerate(result.profiles) if p.accepted)
    point = result.profiles[first_accepted_index].point

    emitted = []
    win.image_canvas.point_clicked.disconnect()
    win.image_canvas.point_clicked.connect(emitted.append)
    event = types.SimpleNamespace(inaxes=win.image_canvas.axes, xdata=point.col, ydata=point.row)
    win.image_canvas._on_click(event)

    assert emitted == [first_accepted_index]


def test_rejected_points_shown_and_clickable_when_toggled(qapp, sample_result):
    import types

    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    n_accepted_pts = win.image_canvas._rows.size
    win.show_rejected_checkbox.setChecked(True)

    result = win.current_result
    n_rejected = sum(1 for p in result.profiles if not p.accepted)
    assert n_rejected > 0  # lenient-floor raw guarantees some rejects at strict display thresholds
    assert win.image_canvas._rows.size == n_accepted_pts + n_rejected

    # clicking a rejected point emits its profiles-list index
    rejected_index = next(i for i, p in enumerate(result.profiles) if not p.accepted)
    point = result.profiles[rejected_index].point
    emitted = []
    win.image_canvas.point_clicked.disconnect()
    win.image_canvas.point_clicked.connect(emitted.append)
    event = types.SimpleNamespace(inaxes=win.image_canvas.axes, xdata=point.col, ydata=point.row)
    win.image_canvas._on_click(event)

    assert len(emitted) == 1
    assert not result.profiles[emitted[0]].accepted

    win.show_rejected_checkbox.setChecked(False)
    assert win.image_canvas._rows.size == n_accepted_pts


def test_tool_dialogs_construct(qapp, sample_result):
    HistogramDialog(sample_result)
    PolarDialog(sample_result)


def test_histogram_bins_slider_redraws_with_requested_bins(qapp, sample_result):
    dialog = HistogramDialog(sample_result)
    assert len(dialog.canvas.axes.patches) == 30  # default

    dialog.bins_slider.setValue(12)

    assert len(dialog.canvas.axes.patches) == 12
    assert dialog.bins_label.text() == "Bins: 12"


def test_polar_point_pick_emits_image_location_and_marks_selection(qapp, sample_result):
    import types

    dialog = PolarDialog(sample_result)
    accepted = [p for p in sample_result.profiles if p.accepted]
    assert accepted

    received = []
    dialog.point_selected.connect(lambda row, col: received.append((row, col)))
    event = types.SimpleNamespace(artist=dialog._scatter, ind=[3])
    dialog._on_pick(event)

    assert received == [(accepted[3].point.row, accepted[3].point.col)]
    assert dialog._selection_artists  # ring marker drawn on the polar plot


def test_polar_plot_orientation_matches_image(qapp, sample_result):
    # edge-normal angles use a downward-pointing image row axis, so the polar
    # plot must run clockwise from East to line up with the image (otherwise
    # it's vertically mirrored: the top of a particle plots at the bottom).
    dialog = PolarDialog(sample_result)
    axes = dialog.canvas.axes

    assert axes.get_theta_direction() == -1
    assert axes.get_theta_offset() == pytest.approx(0.0)  # 0 rad = East

    # geometric check: a normal pointing up in the image (angle -90 deg)
    # must plot in the upper half of the polar axes
    dialog.canvas.draw()
    cx, cy = axes.transData.transform((0.0, 0.0))
    x_up, y_up = axes.transData.transform((np.deg2rad(-90), 1.0))
    assert y_up > cy  # plotted above center, not below


def test_polar_pick_preserves_radial_zoom(qapp, sample_result):
    import types

    dialog = PolarDialog(sample_result)
    dialog._on_scroll(types.SimpleNamespace(inaxes=dialog.canvas.axes, button="up"))
    dialog._on_scroll(types.SimpleNamespace(inaxes=dialog.canvas.axes, button="up"))
    zoomed_rmax = dialog.canvas.axes.get_rmax()

    dialog._on_pick(types.SimpleNamespace(artist=dialog._scatter, ind=[0]))

    # drawing the selection marker must not re-autoscale and undo the zoom
    assert dialog.canvas.axes.get_rmax() == pytest.approx(zoomed_rmax)


def test_polar_selection_highlights_point_on_main_image(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)
    assert not win.image_canvas._highlight_artists

    win.show_polar_plot()
    win._polar_dialog.point_selected.emit(50.0, 60.0)

    assert win.image_canvas._highlight_artists
    win._polar_dialog.close()


def test_open_tool_dialogs_update_live_on_sensitivity_change(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    win.show_histogram()
    win.show_polar_plot()
    hist, polar = win._histogram_dialog, win._polar_dialog
    before = len(hist._values)

    # move to strictest and release -> refilter should push into open dialogs
    win.sensitivity_slider.setValue(0)
    win._on_sensitivity_released()

    assert len(hist._values) == win.current_result.n_profiles_analyzed
    assert len(polar._values) == win.current_result.n_profiles_analyzed
    assert len(hist._values) != before  # strict thresholds changed the set
    hist.close()
    polar.close()


def test_closed_tool_dialogs_are_not_updated(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)
    win.show_histogram()
    win._histogram_dialog.close()

    # updating after close must not raise (guarded by isVisible)
    win.sensitivity_slider.setValue(10)
    win._on_sensitivity_released()


def test_mode_selector_reanalyzes_single_image_and_invalidates_cache(qapp, sample_result, monkeypatch):
    win = MainWindow()
    win.mode_combo.setCurrentText("edge")  # start in edge, no re-run (no image yet)
    win._on_analysis_finished(sample_result)
    assert win._raw_result is sample_result

    reruns = []
    monkeypatch.setattr(win, "_run_analysis", lambda path: reruns.append((path, win.params.detection_mode)))

    win.mode_combo.setCurrentText("particles")

    assert win.params.detection_mode == "particles"
    assert win._raw_result is None  # structural change invalidated the cache
    assert reruns == [(Path(sample_result.image_path), "particles")]


def test_mode_change_reanalyzes_whole_batch(qapp, sample_result, monkeypatch):
    # regression: switching detection mode with a batch loaded used to clear
    # the batch table (images "disappeared") instead of re-analyzing them
    win = MainWindow()
    win.mode_combo.setCurrentText("edge")
    win._on_batch_file_done("a.tif", _result_with_path(sample_result, "a.tif"))
    win._on_batch_file_done("b.tif", _result_with_path(sample_result, "b.tif"))
    win._on_batch_file_done("c.tif", _result_with_path(sample_result, "c.tif"))
    assert win.batch_table.rowCount() == 3

    started = []
    monkeypatch.setattr(win, "_start_batch", lambda paths: started.append([p.name for p in paths]))

    win.mode_combo.setCurrentText("particles")

    assert win.params.detection_mode == "particles"
    assert len(started) == 1
    assert sorted(started[0]) == ["a.tif", "b.tif", "c.tif"]  # all three re-analyzed


def test_settings_dialog_mode_change_syncs_main_selector(qapp, sample_result, monkeypatch):
    import copy

    win = MainWindow()
    win.mode_combo.setCurrentText("edge")
    monkeypatch.setattr(win, "_run_analysis", lambda path: None)

    def _changed_params(params, parent):
        p = copy.copy(params)  # real SettingsDialog copies, then mutates
        p.detection_mode = "particles"
        return _StubDialog(p)

    monkeypatch.setattr("probesize.gui.main_window.SettingsDialog", _changed_params)
    win.open_settings()

    assert win.mode_combo.currentText() == "particles"
    assert win.params.detection_mode == "particles"


def test_polar_plot_scroll_zooms_radial_axis(qapp, sample_result):
    import types

    dialog = PolarDialog(sample_result)
    initial_rmax = dialog.canvas.axes.get_rmax()

    zoom_in = types.SimpleNamespace(inaxes=dialog.canvas.axes, button="up")
    dialog._on_scroll(zoom_in)
    assert dialog.canvas.axes.get_rmax() < initial_rmax

    zoom_out = types.SimpleNamespace(inaxes=dialog.canvas.axes, button="down")
    dialog._on_scroll(zoom_out)
    dialog._on_scroll(zoom_out)
    assert dialog.canvas.axes.get_rmax() > initial_rmax

    # scroll events outside the plot axes (e.g. over the toolbar) are ignored
    outside = types.SimpleNamespace(inaxes=None, button="up")
    rmax_before = dialog.canvas.axes.get_rmax()
    dialog._on_scroll(outside)
    assert dialog.canvas.axes.get_rmax() == rmax_before


def test_main_canvas_home_restores_full_view_after_zoom(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    home_xlim = win.image_canvas.axes.get_xlim()
    win.image_canvas.axes.set_xlim(100, 200)  # simulate a box zoom
    win.image_toolbar.push_current()

    win.image_toolbar.home()

    assert win.image_canvas.axes.get_xlim() == pytest.approx(home_xlim)


def test_point_click_ignored_while_pan_zoom_tool_active(qapp, sample_result):
    import types

    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    result = win.current_result
    idx = next(i for i, p in enumerate(result.profiles) if p.accepted)
    point = result.profiles[idx].point

    emitted = []
    win.image_canvas.point_clicked.disconnect()
    win.image_canvas.point_clicked.connect(emitted.append)
    event = types.SimpleNamespace(inaxes=win.image_canvas.axes, xdata=point.col, ydata=point.row)

    win.image_toolbar.zoom()  # activate the box-zoom tool
    win.image_canvas._on_click(event)
    assert emitted == []

    win.image_toolbar.zoom()  # deactivate
    win.image_canvas._on_click(event)
    assert emitted == [idx]


def test_polar_reset_view_restores_original_after_scroll_zoom(qapp, sample_result):
    import types

    dialog = PolarDialog(sample_result)
    initial_rmax = dialog.canvas.axes.get_rmax()

    zoom_in = types.SimpleNamespace(inaxes=dialog.canvas.axes, button="up")
    for _ in range(5):
        dialog._on_scroll(zoom_in)
    assert dialog.canvas.axes.get_rmax() != pytest.approx(initial_rmax)

    dialog.reset_view()

    assert dialog.canvas.axes.get_rmax() == pytest.approx(initial_rmax)


def test_profile_dialog_construct(qapp, sample_result):
    accepted_idx = next(i for i, p in enumerate(sample_result.profiles) if p.accepted)
    ProfileDialog(sample_result, AnalysisParams(detection_mode="particles"), accepted_idx)


def test_settings_dialog_roundtrip(qapp):
    dialog = SettingsDialog(AnalysisParams())
    dialog.mode_combo.setCurrentText("particles")
    dialog.min_radius_nm.setValue(5.0)
    params = dialog.get_params()

    assert params.detection_mode == "particles"
    assert params.min_radius_nm == pytest.approx(5.0)


def test_sensitivity_slider_starts_at_default(qapp):
    win = MainWindow()
    assert win.sensitivity_slider.value() == DEFAULT_SENSITIVITY


def test_moving_slider_previews_without_committing(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)
    params_before = win.params

    win.sensitivity_slider.setValue(95)  # emits valueChanged only, not sliderReleased

    assert win.params is params_before
    assert "circularity" in win.sensitivity_description.text()


def test_releasing_slider_refilters_instantly_without_rerunning(qapp, sample_result, monkeypatch):
    win = MainWindow()
    win.params.detection_mode = "particles"
    original_circularity = win.params.min_circularity
    win._on_analysis_finished(sample_result)
    strict_count = win.current_result.n_profiles_analyzed

    calls = []
    monkeypatch.setattr(win, "_run_analysis", lambda path: calls.append(path))

    win.sensitivity_slider.setValue(95)
    win._on_sensitivity_released()

    assert win.params.min_circularity < original_circularity
    assert calls == []  # no re-analysis: pure post-hoc refilter of the cached raw result
    assert win.current_result.n_profiles_analyzed >= strict_count


def test_releasing_slider_without_cached_raw_falls_back_to_rerun(qapp, sample_result, monkeypatch):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)
    win._raw_result = None  # e.g. invalidated by a settings change

    calls = []
    monkeypatch.setattr(win, "_run_analysis", lambda path: calls.append(path))

    win.sensitivity_slider.setValue(95)
    win._on_sensitivity_released()

    assert calls == [Path(sample_result.image_path)]


def test_region_drawn_refilters_instantly_and_restricts_points(qapp, sample_result, monkeypatch):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)
    full_count = win.current_result.n_profiles_analyzed
    assert full_count > 0

    reruns = []
    monkeypatch.setattr(win, "_run_analysis", lambda path: reruns.append(path))

    win.region_edit_checkbox.setChecked(True)
    win._on_region_drawn((0.0, 512.0, 0.0, 512.0))  # top-left quadrant

    assert reruns == []  # pure post-hoc refilter
    restricted = win.current_result
    assert 0 < restricted.n_profiles_analyzed < full_count
    assert all(
        p.point.row <= 512 and p.point.col <= 512
        for p in restricted.profiles
        if p.accepted
    )
    assert "profiles inside" in win.region_status.text()

    # clearing restores the full set and the status label
    win._on_region_cleared()
    assert win.current_result.n_profiles_analyzed == full_count
    assert win.region_status.text() == "Whole image"
    assert win.params.region is None


def test_point_clicks_suppressed_while_editing_region(qapp, sample_result):
    import types

    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    idx = next(i for i, p in enumerate(win.current_result.profiles) if p.accepted)
    point = win.current_result.profiles[idx].point
    emitted = []
    win.image_canvas.point_clicked.disconnect()
    win.image_canvas.point_clicked.connect(emitted.append)
    event = types.SimpleNamespace(inaxes=win.image_canvas.axes, xdata=point.col, ydata=point.row)

    win.region_edit_checkbox.setChecked(True)
    win.image_canvas._on_click(event)
    assert emitted == []  # clicks manipulate the rectangle, not the inspector

    win.region_edit_checkbox.setChecked(False)
    win.image_canvas._on_click(event)
    assert emitted == [idx]


def test_locked_region_survives_rerender_as_static_patch(qapp, sample_result):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    win.region_edit_checkbox.setChecked(True)
    win._on_region_drawn((100.0, 400.0, 100.0, 400.0))
    win.region_edit_checkbox.setChecked(False)  # lock it

    # a sensitivity refilter re-renders the canvas; the static outline must persist
    win.sensitivity_slider.setValue(80)
    win._on_sensitivity_released()

    assert win.image_canvas._region_patch is not None
    assert win.params.region == (100.0, 400.0, 100.0, 400.0)


def test_settings_change_reanalyzes_loaded_batch(qapp, sample_result, monkeypatch):
    # a structural settings change with a batch loaded re-analyzes the whole
    # batch (rather than clearing it or leaving stale rows)
    import copy

    win = MainWindow()
    win.mode_combo.setCurrentText("edge")
    win._on_batch_file_done("a.tif", _result_with_path(sample_result, "a.tif"))
    win._on_batch_file_done("b.tif", _result_with_path(sample_result, "b.tif"))

    started = []
    monkeypatch.setattr(win, "_start_batch", lambda paths: started.append([p.name for p in paths]))

    def _changed(params, parent):
        p = copy.copy(params)
        p.min_radius_nm = params.min_radius_nm + 1.0  # a genuine structural change
        return _StubDialog(p)

    monkeypatch.setattr("probesize.gui.main_window.SettingsDialog", _changed)
    win.open_settings()

    assert len(started) == 1
    assert sorted(started[0]) == ["a.tif", "b.tif"]


def test_settings_dialog_no_change_is_a_noop(qapp, sample_result, monkeypatch):
    # accepting the dialog without changing anything must not blow away the
    # loaded batch or trigger a re-analysis
    import copy

    win = MainWindow()
    win.mode_combo.setCurrentText("edge")
    win._on_batch_file_done("a.tif", _result_with_path(sample_result, "a.tif"))

    started = []
    monkeypatch.setattr(win, "_start_batch", lambda paths: started.append(paths))
    monkeypatch.setattr(
        "probesize.gui.main_window.SettingsDialog",
        lambda params, parent: _StubDialog(copy.copy(params)),  # identical params
    )
    win.open_settings()

    assert started == []
    assert win.batch_table.rowCount() == 1


def test_reopening_tool_dialog_closes_previous_instance(qapp, sample_result):
    # regression: reopening the histogram left the first window open but
    # untracked, so it never received live updates again
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    win.show_histogram()
    first = win._histogram_dialog
    win.show_histogram()

    assert not first.isVisible()
    assert win._histogram_dialog is not first
    win._histogram_dialog.close()


def test_rerender_of_same_image_preserves_canvas_zoom(qapp, sample_result):
    # regression: a sensitivity refilter or rejected-toggle re-render used
    # to reset the user's pan/zoom on the main image
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)

    win.image_canvas.axes.set_xlim(100, 200)
    win.image_canvas.axes.set_ylim(220, 120)

    win.sensitivity_slider.setValue(10)
    win._on_sensitivity_released()  # refilter -> re-render of the same image

    assert win.image_canvas.axes.get_xlim() == pytest.approx((100, 200))
    assert win.image_canvas.axes.get_ylim() == pytest.approx((220, 120))

    # a genuinely different image DOES reset the view
    other = analyze_image(EXAMPLES / "1.tif", AnalysisParams(detection_mode="particles"))
    win._on_analysis_finished(other)
    assert win.image_canvas.axes.get_xlim() != pytest.approx((100, 200))


def test_uncalibrated_result_labeled_in_pixels_across_gui(qapp, tmp_path):
    import numpy as np
    from PIL import Image
    from scipy.special import erf

    # uncalibrated synthetic edge PNG -> pipeline falls back to px units
    size, sigma = 240, 2.0
    cols = np.arange(size)
    img = np.tile(40 + 180 * 0.5 * (1 + erf((cols - 120) / (np.sqrt(2) * sigma))), (size, 1))
    path = tmp_path / "uncalibrated.png"
    Image.fromarray(np.clip(img + np.random.default_rng(1).normal(0, 3, img.shape), 0, 255).astype(np.uint8)).save(path)

    result = analyze_image(path, AnalysisParams(canny_sigma=2.0))
    assert result.units == "px"

    win = MainWindow()
    win._on_analysis_finished(result)

    assert win.current_result.units == "px"
    assert win.results_panel._labels["pixel_size"].text() == "uncalibrated — measuring in pixels"
    assert win.results_panel._labels["resolution_median"].text().endswith("px")

    win.show_histogram()
    assert win._histogram_dialog.canvas.axes.get_xlabel() == "resolution (px)"
    win._histogram_dialog.close()


def _uncalibrated_edge_result(tmp_path):
    import numpy as np
    from PIL import Image
    from scipy.special import erf

    size, sigma = 240, 2.0
    cols = np.arange(size)
    img = np.tile(40 + 180 * 0.5 * (1 + erf((cols - 120) / (np.sqrt(2) * sigma))), (size, 1))
    path = tmp_path / "uncal.png"
    Image.fromarray(np.clip(img + np.random.default_rng(1).normal(0, 3, img.shape), 0, 255).astype(np.uint8)).save(path)
    return analyze_image(path, AnalysisParams(canny_sigma=2.0))


def test_manual_calibration_group_shown_only_when_uncalibrated(qapp, sample_result, tmp_path):
    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)  # real Zeiss result: calibrated
    assert not win.calibration_group.isVisibleTo(win)

    win.params.detection_mode = "edge"
    win._on_analysis_finished(_uncalibrated_edge_result(tmp_path))
    assert win.calibration_group.isVisibleTo(win)


def test_manual_calibration_applies_instantly_and_converts_to_nm(qapp, tmp_path, monkeypatch):
    win = MainWindow()
    win.params.detection_mode = "edge"
    raw = _uncalibrated_edge_result(tmp_path)
    win._on_analysis_finished(raw)
    px_median = win.current_result.resolution_median_nm
    assert win.current_result.units == "px"

    reruns = []
    monkeypatch.setattr(win, "_run_analysis", lambda path: reruns.append(path))
    win.calibration_spin.setValue(0.5)
    win._on_manual_calibration_applied()

    assert reruns == []  # instant rescale, no re-analysis
    assert win.current_result.units == "nm"
    assert win.current_result.calibration == "fallback"
    assert win.current_result.resolution_median_nm == pytest.approx(0.5 * px_median)
    assert "(manual)" in win.results_panel._labels["pixel_size"].text()
    # group stays visible so the value can be corrected
    assert win.calibration_group.isVisibleTo(win)

    # correcting the value rescales from the pristine raw, not cumulatively
    win.calibration_spin.setValue(1.0)
    win._on_manual_calibration_applied()
    assert win.current_result.resolution_median_nm == pytest.approx(px_median)


def test_settings_change_invalidates_cached_raw_result(qapp, sample_result, monkeypatch):
    import copy

    win = MainWindow()
    win.params.detection_mode = "particles"
    win._on_analysis_finished(sample_result)
    assert win._raw_result is not None
    monkeypatch.setattr(win, "_run_analysis", lambda path: None)  # avoid a real thread

    def _changed(params, parent):
        p = copy.copy(params)
        p.min_radius_nm = params.min_radius_nm + 1.0
        return _StubDialog(p)

    monkeypatch.setattr("probesize.gui.main_window.SettingsDialog", _changed)
    win.open_settings()

    assert win._raw_result is None


def test_lenient_sensitivity_recovers_a_zero_detection_image(qapp):
    # the case the slider exists for: at the strictest setting, 3.tif's
    # particle boundaries are marginal enough that every candidate is
    # rejected (0 accepted profiles) -- the exact "not working" scenario
    # reported for some gold-on-carbon images.
    from probesize.gui.sensitivity import apply_sensitivity

    path = EXAMPLES / "3.tif"
    if not path.exists():
        pytest.skip("sample file not present")

    strict = apply_sensitivity(AnalysisParams(detection_mode="particles"), 0)
    lenient = apply_sensitivity(AnalysisParams(detection_mode="particles"), 100)

    strict_result = analyze_image(path, strict)
    assert strict_result.n_profiles_analyzed == 0
    lenient_result = analyze_image(path, lenient)

    assert lenient_result.n_profiles_analyzed > strict_result.n_profiles_analyzed
