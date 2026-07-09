"""Histogram and polar-plot tool windows for the currently loaded result.

Both are shown non-modally so the main window stays interactive alongside
them -- in particular, clicking a point in the polar plot highlights the
corresponding location on the main image view -- and both expose
:meth:`update_result` so the main window can refresh them live when the
detection sensitivity changes.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

from ..analyze import AnalysisResult
from .canvas import MplCanvas

_DEFAULT_BINS = 30


class HistogramDialog(QDialog):
    def __init__(self, result: AnalysisResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resolution histogram")
        self.resize(500, 470)

        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(figsize=(5, 4))
        layout.addWidget(self.canvas)

        self.bins_slider = QSlider(Qt.Horizontal)
        self.bins_slider.setRange(5, 100)
        self.bins_slider.setValue(_DEFAULT_BINS)
        self.bins_slider.valueChanged.connect(self._redraw)
        self.bins_label = QLabel(f"Bins: {_DEFAULT_BINS}")

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.bins_label)
        slider_row.addWidget(self.bins_slider)
        layout.addLayout(slider_row)

        self.update_result(result)

    def update_result(self, result: AnalysisResult) -> None:
        self._values = [p.resolution_nm for p in result.profiles if p.accepted]
        self._mean = result.resolution_mean_nm
        self._median = result.resolution_median_nm
        self._units = result.units
        self._redraw(self.bins_slider.value())

    def _redraw(self, n_bins: int) -> None:
        self.bins_label.setText(f"Bins: {n_bins}")
        axes = self.canvas.axes
        axes.clear()
        if self._values:
            axes.hist(self._values, bins=n_bins, color="tab:green", edgecolor="black")
            axes.axvline(self._mean, color="tab:red", label="mean")
            axes.axvline(self._median, color="tab:blue", label="median")
            axes.legend()
        axes.set_xlabel(f"resolution ({self._units})")
        axes.set_ylabel("count")
        self.canvas.draw_idle()


_RADIAL_ZOOM_FACTOR = 0.85  # per zoom step / scroll notch


class PolarDialog(QDialog):
    """Angular distribution of resolution values -- anisotropic effects like
    astigmatism or coma show up as a non-circular distribution.

    matplotlib's navigation toolbar is a poor fit for a polar projection
    (it disables rectangle-zoom and pan barely moves a full-circle plot), so
    this dialog provides its own radial (resolution-axis) zoom controls plus
    mouse-wheel zoom instead. Clicking a point emits :attr:`point_selected`
    with that profile's (row, col) image location so the owner can highlight
    it on the main image view.
    """

    point_selected = Signal(float, float)  # row, col in image coordinates

    def __init__(self, result: AnalysisResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Polar plot")
        self.resize(480, 560)
        layout = QVBoxLayout(self)

        self.canvas = MplCanvas(figsize=(4.5, 4.5), polar=True)
        self._scatter = None
        self._selection_artists: list = []
        self._home_rmax = 1.0
        layout.addWidget(self.canvas)

        controls = QHBoxLayout()
        zoom_in = QPushButton("Zoom in")
        zoom_in.clicked.connect(lambda: self._zoom(_RADIAL_ZOOM_FACTOR))
        zoom_out = QPushButton("Zoom out")
        zoom_out.clicked.connect(lambda: self._zoom(1.0 / _RADIAL_ZOOM_FACTOR))
        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        controls.addWidget(zoom_in)
        controls.addWidget(zoom_out)
        controls.addWidget(reset)
        controls.addStretch()
        layout.addLayout(controls)

        hint = QLabel(
            "Angle = edge-normal orientation, aligned to the image "
            "(0° = right, 90° = down). Scroll or use the buttons to zoom "
            "the resolution axis; click a point to locate it on the main image."
        )
        hint.setStyleSheet("color: gray; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.update_result(result)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def update_result(self, result: AnalysisResult) -> None:
        self.setWindowTitle(f"Polar plot — resolution in {result.units}")
        accepted = [p for p in result.profiles if p.accepted]
        self._angles = [p.point.angle for p in accepted]
        self._values = [p.resolution_nm for p in accepted]
        self._rows = [p.point.row for p in accepted]
        self._cols = [p.point.col for p in accepted]

        axes = self.canvas.axes
        # keep the user's current zoom across a data refresh, if they had one
        prev_rmax = axes.get_rmax() if self._scatter is not None else None
        axes.clear()
        # Align angular orientation with the image: edge-normal angles are
        # atan2(d_row, d_col) with the image row axis pointing DOWN, whereas a
        # default polar plot measures theta counter-clockwise (up). Making
        # theta increase clockwise (from East) un-mirrors it, so e.g. the top
        # of a particle's boundary appears at the top of the polar plot.
        axes.set_theta_zero_location("E")
        axes.set_theta_direction(-1)
        self._scatter = None
        self._selection_artists = []
        if accepted:
            self._scatter = axes.scatter(
                self._angles, self._values, s=8, c="tab:blue", alpha=0.6, picker=5
            )
        self._home_rmax = (max(self._values) * 1.05) if self._values else 1.0
        axes.set_rmax(prev_rmax if prev_rmax else self._home_rmax)
        # disable autoscaling so later artists (the pick marker) never rescale
        # the radial axis and undo the current zoom
        axes.autoscale(False)
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        self.canvas.axes.set_rmax(self._home_rmax)
        self.canvas.draw_idle()

    def _zoom(self, factor: float) -> None:
        axes = self.canvas.axes
        axes.set_rmax(max(axes.get_rmax() * factor, 1e-6))
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.canvas.axes:
            return
        self._zoom(_RADIAL_ZOOM_FACTOR if event.button == "up" else 1.0 / _RADIAL_ZOOM_FACTOR)

    def _on_pick(self, event) -> None:
        if event.artist is not self._scatter or len(event.ind) == 0:
            return
        index = int(event.ind[0])

        for artist in self._selection_artists:
            artist.remove()
        self._selection_artists = self.canvas.axes.plot(
            self._angles[index],
            self._values[index],
            "o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="tab:red",
            markeredgewidth=2,
        )
        self.canvas.draw_idle()

        self.point_selected.emit(self._rows[index], self._cols[index])
