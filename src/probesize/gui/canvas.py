"""Matplotlib-in-Qt canvases used by the main window and tool dialogs."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PySide6.QtCore import Signal

Region = tuple[float, float, float, float]  # (row_min, row_max, col_min, col_max)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 5), polar: bool = False):
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.axes = self.figure.add_subplot(111, projection="polar" if polar else None)
        super().__init__(self.figure)


class ImageCanvas(MplCanvas):
    """Displays a grayscale image with an overlaid, resolution-colored
    scatter of accepted profile points, and optionally the rejected points
    in grey. Clicking near any displayed point emits the profile index the
    caller associated with it via :meth:`set_points` /
    :meth:`set_rejected_points`."""

    point_clicked = Signal(int)

    def __init__(self):
        super().__init__(figsize=(6, 6))
        self._scatter = None
        self._rejected_scatter = None
        self._sampling_limited_scatter = None
        self._colorbar = None
        self._highlight_artists = []
        self._rows = np.array([])
        self._cols = np.array([])
        self._profile_indices = np.array([], dtype=int)
        self._region_selector: Optional[RectangleSelector] = None
        self._region_callback: Optional[Callable[[Region], None]] = None
        self._region_patch: Optional[Rectangle] = None
        self.mpl_connect("button_press_event", self._on_click)

    def _remove_colorbar(self) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

    def set_image(self, image: np.ndarray, title: str = "") -> None:
        self._remove_colorbar()
        was_editing = self._region_selector is not None
        callback = self._region_callback
        editing_region = self.stop_region_edit() if was_editing else None
        self.axes.clear()  # also discards the static region patch artist
        self._region_patch = None
        self._scatter = None
        self._rejected_scatter = None
        self._sampling_limited_scatter = None
        self._highlight_artists = []
        self._rows = np.array([])
        self._cols = np.array([])
        self._profile_indices = np.array([], dtype=int)
        self.axes.imshow(image, cmap="gray")
        self.axes.set_title(title, fontsize=9)
        self.axes.axis("off")
        if was_editing and callback is not None:
            # re-create the selector on the fresh axes so region editing
            # survives re-renders (refilters re-draw the whole canvas)
            self.start_region_edit(callback, initial=editing_region)
        self.draw_idle()

    def set_points(self, rows, cols, values, profile_indices=None, units: str = "nm") -> None:
        """Show accepted points colored by resolution. ``profile_indices``
        maps each point back to its index in ``result.profiles`` for click
        handling; defaults to 0..n-1."""
        rows, cols = np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)
        if profile_indices is None:
            profile_indices = np.arange(len(rows))
        self._rows, self._cols = rows, cols
        self._profile_indices = np.asarray(profile_indices, dtype=int)
        self._remove_colorbar()
        if len(rows):
            self._scatter = self.axes.scatter(cols, rows, c=values, cmap="jet", s=10, linewidths=0, picker=5)
            self._colorbar = self.figure.colorbar(
                self._scatter, ax=self.axes, label=f"resolution ({units})", fraction=0.046, pad=0.04
            )
        self.draw_idle()

    def set_sampling_limited_points(self, rows, cols, profile_indices) -> None:
        """Show profiles excluded for being finer than the pixel grid can
        resolve, in a flat high-visibility colour.

        Drawn unconditionally (unlike the rejected-points overlay, which is
        opt-in) because these are measurements the image *could* have given
        at a finer pixel size -- if they simply vanished, an undersampled
        image would look sparsely detected for no visible reason.
        """
        rows, cols = np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)
        if len(rows):
            self._sampling_limited_scatter = self.axes.scatter(
                cols, rows, c="magenta", s=8, linewidths=0, alpha=0.85, picker=5,
                label="excluded: sampling-limited",
            )
            self._rows = np.concatenate([self._rows, rows])
            self._cols = np.concatenate([self._cols, cols])
            self._profile_indices = np.concatenate(
                [self._profile_indices, np.asarray(profile_indices, dtype=int)]
            )
            self.axes.legend(loc="upper right", fontsize=7, framealpha=0.8)
        self.draw_idle()

    def set_rejected_points(self, rows, cols, profile_indices) -> None:
        """Additionally show rejected points in grey (call after
        :meth:`set_points`); clicking one emits its profile index too."""
        rows, cols = np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)
        if len(rows):
            self._rejected_scatter = self.axes.scatter(
                cols, rows, c="lightgray", s=6, linewidths=0, alpha=0.7, zorder=1
            )
            self._rows = np.concatenate([self._rows, rows])
            self._cols = np.concatenate([self._cols, cols])
            self._profile_indices = np.concatenate(
                [self._profile_indices, np.asarray(profile_indices, dtype=int)]
            )
        self.draw_idle()

    def highlight_point(self, row: float, col: float) -> None:
        for artist in self._highlight_artists:
            artist.remove()
        self._highlight_artists = self.axes.plot(col, row, "o", markersize=14, markerfacecolor="none", markeredgecolor="white", markeredgewidth=2)
        self.draw_idle()

    # -- region of interest --------------------------------------------------

    def start_region_edit(self, callback: Callable[[Region], None], initial: Optional[Region] = None) -> None:
        """Enter region-edit mode: the user drags a rectangle on the image
        (movable/resizable via its handles); `callback` receives the region
        as (row_min, row_max, col_min, col_max) on each release. Point-click
        inspection is suspended while editing."""
        self.show_region(None)
        self._region_callback = callback
        self._region_selector = RectangleSelector(
            self.axes,
            self._on_region_select,
            useblit=False,  # blitting misbehaves on freshly re-rendered axes
            interactive=True,
            button=[1],
            props={"edgecolor": "tab:blue", "fill": False, "linewidth": 1.5},
        )
        if initial is not None:
            row_min, row_max, col_min, col_max = initial
            self._region_selector.extents = (col_min, col_max, row_min, row_max)
        self.draw_idle()

    def stop_region_edit(self) -> Optional[Region]:
        """Leave region-edit mode, returning the final region (or None if
        no rectangle was drawn)."""
        region = None
        if self._region_selector is not None:
            col_min, col_max, row_min, row_max = self._region_selector.extents
            if row_max > row_min and col_max > col_min:
                region = (row_min, row_max, col_min, col_max)
            self._region_selector.set_active(False)
            self._region_selector.set_visible(False)
            self._region_selector = None
        self._region_callback = None
        self.draw_idle()
        return region

    def show_region(self, region: Optional[Region]) -> None:
        """Draw (or clear, with None) a static outline of the locked region."""
        if self._region_patch is not None:
            self._region_patch.remove()
            self._region_patch = None
        if region is not None:
            row_min, row_max, col_min, col_max = region
            self._region_patch = Rectangle(
                (col_min, row_min), col_max - col_min, row_max - row_min,
                fill=False, edgecolor="tab:blue", linewidth=1.5,
            )
            self.axes.add_patch(self._region_patch)
        self.draw_idle()

    def _on_region_select(self, _eclick, _erelease) -> None:
        if self._region_selector is None or self._region_callback is None:
            return
        col_min, col_max, row_min, row_max = self._region_selector.extents
        if row_max > row_min and col_max > col_min:
            self._region_callback((row_min, row_max, col_min, col_max))

    def _on_click(self, event) -> None:
        if self._region_selector is not None:
            return  # region-edit mode: clicks manipulate the rectangle
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # pan/zoom tool is active; don't treat drags as point clicks
        if event.inaxes != self.axes or self._rows.size == 0:
            return
        distances = np.hypot(self._rows - event.ydata, self._cols - event.xdata)
        idx = int(np.argmin(distances))
        if distances[idx] < 15:  # pixels, generous click tolerance
            self.point_clicked.emit(int(self._profile_indices[idx]))
