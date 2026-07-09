from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from .. import __version__


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About probesize")
        layout = QVBoxLayout(self)
        label = QLabel(
            f"<b>probesize</b> v{__version__}<br><br>"
            "Resolution analysis for charged-particle microscope "
            "(SEM / FIB / HIM) images, using the knife-edge / "
            "edge-spread-function method.<br><br>"
            "MIT licensed."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
