"""Entry point for the probesize desktop GUI."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("probesize")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
