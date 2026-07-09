"""Background-thread workers so long analyses don't freeze the UI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from ..analyze import AnalysisParams, AnalysisResult, analyze_image


class AnalysisWorker(QObject):
    """Runs :func:`analyze_image` for a single file on a worker thread."""

    finished = Signal(object)  # AnalysisResult
    failed = Signal(str)

    def __init__(self, path: Path, params: AnalysisParams):
        super().__init__()
        self.path = path
        self.params = params

    def run(self) -> None:
        try:
            result = analyze_image(self.path, self.params)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class BatchAnalysisWorker(QObject):
    """Runs :func:`analyze_image` over every image in a folder, reporting
    progress after each file and continuing past per-file failures."""

    progress = Signal(int, int, str)  # done, total, current filename
    file_done = Signal(str, object)  # filename, AnalysisResult or None
    file_failed = Signal(str, str)  # filename, error message
    finished = Signal()

    def __init__(self, paths: list[Path], params: AnalysisParams):
        super().__init__()
        self.paths = paths
        self.params = params
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._stop_requested:
                break
            self.progress.emit(i, total, path.name)
            try:
                result: AnalysisResult = analyze_image(path, self.params)
            except Exception as exc:  # noqa: BLE001
                self.file_failed.emit(path.name, str(exc))
                continue
            self.file_done.emit(path.name, result)
        self.progress.emit(total, total, "")
        self.finished.emit()


def run_in_thread(worker: QObject, parent: QObject) -> QThread:
    """Move `worker` to a new QThread, wire start/cleanup, and start it.
    The caller must keep a reference to the returned thread (and the
    worker) alive until it finishes -- store both on the parent widget.
    """
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for signal_name in ("finished", "failed"):
        signal = getattr(worker, signal_name, None)
        if signal is not None:
            signal.connect(thread.quit)
    thread.start()
    return thread
