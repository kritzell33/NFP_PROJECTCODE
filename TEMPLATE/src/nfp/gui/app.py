"""NFP Assessor desktop shell (PySide6).

v0.1 is a thin window over the same library the CLI uses: pick a crew file,
a scenario, set replications and seed, run, open the report. Phase 2 adds a
registry-driven profile editor, a live pyqtgraph dashboard while the batch
runs, and embedded report viewing.

The Monte Carlo batch runs on a worker QThread so the window stays
responsive.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

INK = "#0B1D3A"
ACCENT = "#D9A441"
PAPER = "#F7F6F2"


class BatchWorker(QThread):
    finished_ok = Signal(str, str)   # (report path, summary line)
    failed = Signal(str)

    def __init__(self, crew: str, scenario: str, archetypes: str,
                 runs: int, seed: int):
        super().__init__()
        self.crew, self.scenario, self.archetypes = crew, scenario, archetypes
        self.runs, self.seed = runs, seed

    def run(self) -> None:  # executes on the worker thread
        try:
            from ..report.builder import build_report
            from ..sim.montecarlo import run_batch, save_batch

            result = run_batch(self.crew, self.scenario, self.archetypes,
                               n_runs=self.runs, seed=self.seed)
            run_dir = save_batch(result, Path("reports"))
            report = build_report(result, run_dir)
            s = result.summary()
            line = (f"Cascade-free {100 * s['p_cascade_free']:.1f}%  ·  "
                    f"completion {100 * s['completion_rate']['mean']:.1f}%  ·  "
                    f"cohesion {s['final_cohesion']['mean']:.2f}")
            self.finished_ok.emit(str(report), line)
        except Exception as exc:  # surfaced to the user, not swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NFP Assessor")
        self.setMinimumWidth(560)
        self.worker: BatchWorker | None = None
        self.last_report: str | None = None

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("New Frontier Project — Assessor")
        f = QFont()
        f.setPointSize(15)
        title.setFont(f)
        subtitle = QLabel("Bayesian candidate simulation · mission Monte Carlo")
        subtitle.setStyleSheet(f"color: {ACCENT};")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setVerticalSpacing(8)

        self.crew_edit = QLineEdit("data/crew/sample_crew.json")
        self.scenario_edit = QLineEdit("scenarios/shakedown_14d.json")
        self.arch_edit = QLineEdit("data/archetypes/archetypes_sample.json")
        for row, (label, edit) in enumerate([
            ("Crew file", self.crew_edit),
            ("Scenario", self.scenario_edit),
            ("Archetypes", self.arch_edit),
        ]):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            btn = QPushButton("Browse…")
            btn.clicked.connect(lambda _=False, e=edit: self.browse(e))
            grid.addWidget(btn, row, 2)

        grid.addWidget(QLabel("Replications"), 3, 0)
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(10, 20000)
        self.runs_spin.setValue(200)
        grid.addWidget(self.runs_spin, 3, 1)

        grid.addWidget(QLabel("Seed"), 4, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        grid.addWidget(self.seed_spin, 4, 1)

        layout.addLayout(grid)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("Run simulation batch")
        self.run_btn.clicked.connect(self.start_batch)
        self.open_btn = QPushButton("Open last report")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_report)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.open_btn)
        layout.addLayout(buttons)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.status, stretch=1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            f"QMainWindow {{ background: {PAPER}; }}"
            f"QLabel {{ color: {INK}; }}"
            f"QPushButton {{ padding: 6px 14px; }}"
        )

    def browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", str(Path.cwd()),
            "Data files (*.json *.csv);;All files (*)")
        if path:
            edit.setText(path)

    def start_batch(self) -> None:
        self.run_btn.setEnabled(False)
        self.status.setText(
            f"Running {self.runs_spin.value()} replications… "
            "(window stays responsive; this usually takes a few seconds)")
        self.worker = BatchWorker(
            self.crew_edit.text(), self.scenario_edit.text(),
            self.arch_edit.text(), self.runs_spin.value(),
            self.seed_spin.value())
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

    def on_done(self, report_path: str, summary_line: str) -> None:
        self.last_report = report_path
        self.run_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.status.setText(f"Done. {summary_line}\nReport: {report_path}")

    def on_fail(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText("Batch failed.")
        QMessageBox.critical(self, "Simulation error", message)

    def open_report(self) -> None:
        if self.last_report:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(self.last_report).resolve())))


def launch() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
