"""widgets.py -- small widgets shared by the viewer and scan-core's live result pane."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import view as V
from ..segments import Segments
from .theme import C


class DimRow(QtWidgets.QWidget):
    """One leftover dimension: hold it somewhere, or average over it.

    A dimension that goes UP AND DOWN (a hysteresis loop; segments.py) is held
    by POINT INDEX like any other, but its label names the run ("50 mT ·
    down"), it can be averaged over ONE run ("mean of run"), and an average
    across runs says that it mixes the branches -- never silently.
    """

    changed = QtCore.Signal()

    MODES = ("at", "mean all", "mean of")
    RUN_MODE = "mean of run"

    def __init__(self, dim: str, coords: np.ndarray, unit: str = "",
                 segments: Segments | None = None):
        super().__init__()
        self.dim = dim
        self.coords = np.asarray(coords)
        self.unit = unit
        self.n = max(1, len(self.coords))
        self.segs = segments if (segments is not None
                                 and segments.seg.size == len(self.coords)) else None

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        name = QtWidgets.QLabel(dim)
        name.setMinimumWidth(78)
        name.setMaximumWidth(78)
        name.setToolTip(f"{dim} — {self.n} points")
        name.setStyleSheet(f"color:{C['text']};")
        h.addWidget(name)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(self.modes())
        self.mode.setFixedWidth(104 if self.segs else 84)
        tip = ("at        — show this one slice of the cube\n"
               "mean all  — average the whole axis away\n"
               "mean of   — average a range of it")
        if self.segs:
            tip += ("\nmean of run — average ONE run of it (this axis goes up and\n"
                    "              down: the same value is measured on several runs,\n"
                    "              and an average across them mixes the branches)")
        self.mode.setToolTip(tip)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        h.addWidget(self.mode)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.n - 1)
        self.slider.setPageStep(max(1, self.n // 10))
        self.slider.setMinimumWidth(160)
        self.slider.setMaximumWidth(420)
        self.slider.valueChanged.connect(self._emit)
        h.addWidget(self.slider, 1)

        self.lo = QtWidgets.QSpinBox(); self.lo.setRange(0, self.n - 1)
        self.hi = QtWidgets.QSpinBox(); self.hi.setRange(0, self.n - 1)
        self.hi.setValue(self.n - 1)
        for sb in (self.lo, self.hi):
            sb.setFixedWidth(56)
            sb.setPrefix("#")
            sb.valueChanged.connect(self._emit)
            sb.setVisible(False)
        h.addWidget(self.lo); h.addWidget(self.hi)

        # the run to average over (only for an axis that goes up and down)
        self.run = QtWidgets.QComboBox()
        if self.segs:
            self.run.addItems([self.segs.name(k) for k in range(self.segs.n)])
        self.run.setVisible(False)
        self.run.currentIndexChanged.connect(self._emit)
        h.addWidget(self.run)

        self.value = QtWidgets.QLabel("")
        self.value.setMinimumWidth(124)
        self.value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.value.setStyleSheet(f"color:{C['accent']};")
        h.addWidget(self.value)
        h.addStretch(1)

        self._sync()

    # ---- state ------------------------------------------------------------
    def modes(self) -> tuple:
        return self.MODES + ((self.RUN_MODE,) if self.segs else ())

    def slice_(self) -> V.Slice:
        m = self.mode.currentText()
        if m == "at":
            return V.Slice("at", self.slider.value())
        if m == "mean all":
            return V.Slice("mean", 0, self.n - 1)
        if m == self.RUN_MODE and self.segs:
            # a run is a contiguous block of the axis: its first..last index
            return V.Slice("mean", *self.segs.span(max(0, self.run.currentIndex())))
        return V.Slice("mean", self.lo.value(), self.hi.value())

    def state(self) -> tuple:
        """Everything worth restoring when the same scan is redrawn:
        (mode, at, lo, hi), plus the run averaged over -- ONLY for an axis that
        goes up and down. An ordinary row keeps its 4-tuple exactly: scan-core
        sends it to other PCs as a view state and checks its length."""
        st = (self.mode.currentText(), self.slider.value(), self.lo.value(), self.hi.value())
        return st + (self.run.currentIndex(),) if self.segs else st

    def restore(self, st: tuple) -> None:
        mode, at, lo, hi = st[:4]
        run = st[4] if len(st) > 4 else 0
        for w in (self.mode, self.slider, self.lo, self.hi, self.run):
            w.blockSignals(True)
        if mode in self.modes():
            self.mode.setCurrentText(mode)
        self.slider.setValue(min(at, self.n - 1))
        self.lo.setValue(min(lo, self.n - 1))
        self.hi.setValue(min(hi, self.n - 1))
        if 0 <= run < self.run.count():
            self.run.setCurrentIndex(run)
        for w in (self.mode, self.slider, self.lo, self.hi, self.run):
            w.blockSignals(False)
        self._sync()

    def set_axis(self, coords: np.ndarray, unit: str = "") -> None:
        """New coordinate VALUES for an axis of the same length (another file of
        the same shape). Only the labels change; the operator's choice (the
        index held, the range averaged) stays. Without this the row went on
        showing the previous file's values, e.g. "1 GHz" over a 6 GHz slice."""
        self.coords = np.asarray(coords)
        self.unit = unit
        self._sync()

    # ---- internals --------------------------------------------------------
    def _mode_changed(self):
        self._sync()
        self.changed.emit()

    def _emit(self):
        self._sync()
        self.changed.emit()

    def _text(self, i: int) -> str:
        try:
            t = f"{float(self.coords[i]):g} {self.unit}".strip()
        except Exception:
            return f"#{i}"
        if self.segs is not None and 0 <= i < self.segs.seg.size:
            t = f"{t} · {self.segs.name_at(i)}"       # 50 mT is on two runs
        return t

    def _mixed(self, a: int, b: int) -> bool:
        """Does the index range a..b take points from more than one run?"""
        return self.segs is not None and len(set(self.segs.seg[a:b + 1].tolist())) > 1

    def _sync(self):
        m = self.mode.currentText()
        self.slider.setVisible(m == "at")
        self.lo.setVisible(m == "mean of")
        self.hi.setVisible(m == "mean of")
        self.run.setVisible(m == self.RUN_MODE)
        mixed = False
        if m == "at":
            self.value.setText(self._text(self.slider.value()))
        elif m == "mean all":
            mixed = self._mixed(0, self.n - 1)
            self.value.setText(f"all {self.n}" + (" · up and down mixed" if mixed else ""))
        elif m == self.RUN_MODE and self.segs:
            a, b = self.segs.span(max(0, self.run.currentIndex()))
            lo, hi = sorted((float(self.coords[a]), float(self.coords[b])))
            self.value.setText(f"{b - a + 1} points, {lo:g} … {hi:g} {self.unit}".strip())
        else:
            a, b = sorted((self.lo.value(), self.hi.value()))
            mixed = self._mixed(a, b)
            self.value.setText(f"{self._text(a)} … {self._text(b)}"
                               + (" · mixed" if mixed else ""))
        # an average across a turning point is legal but rarely meant: say so
        self.value.setStyleSheet(f"color:{C['danger' if mixed else 'accent']};")
        self.value.setToolTip("This average mixes runs that go up and runs that go down\n"
                              "(both branches of the loop); 'mean of run' averages one run."
                              if mixed else "")
