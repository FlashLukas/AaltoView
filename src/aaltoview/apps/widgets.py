"""widgets.py -- small widgets shared by the viewer and scan-core's live result pane."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import view as V
from .theme import C


class DimRow(QtWidgets.QWidget):
    """One leftover dimension: hold it somewhere, or average over it."""

    changed = QtCore.Signal()

    MODES = ("at", "mean all", "mean of")

    def __init__(self, dim: str, coords: np.ndarray, unit: str = ""):
        super().__init__()
        self.dim = dim
        self.coords = np.asarray(coords)
        self.unit = unit
        self.n = max(1, len(self.coords))

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
        self.mode.addItems(self.MODES)
        self.mode.setFixedWidth(84)
        self.mode.setToolTip(
            "at        — show this one slice of the cube\n"
            "mean all  — average the whole axis away\n"
            "mean of   — average a range of it")
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

        self.value = QtWidgets.QLabel("")
        self.value.setMinimumWidth(124)
        self.value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.value.setStyleSheet(f"color:{C['accent']};")
        h.addWidget(self.value)
        h.addStretch(1)

        self._sync()

    # ---- state ------------------------------------------------------------
    def slice_(self) -> V.Slice:
        m = self.mode.currentText()
        if m == "at":
            return V.Slice("at", self.slider.value())
        if m == "mean all":
            return V.Slice("mean", 0, self.n - 1)
        return V.Slice("mean", self.lo.value(), self.hi.value())

    def state(self) -> tuple:
        """Everything worth restoring when the same scan is redrawn."""
        return (self.mode.currentText(), self.slider.value(),
                self.lo.value(), self.hi.value())

    def restore(self, st: tuple) -> None:
        mode, at, lo, hi = st
        for w in (self.mode, self.slider, self.lo, self.hi):
            w.blockSignals(True)
        if mode in self.MODES:
            self.mode.setCurrentText(mode)
        self.slider.setValue(min(at, self.n - 1))
        self.lo.setValue(min(lo, self.n - 1))
        self.hi.setValue(min(hi, self.n - 1))
        for w in (self.mode, self.slider, self.lo, self.hi):
            w.blockSignals(False)
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
            return f"{float(self.coords[i]):g} {self.unit}".strip()
        except Exception:
            return f"#{i}"

    def _sync(self):
        m = self.mode.currentText()
        self.slider.setVisible(m == "at")
        self.lo.setVisible(m == "mean of")
        self.hi.setVisible(m == "mean of")
        if m == "at":
            self.value.setText(self._text(self.slider.value()))
        elif m == "mean all":
            self.value.setText(f"all {self.n}")
        else:
            a, b = sorted((self.lo.value(), self.hi.value()))
            self.value.setText(f"{self._text(a)} … {self._text(b)}")
