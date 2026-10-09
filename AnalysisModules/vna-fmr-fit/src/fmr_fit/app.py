"""app.py -- the VNA-FMR fit window. The maths is in model.py; this is the widgets.

    +-----------------------+------------------------------------------------+
    | CURVES (from the      |  data + fit  (drag the shaded band = range)    |
    |  viewer) + fit result |                                                |
    |                       |  residuals                                     |
    | MODEL peaks / fit /   +------------------------------------------------+
    |  background / hand    |  RESULTS: one row per fitted curve             |
    | PARAMETERS            |  Copy | Save… | Send to Origin | Save image…   |
    | Guess | Fit | Fit all |                                                |
    +-----------------------+------------------------------------------------+

Every curve keeps its own model, start values and result: switch between them
freely; "Fit all" fits every curve with the model of the one on screen, each
from its own guessed start.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from aaltoview.apps.theme import C
from aaltoview.apps.viewer import TAB10, _float, _plain_axes
from aaltoview.export import (Curve, MapData, axis_title, map_rows, map_to_curves,
                              pick_rows, save_figure, transpose_map)

from . import model as M
from .dispersion import unit_kind
from .dispersion_tab import DispersionTab

#: what a range drawn on one curve does to the others
#: the settings that go from one curve to the next (the range has its own rule)
CARRIED = ("n_peaks", "mode", "baseline", "hand", "lineshape", "delay", "dd")
RANGE_RULES = {"follow": "follows the peak", "same": "same values", "off": "not carried"}
MODE_TEXT = {"complex": "Re + Im together", "real": "one channel (as shown)"}
HAND_TEXT = {0: "auto", 1: "+1", -1: "−1"}
PARAM_TEXT = {"center": "resonance x0", "hwhm": "HWHM Δ", "amp": "amplitude A",
              "phase": "mixing phase φ"}


@dataclass(eq=False)          # compared by identity: it holds arrays
class Entry:
    """One curve and everything the window knows about fitting it."""
    curve: Curve
    setup: M.Setup
    start: dict[str, M.Spec] | None = None
    result: M.Result | None = None
    own_range: bool = False             # the operator dragged this curve's band
    located: tuple | None = None        # (key, (x0, hwhm)): M.locate is a fit, done once
    reference: tuple | None = None      # (Entry, "divide" | "subtract"): taken out first
    touched: bool = False               # the operator set this curve up or fitted it
    predicted: bool = False             # fitted from the Dispersion tab's prediction

    def y(self) -> np.ndarray:
        """What is fitted: the complex values or the shown channel, with the
        reference sweep divided or subtracted out when one is chosen."""
        complex_ = self.setup.mode == "complex"
        y = self.curve.z if complex_ else self.curve.y
        if self.reference is None:
            return y
        ref, how = self.reference
        if complex_ and ref.curve.z is None:
            raise ValueError(f"the reference '{ref.curve.label}' has no complex data")
        yr = ref.curve.z if complex_ else ref.curve.y
        return M.reference(self.curve.x, y, ref.curve.x, yr, how)


#: at most this many symbols across the plot: with more points in view, every
#: 2nd, 3rd, ... is drawn (asked for 2026-09-30). All 40 000 symbols of a YIG
#: sweep (Re + Im + residuals) took 0.4 s per redraw, and 10 000 on 900 pixels
#: are a smear anyway. Zooming in brings them all back.
MAX_SYMBOLS = 600


def _points(x, y, color: str, size: int, name: str | None = None) -> pg.PlotDataItem:
    """Measured points, as symbols (thinned to the view by FitWindow._thin)."""
    col = QtGui.QColor(color)
    item = pg.PlotDataItem(x, y, pen=None, symbol="o", symbolSize=size, symbolPen=None,
                           symbolBrush=pg.mkBrush(col), name=name)
    item.thinned = True
    return item


def thin_step(x, lo: float, hi: float, most: int = MAX_SYMBOLS) -> int:
    """Draw every k-th point so that no more than `most` fall between lo and hi."""
    x = np.asarray(x, dtype=float)
    visible = int(np.count_nonzero((x >= lo) & (x <= hi)))
    return max(1, int(np.ceil(visible / most)))


def _pretty(name: str) -> str:
    """p1_hwhm -> 'peak 1: HWHM Δ'; bg_re -> 'background Re'."""
    if name.startswith("p") and "_" in name:
        k, p = name[1:].split("_", 1)
        return f"peak {k}: {PARAM_TEXT.get(p, p)}"
    return {"bg": "background", "bg_re": "background Re", "bg_im": "background Im",
            "slope": "slope", "slope_re": "slope Re", "slope_im": "slope Im",
            "delay": "electrical delay τ"}.get(name, name)


#: more sweeps than this from one map: ask before taking them all (a map of
#: 8192 frequencies sent with field on Y is 8192 field sweeps, ~35 min of Fit all)
MANY_SWEEPS = 200


class FewerSweeps(QtWidgets.QDialog):
    """'This map gives 8192 sweeps -- take fewer?' A range of the map's X (the
    values the sweeps are taken at) and how many, spread evenly over it."""

    def __init__(self, parent, t: MapData):
        super().__init__(parent)
        self.t = t
        n, npts = t.y.size, t.x.size
        self.setWindowTitle("Many sweeps")
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            f"This map gives {n} sweeps along {t.x_name} (one per {t.y_name}), "
            f"{npts} points each.\nThat is a lot to fit: about {n * 0.25 / 60:.0f} min "
            f"for Fit all. Take fewer?"))
        g = QtWidgets.QGridLayout()
        y = np.asarray(t.y, dtype=float)
        unit = f" {t.y_unit}" if t.y_unit else ""
        g.addWidget(QtWidgets.QLabel(f"{t.y_name} from"), 0, 0)
        self.lo = QtWidgets.QLineEdit(f"{np.nanmin(y):g}")
        self.hi = QtWidgets.QLineEdit(f"{np.nanmax(y):g}")
        g.addWidget(self.lo, 0, 1)
        g.addWidget(QtWidgets.QLabel("to"), 0, 2)
        g.addWidget(self.hi, 0, 3)
        g.addWidget(QtWidgets.QLabel(unit.strip()), 0, 4)
        g.addWidget(QtWidgets.QLabel("number of sweeps"), 1, 0)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, n)
        self.count.setValue(min(50, n))
        g.addWidget(self.count, 1, 1)
        v.addLayout(g)
        self.info = QtWidgets.QLabel()
        v.addWidget(self.info)
        b = QtWidgets.QDialogButtonBox()
        self.take = b.addButton("Take these", QtWidgets.QDialogButtonBox.AcceptRole)
        self.all = b.addButton(f"Take all {n}", QtWidgets.QDialogButtonBox.AcceptRole)
        b.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.take.clicked.connect(lambda: self._done(False))
        self.all.clicked.connect(lambda: self._done(True))
        b.rejected.connect(self.reject)
        v.addWidget(b)
        self._all = False
        for w in (self.lo, self.hi):
            w.textChanged.connect(self._update)
        self.count.valueChanged.connect(self._update)
        self._update()

    def _range(self):
        return (_float(self.lo.text(), float(np.nanmin(self.t.y))),
                _float(self.hi.text(), float(np.nanmax(self.t.y))))

    def _update(self, *_):
        r = pick_rows(self.t.y, self.count.value(), *self._range())
        if r.size:
            ys = np.asarray(self.t.y)[r]
            self.info.setText(f"-> {r.size} sweeps, at {self.t.y_name} = {ys.min():g} ... "
                              f"{ys.max():g} {self.t.y_unit}".rstrip())
        else:
            self.info.setText("-> nothing in that range")
        self.take.setEnabled(r.size > 0)

    def _done(self, everything: bool):
        self._all = everything
        self.accept()

    def rows(self):
        if self._all:
            return np.arange(self.t.y.size)
        return pick_rows(self.t.y, self.count.value(), *self._range())


class FitWindow(QtWidgets.QWidget):
    COLS = ("Parameter", "Value", "±", "Unit", "Fixed", "Min", "Max")

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.resize(1500, 920)
        self.entries: list[Entry] = []
        self._filling = False
        self._origin_proc: QtCore.QProcess | None = None
        self.last_dir: Path | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        title = QtWidgets.QLabel("VNA-FMR FIT"); title.setObjectName("title")
        outer.addWidget(title)
        # two stages: each curve's resonance, then all resonances together
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, 1)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.tabs.addTab(split, "Resonances")

        # ── left: curves, model, parameters ────────────────────────────────
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(6)
        lv.addWidget(self._tag("CURVES"))
        self.curve_list = QtWidgets.QTreeWidget()
        self.curve_list.setHeaderLabels(["Curve", "Fit"])
        self.curve_list.setRootIsDecorated(False)
        self.curve_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.curve_list.currentItemChanged.connect(self._curve_changed)
        self.curve_list.itemSelectionChanged.connect(self._selection_changed)
        self.curve_list.setToolTip("Send curves here from AaltoView: 1D plots -> Analysis")
        lv.addWidget(self.curve_list, 2)
        rb = QtWidgets.QHBoxLayout()
        rm = QtWidgets.QPushButton("Remove"); rm.clicked.connect(self.remove_selected)
        clr = QtWidgets.QPushButton("Clear all"); clr.setObjectName("danger")
        clr.clicked.connect(self.clear)
        rb.addWidget(rm); rb.addWidget(clr); rb.addStretch(1)
        lv.addLayout(rb)

        lv.addWidget(self._tag("MODEL"))
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("peaks"), 0, 0)
        self.peaks = QtWidgets.QSpinBox(); self.peaks.setRange(1, 6)
        self.peaks.setToolTip("Resonances in the range (Kittel + standing spin waves ...).\n"
                              "A new peak is guessed from what the others leave.")
        self.peaks.valueChanged.connect(self._model_changed)
        grid.addWidget(self.peaks, 0, 1)
        grid.addWidget(QtWidgets.QLabel("fit"), 0, 2)
        self.mode = QtWidgets.QComboBox()
        for k, t in MODE_TEXT.items():
            self.mode.addItem(t, k)
        self.mode.setToolTip(
            "Re + Im together: both quadratures of the complex signal -- the most\n"
            "reliable position and width.\n"
            "One channel: the part AaltoView showed (Re, Im or |z|). |z| is only a\n"
            "mixed Lorentzian when the background is much larger than the peak.")
        self.mode.currentIndexChanged.connect(self._model_changed)
        grid.addWidget(self.mode, 0, 3)
        grid.addWidget(QtWidgets.QLabel("background"), 1, 0)
        self.baseline = QtWidgets.QComboBox(); self.baseline.addItems(list(M.BASELINES))
        self.baseline.setCurrentText("linear")
        self.baseline.currentIndexChanged.connect(self._model_changed)
        grid.addWidget(self.baseline, 1, 1)
        grid.addWidget(QtWidgets.QLabel("hand"), 1, 2)
        self.hand = QtWidgets.QComboBox()
        for k, t in HAND_TEXT.items():
            self.hand.addItem(t, k)
        self.hand.setToolTip("Which way the complex signal turns through the resonance\n"
                             "(depends on sweep direction and instrument convention).\n"
                             "auto: fit both, keep the better.")
        self.hand.currentIndexChanged.connect(self._model_changed)
        grid.addWidget(self.hand, 1, 3)
        lv.addLayout(grid)

        # ── options for frequency sweeps (a VNA is harder to fit in f) ────
        lv.addWidget(self._tag("FREQUENCY SWEEPS  (options; see the tooltips)"))
        og = QtWidgets.QGridLayout()
        og.addWidget(QtWidgets.QLabel("lineshape"), 0, 0)
        self.lineshape = QtWidgets.QComboBox()
        self.lineshape.addItem("Lorentzian", "lorentzian")
        self.lineshape.addItem("oscillator (exact in f)", "oscillator")
        self.lineshape.setToolTip(
            "Oscillator: 2 f0 Δ / (f0² − f² − i f 2Δ), the damped oscillator, exact for a\n"
            "FREQUENCY sweep. The Lorentzian is its limit near resonance: off by ~ Δ/f0\n"
            "(a few % for a broad line at low frequency). Same x0 and HWHM Δ.\n"
            "For field sweeps keep the Lorentzian.")
        self.lineshape.currentIndexChanged.connect(self._model_changed)
        og.addWidget(self.lineshape, 0, 1)
        self.delay = QtWidgets.QCheckBox("electrical delay τ")
        self.delay.setToolTip(
            "Fit the cable's delay: the signal times e^(−i2π τ (f − fc)), the phase that\n"
            "winds with frequency (3 ns = a full turn every 0.33 GHz). τ in ns for GHz.\n"
            "Use it WITH derivative-divide too: then the model is exact for the delay.")
        self.delay.toggled.connect(self._model_changed)
        og.addWidget(self.delay, 0, 2, 1, 2)
        self.dd = QtWidgets.QCheckBox("derivative-divide, step k")
        self.dd.setToolTip(
            "Fit D = (S(f+) − S(f−)) / ((f+ − f−) S(f)), f± = k points either side\n"
            "(Maier-Flaig et al. 2018): a background that MULTIPLIES the signal and\n"
            "varies slowly drops out; the model is transformed exactly the same way,\n"
            "so k costs no accuracy -- choose it about as wide as the line (noise is\n"
            "divided by the step). The amplitude becomes relative to the background.\n"
            "Does NOT remove a standing-wave ripple as large as the resonance: use a\n"
            "reference for that.")
        self.dd.toggled.connect(self._model_changed)
        og.addWidget(self.dd, 1, 0, 1, 2)
        self.dd_k = QtWidgets.QSpinBox(); self.dd_k.setRange(1, 200); self.dd_k.setValue(5)
        self.dd_k.valueChanged.connect(self._model_changed)
        og.addWidget(self.dd_k, 1, 2)
        og.addWidget(QtWidgets.QLabel("reference"), 2, 0)
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.setToolTip(
            "Another received curve -- a sweep where nothing resonates in the band\n"
            "(e.g. at a high field) -- taken out BEFORE fitting: divide for a VNA (the\n"
            "background multiplies), subtract for an additive one. Interpolated onto\n"
            "this curve's axis: record it on the same frequency grid if the phase winds fast.")
        self.ref_combo.currentIndexChanged.connect(self._model_changed)
        og.addWidget(self.ref_combo, 2, 1)
        self.ref_how = QtWidgets.QComboBox(); self.ref_how.addItems(["divide", "subtract"])
        self.ref_how.currentIndexChanged.connect(self._model_changed)
        og.addWidget(self.ref_how, 2, 2)
        lv.addLayout(og)
        self.formula = QtWidgets.QLabel(
            "S = Σ A·e^{iφ}·Δ/(x0 − x − iΔ) + b0 + b1·(x − xc)     Δ = HWHM, FWHM = 2Δ"
            "     ·  Ctrl+click on the plot: a peak there")
        self.formula.setStyleSheet(f"color:{C['muted']}; font-size:11px;")
        self.formula.setWordWrap(True)
        lv.addWidget(self.formula)

        lv.addWidget(self._tag("PARAMETERS"))
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i in range(1, len(self.COLS)):
            hh.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setToolTip("Type a start value, tick Fixed to hold it, set Min/Max to "
                              "bound it.\nThen Fit.")
        self.table.itemChanged.connect(self._param_edited)
        lv.addWidget(self.table, 3)
        fb = QtWidgets.QHBoxLayout()
        self.guess_btn = QtWidgets.QPushButton("Guess")
        self.guess_btn.setToolTip("Fresh start values from the data (forgets your edits).")
        self.guess_btn.clicked.connect(self.guess)
        self.fit_btn = QtWidgets.QPushButton("Fit"); self.fit_btn.setObjectName("primary")
        self.fit_btn.clicked.connect(self.fit)
        self.fit_all_btn = QtWidgets.QPushButton("Fit all")
        self.fit_all_btn.setToolTip("With several curves selected (Ctrl/Shift-click): only "
                                    "those.\nOtherwise every curve. THIS model; each curve "
                                    "from its own guessed start.")
        self.fit_all_btn.clicked.connect(self.fit_all)
        self.full_btn = QtWidgets.QPushButton("Full range")
        self.full_btn.setToolTip("This curve, and the range carried to the others, back to "
                                 "the whole sweep.")
        self.full_btn.clicked.connect(self.full_range)
        for b in (self.guess_btn, self.fit_btn, self.fit_all_btn, self.full_btn):
            fb.addWidget(b)
        lv.addLayout(fb)
        rr = QtWidgets.QHBoxLayout()
        rr.addWidget(QtWidgets.QLabel("range for the other curves"))
        self.rule_combo = QtWidgets.QComboBox()
        for k, t in RANGE_RULES.items():
            self.rule_combo.addItem(t, k)
        self.rule_combo.setToolTip(
            "Drag the shaded band on the plot to fit only part of the sweep. Then:\n"
            "follows the peak: every other curve gets the same window AROUND ITS OWN\n"
            "  resonance, measured in linewidths (e.g. x0 - 9 HWHM ... x0 + 11 HWHM) --\n"
            "  right when the resonance moves and broadens with frequency;\n"
            "same values: the same field (or frequency) values on every curve;\n"
            "not carried: the other curves keep the whole sweep.\n"
            "A curve whose band you drag yourself keeps its own.")
        self.rule_combo.currentIndexChanged.connect(self._rule_changed)
        rr.addWidget(self.rule_combo, 1)
        lv.addLayout(rr)
        #: the last band the operator drew: {source, lo, hi (in HWHM from x0), xmax, xmin}
        self.range_rule: dict | None = None
        #: the settings of the curve last set up or fitted, for the next ones:
        #: ({CARRIED: value}, reference) -- None until the operator does something
        self.template: tuple | None = None
        split.addWidget(left)

        # ── right: plots, results ──────────────────────────────────────────
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        rv.setSpacing(6)
        self.two_axes = QtWidgets.QCheckBox("Im on its own axis (right)")
        self.two_axes.setChecked(True)
        self.two_axes.setToolTip(
            "Re on the left axis, Im on the right, each scaled to itself: a VNA's Re\n"
            "sits near 1 and Im near 0, and on one shared scale the resonance is a few\n"
            "pixels. Drag or scroll on the right axis to scale Im alone.")
        self.two_axes.toggled.connect(lambda *_: self._redraw())
        rv.addWidget(self.two_axes)
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        # the right axis (asked for 2026-10-08): a second view, x linked, drawn
        # BEHIND the main one so the range band, Ctrl+click and zoom keep working
        self.vb2 = pg.ViewBox()
        self.vb2.setZValue(-100)
        self.plot.scene().addItem(self.vb2)
        self.plot.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(self.plot)
        self.plot.vb.sigResized.connect(
            lambda *_: self.vb2.setGeometry(self.plot.vb.sceneBoundingRect()))
        self.rplot = self.glw.addPlot(row=1, col=0)
        self.rplot.setXLink(self.plot)
        self.rplot.setLabel("left", "residual")
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        _plain_axes(self.plot, self.plot.getAxis("right"))
        _plain_axes(self.rplot)
        self.plot.hideAxis("right")
        for plot in (self.plot, self.rplot):
            # long sweeps: draw what the screen can show, not every point
            plot.setClipToView(True)            # only what is in view is drawn
        band = QtGui.QColor(C["accent"]); band.setAlpha(22)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(band))
        self.region.setZValue(-10)
        self.region.sigRegionChangeFinished.connect(self._range_dragged)
        self.plot.sigXRangeChanged.connect(self._thin)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)
        self.plot.addItem(self.region)
        self._items: list = []
        rv.addWidget(self.glw, 3)

        rv.addWidget(self._tag("RESULTS"))
        self.results = QtWidgets.QTableWidget(0, 0)
        self.results.verticalHeader().setVisible(False)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        rv.addWidget(self.results, 1)
        eb = QtWidgets.QHBoxLayout()
        spec = [("Copy results", self.copy_results,
                 "The results table, tab-separated, onto the clipboard."),
                ("Save results…", self.save_results,
                 ".csv / .dat with name / unit / comment header rows."),
                ("Send to Origin", self.send_to_origin,
                 "The results table into a running Origin (starts one if needed)."),
                ("Save image…", self.save_image,
                 "Data, fit and residuals of the curve on screen: PNG, PDF or SVG.")]
        for text, fn, tip in spec:
            b = QtWidgets.QPushButton(text); b.setToolTip(tip); b.clicked.connect(fn)
            eb.addWidget(b)
        eb.addStretch(1)
        rv.addLayout(eb)
        self.status = QtWidgets.QLabel("Waiting for curves: in AaltoView, 1D plots → Analysis")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([560, 940])
        self._set_enabled(False)
        self.dispersion = DispersionTab(self)
        self.tabs.addTab(self.dispersion, "Dispersion")
        self.tabs.setTabToolTip(1, "Resonance positions and widths of all fitted curves -> "
                                   "γ, M_eff, anisotropy, PSSW exchange, damping")
        self.tabs.currentChanged.connect(self._tab_changed)

    def _tab_changed(self, i: int):
        if self.tabs.widget(i) is self.dispersion:
            self.dispersion.refresh()           # the fits may have changed meanwhile

    @staticmethod
    def _tag(text):
        t = QtWidgets.QLabel(text); t.setObjectName("tag")
        return t

    def say(self, text: str, error: bool = False):
        self.status.setStyleSheet(f"color:{C['danger'] if error else C['muted']};"
                                  " font-size:11px;")
        self.status.setText(text)

    # ── a batch of fits that keeps the window alive ─────────────────────────
    # 174 angle sweeps x 0.24 s = 42 s in one block: Windows called the window
    # "Not Responding" and nothing could stop it (2026-09-30). Now one curve at
    # a time, its row updated, events handled in between, and a Stop button.
    _batch = False

    def _batch_begin(self, stop_buttons):
        self._batch, self._stop = True, False
        self._set_enabled(False)
        self.curve_list.setEnabled(False)       # the list switches the curve on screen
        self._stop_buttons = [(b, b.text(), b.isEnabled()) for b in stop_buttons]
        for b in stop_buttons:
            b.setText("Stop")
            b.setEnabled(True)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)

    def _batch_step(self, k: int, n: int, e: "Entry", what: str = "fitting") -> bool:
        """After curve k of n: its row, the status, the events. False = Stop pressed."""
        self._update_row(e)
        self.say(f"{what} {k} / {n}: {e.curve.label}   (Stop to stop here)")
        QtWidgets.QApplication.processEvents()
        return not self._stop

    def _batch_end(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        for b, text, on in self._stop_buttons:
            b.setText(text)
            b.setEnabled(on)
        self._batch = False
        self.curve_list.setEnabled(True)
        self._set_enabled(True)
        self._selection_changed()

    def request_stop(self) -> bool:
        """Stop pressed (the batch's button): True if a batch was running."""
        if self._batch:
            self._stop = True
            return True
        return False

    def _set_enabled(self, on: bool):
        for w in (self.guess_btn, self.fit_btn, self.fit_all_btn, self.full_btn, self.table,
                  self.peaks, self.mode, self.baseline, self.hand, self.lineshape, self.delay,
                  self.dd, self.dd_k, self.ref_combo, self.ref_how):
            w.setEnabled(on)

    # ── curves in ──────────────────────────────────────────────────────────
    def add_curves(self, curves: list[Curve]):
        first_new = len(self.entries)
        notes = set()
        for c in curves:
            mode = "complex" if c.z is not None else "real"
            setup = M.Setup(mode=mode, baseline="linear")
            if unit_kind(c.x_unit)[0] == "freq":
                # a frequency sweep: the exact lineshape; and the delay on when
                # the cable winds the phase (raw, the data look like noise)
                setup.lineshape = "oscillator"
                notes.add("oscillator lineshape")
                turns = M.phase_turns(c.x, c.z) if c.z is not None else 0.0
                if turns > 2:
                    setup.delay = True
                    notes.add(f"electrical delay on (the phase winds ~{turns:.0f} turns)")
            e = Entry(c, setup)
            self.entries.append(e)
            it = QtWidgets.QTreeWidgetItem([c.label, "—"])
            it.setToolTip(0, f"{c.label}\n{c.source or 'unsaved run'}\n"
                             f"{'complex' if c.z is not None else 'real'}: {c.y_name}")
            self.curve_list.addTopLevelItem(it)
        self.curve_list.resizeColumnToContents(0)
        self.curve_list.setCurrentItem(self.curve_list.topLevelItem(first_new))
        n = len(curves)
        msg = f"received {n} curve{'s' if n != 1 else ''}"
        if notes and self.template is None:
            msg += "; frequency sweeps: " + ", ".join(sorted(notes))
        self.say(msg)

    def add_maps(self, maps: list[MapData]):
        """A whole map from the viewer's Map tab: the map's Y axis becomes the
        fit's x axis (his rule, 2026-10-09) -- one sweep per X value, along Y.
        X = rf_freq, Y = field: field sweeps, one per frequency; X = field,
        Y = rf_freq: frequency sweeps, one per field. More than MANY_SWEEPS
        sweeps: ask first, and offer fewer (a range of X, a number)."""
        curves = []
        for m in maps:
            t = transpose_map(m)                 # rows of t = the map's X values
            if t.y.size > MANY_SWEEPS:
                rows = self.ask_fewer(t)
                if rows is None:
                    self.say(f"map ({m.label}) not taken: cancelled")
                    continue
                t = map_rows(t, rows)
            curves += map_to_curves(t)
        if not curves:
            return
        self.add_curves(curves)
        m = maps[0]
        self.say(f"received a map ({m.label}): {len(curves)} sweeps along {m.y_name} "
                 f"(the map's Y), one per {m.x_name}")

    def ask_fewer(self, t: MapData):
        """Many sweeps: which to take (row indices of t), or None = cancel."""
        dlg = FewerSweeps(self, t)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dlg.rows()

    def targets(self) -> list[Entry]:
        """What Fit all works on: the selected curves when there are several,
        otherwise all of them (one selected = the one on screen = Fit)."""
        rows = sorted({self.curve_list.indexOfTopLevelItem(it)
                       for it in self.curve_list.selectedItems()})
        return [self.entries[i] for i in rows] if len(rows) > 1 else list(self.entries)

    def _selection_changed(self):
        n = len({id(it) for it in self.curve_list.selectedItems()})
        self.fit_all_btn.setText(f"Fit selected ({n})" if n > 1 else "Fit all")

    def current(self) -> Entry | None:
        i = self.curve_list.indexOfTopLevelItem(self.curve_list.currentItem())
        return self.entries[i] if 0 <= i < len(self.entries) else None

    def remove_selected(self):
        idx = sorted({self.curve_list.indexOfTopLevelItem(it)
                      for it in self.curve_list.selectedItems()}, reverse=True)
        for i in idx:
            self.curve_list.takeTopLevelItem(i)
            gone = self.entries.pop(i)
            for other in self.entries:
                if other.reference and other.reference[0] is gone:
                    other.reference, other.result, other.start = None, None, None
        self._curve_changed()
        self._fill_results()

    def clear(self):
        self.curve_list.clear()
        self.entries = []
        self._curve_changed()
        self._fill_results()

    # ── model controls ─────────────────────────────────────────────────────
    def _curve_changed(self, *_):
        e = self.current()
        self._set_enabled(e is not None)
        if e is None:
            self.table.setRowCount(0)
            self._redraw()
            return
        self._take_template(e)
        self._filling = True
        try:
            self.peaks.setValue(e.setup.n_peaks)
            self.mode.setCurrentIndex(self.mode.findData(e.setup.mode))
            self.baseline.setCurrentText(e.setup.baseline)
            self.hand.setCurrentIndex(self.hand.findData(e.setup.hand))
            self.lineshape.setCurrentIndex(self.lineshape.findData(e.setup.lineshape))
            self.delay.setChecked(e.setup.delay)
            self.dd.setChecked(bool(e.setup.dd))
            if e.setup.dd:
                self.dd_k.setValue(e.setup.dd)
            self._fill_ref_combo(e)
            # complex needs complex data
            self.mode.model().item(0).setEnabled(e.curve.z is not None)
            self._complex_only(e.setup.mode == "complex")
        finally:
            self._filling = False
        before = (e.setup.xmin, e.setup.xmax)
        note = self._apply_rule(e)
        # a curve not fitted yet gets start values from the range it will be fitted in
        if e.start is None or (e.result is None and (e.setup.xmin, e.setup.xmax) != before):
            self._guess(e, keep=None, quiet=True)
        self._set_region(e)
        self._fill_table()
        self._redraw()
        if note:
            self.say(note)

    def _setup_from_controls(self, e: Entry) -> M.Setup:
        complex_ = self.mode.currentData() == "complex"
        return M.Setup(n_peaks=self.peaks.value(), mode=self.mode.currentData(),
                       baseline=self.baseline.currentText(), hand=self.hand.currentData(),
                       xmin=e.setup.xmin, xmax=e.setup.xmax,
                       lineshape=self.lineshape.currentData(),
                       delay=complex_ and self.delay.isChecked(),
                       dd=self.dd_k.value() if (complex_ and self.dd.isChecked()) else 0)

    def _remember(self, e: Entry):
        """What this curve is set to becomes the settings for the next ones."""
        e.touched = True
        self.template = ({k: getattr(e.setup, k) for k in CARRIED}, e.reference)

    def _take_template(self, e: Entry):
        """A curve not set up or fitted yet takes the last settings (asked for
        2026-09-29: fit one sweep, go to the next, same settings)."""
        if self.template is None or e.touched or e.result is not None:
            return
        opts, ref = self.template
        if opts["mode"] == "complex" and e.curve.z is None:
            opts = {**opts, "mode": "real", "delay": False, "dd": 0}
        new = M.Setup(**{**e.setup.__dict__, **opts})
        new_ref = ref if (ref is None or ref[0] is not e) else None
        if new != e.setup or new_ref != e.reference:
            e.setup, e.reference = new, new_ref
            e.start, e.located = None, None     # guessed again with these settings

    def _complex_only(self, on: bool):
        """Hand, delay and derivative-divide need both quadratures."""
        for w in (self.hand, self.delay, self.dd, self.dd_k):
            w.setEnabled(on)

    def _fill_ref_combo(self, e: Entry):
        """none + every OTHER curve; the current choice kept."""
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItem("none", None)
        for other in self.entries:
            if other is not e:
                self.ref_combo.addItem(other.curve.label, id(other))
        i = self.ref_combo.findData(id(e.reference[0])) if e.reference else 0
        self.ref_combo.setCurrentIndex(max(i, 0))
        if e.reference:
            self.ref_how.setCurrentText(e.reference[1])
        self.ref_combo.blockSignals(False)

    def _reference_from_controls(self, e: Entry):
        rid = self.ref_combo.currentData()
        ref = next((o for o in self.entries if id(o) == rid), None)
        return (ref, self.ref_how.currentText()) if ref is not None else None

    def _model_changed(self, *_):
        e = self.current()
        if self._filling or e is None:
            return
        old, old_ref = e.setup, e.reference
        e.setup = self._setup_from_controls(e)
        e.reference = self._reference_from_controls(e)
        e.located = None                        # the data may have changed
        self._complex_only(e.setup.mode == "complex")
        e.result = None
        # Keep the peaks already tuned -- unless what the amplitude and phase MEAN
        # changed: derivative-divide makes them relative, a delay or a reference
        # changes the background they are measured against. Kept across such a
        # change they were a wrong start (7.40 GHz instead of 7.06, 2026-09-29).
        same = (all(getattr(old, k) == getattr(e.setup, k)
                    for k in ("mode", "lineshape", "delay", "dd"))
                and old_ref == e.reference)
        keep = {k: v for k, v in (e.start or {}).items() if k.startswith("p")} if same else {}
        self._guess(e, keep=keep or None)
        self._remember(e)
        self._fill_table()
        self._redraw()

    def _set_region(self, e: Entry):
        x = e.curve.x[np.isfinite(e.curve.x)]
        lo = e.setup.xmin if e.setup.xmin is not None else float(np.min(x))
        hi = e.setup.xmax if e.setup.xmax is not None else float(np.max(x))
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)

    def _range_dragged(self):
        e = self.current()
        if e is None:
            return
        lo, hi = self.region.getRegion()
        x = e.curve.x[np.isfinite(e.curve.x)]
        e.setup.xmin = None if lo <= np.min(x) else float(lo)
        e.setup.xmax = None if hi >= np.max(x) else float(hi)
        e.own_range = True
        self.range_rule = {"source": e.curve.label, "xmin": float(lo), "xmax": float(hi),
                           "lo": None, "hi": None}
        try:
            x0, w = self._locate(e)
            self.range_rule["lo"], self.range_rule["hi"] = M.range_in_widths(x0, w, lo, hi)
            peak = (f"; around the peak: x0 {self.range_rule['lo']:+.1f} … "
                    f"{self.range_rule['hi']:+.1f} HWHM")
        except Exception:
            peak = ""
        self.say(f"fit range {lo:.6g} … {hi:.6g} {e.curve.x_unit}{peak}  "
                 "(Guess or Fit to use it; the other curves get it too)")

    def full_range(self):
        e = self.current()
        if e is None:
            return
        e.setup.xmin = e.setup.xmax = None
        e.own_range = False
        self.range_rule = None
        for other in self.entries:
            if not other.own_range:
                other.setup.xmin = other.setup.xmax = None
        self._set_region(e)
        self.say("whole sweep, for this curve and for the ones that took the carried range")

    def _rule_changed(self, *_):
        e = self.current()
        if e is not None and not e.own_range:
            self._curve_changed()

    def _locate(self, e: Entry) -> tuple[float, float]:
        """(x0, HWHM) of the main peak over the whole sweep, cached per model."""
        key = (e.setup.mode, e.setup.baseline, e.setup.hand)
        if e.located is None or e.located[0] != key:
            e.located = (key, M.locate(e.curve.x, e.y(), e.setup))
        return e.located[1]

    def _apply_rule(self, e: Entry) -> str:
        """Give a curve without its own band the carried range. Returns what was
        done, for the status line ("" when nothing was)."""
        if e.own_range:
            return ""
        rule, kind = self.range_rule, self.rule_combo.currentData()
        if rule is None or kind == "off":
            e.setup.xmin = e.setup.xmax = None
            return ""
        if kind == "same" or rule["lo"] is None:
            e.setup.xmin, e.setup.xmax = rule["xmin"], rule["xmax"]
            return f"range from '{rule['source']}': same values"
        try:
            x0, w = self._locate(e)
            xs = e.curve.x[np.isfinite(e.curve.x)]
            e.setup.xmin = max(x0 + rule["lo"] * w, float(xs.min()))
            e.setup.xmax = min(x0 + rule["hi"] * w, float(xs.max()))
        except Exception as exc:
            e.setup.xmin = e.setup.xmax = None
            return f"could not place the carried range ({exc}); using the whole sweep"
        return (f"range from '{rule['source']}', following the peak: x0 {rule['lo']:+.1f} … "
                f"{rule['hi']:+.1f} HWHM = {e.setup.xmin:.4g} … {e.setup.xmax:.4g} "
                f"{e.curve.x_unit}")

    # ── parameters ─────────────────────────────────────────────────────────
    def _plot_clicked(self, ev):
        """Ctrl+click on the plot: a peak there."""
        if not (ev.modifiers() & QtCore.Qt.ControlModifier):
            return
        if not self.plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.add_peak_at(self.plot.vb.mapSceneToView(ev.scenePos()).x())

    def add_peak_at(self, x: float):
        """One more peak, put at x (the others kept as they are)."""
        e = self.current()
        if e is None:
            return
        if e.setup.n_peaks >= self.peaks.maximum():
            self.say(f"at most {self.peaks.maximum()} peaks", error=True)
            return
        keep = {n: s for n, s in (e.start or {}).items() if n.startswith("p")}
        widths = [s.value for n, s in keep.items() if n.endswith("_hwhm")]
        k = e.setup.n_peaks + 1
        e.setup = M.Setup(**{**e.setup.__dict__, "n_peaks": k})
        self._filling = True
        try:
            self.peaks.setValue(k)
        finally:
            self._filling = False
        e.result = None
        if self._guess(e, keep=keep or None,
                       at={k: (float(x), float(np.median(widths)) if widths else None)}):
            self.say(f"peak {k} put at {x:.6g} {e.curve.x_unit} -- Fit to fit it")
        self._remember(e)
        self._fill_table()
        self._redraw()

    def _guess(self, e: Entry, keep, quiet=False, at=None) -> bool:
        try:
            y = e.y()
            hand = e.setup.hand
            if not hand and e.setup.mode == "complex":
                # as fit() does -- for ONE peak too: peaks kept when more are
                # added must belong to the same hand as the new ones (a start
                # mixing both made every fit from it fail, 2026-09-29)
                hand = M._pick_hand(e.curve.x, y, e.setup)
            e.start = M.guess(e.curve.x, y, e.setup, hand=hand or 1, keep=keep, at=at)
        except Exception as exc:
            e.start = None
            if not quiet:
                self.say(f"guess failed: {exc}", error=True)
            return False
        return True

    def guess(self):
        e = self.current()
        if e is None:
            return
        e.setup = self._setup_from_controls(e)
        e.result = None
        if self._guess(e, keep=None):
            self.say("start values guessed from the data")
        self._fill_table()
        self._redraw()

    def _fill_table(self):
        e = self.current()
        self._filling = True
        try:
            self.table.setRowCount(0)
            if e is None or e.start is None:
                return
            names = M.param_names(e.setup)
            self.table.setRowCount(len(names))
            res = e.result
            for r, name in enumerate(names):
                sp = e.start[name]
                val = res.values[name] if res else sp.value
                err = res.errors.get(name) if res else None
                cells = [_pretty(name), f"{val:.6g}",
                         "" if err is None else f"{err:.2g}",
                         M.param_unit(name, e.curve.x_unit, e.curve.y_unit, e.setup), "",
                         "" if not np.isfinite(sp.min) else f"{sp.min:.6g}",
                         "" if not np.isfinite(sp.max) else f"{sp.max:.6g}"]
                for c, text in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(text)
                    if c in (0, 2, 3):
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    if c == 0:
                        it.setData(QtCore.Qt.UserRole, name)
                    if c == 4:
                        it.setFlags((it.flags() | QtCore.Qt.ItemIsUserCheckable)
                                    & ~QtCore.Qt.ItemIsEditable)
                        it.setCheckState(QtCore.Qt.Unchecked if sp.vary else QtCore.Qt.Checked)
                    self.table.setItem(r, c, it)
        finally:
            self._filling = False

    def _param_edited(self, item):
        e = self.current()
        if self._filling or e is None or e.start is None:
            return
        name = self.table.item(item.row(), 0).data(QtCore.Qt.UserRole)
        sp = e.start[name]
        text = item.text()
        col = item.column()
        if col == 1:
            sp.value = _float(text, sp.value)
        elif col == 4:
            sp.vary = item.checkState() != QtCore.Qt.Checked
        elif col == 5:
            sp.min = _float(text, -np.inf) if text.strip() else -np.inf
        elif col == 6:
            sp.max = _float(text, np.inf) if text.strip() else np.inf
        self._redraw(show_start=True)

    # ── fitting ────────────────────────────────────────────────────────────
    def _fit_entry(self, e: Entry, start) -> M.Result:
        res = M.fit(e.curve.x, e.y(), e.setup, start=start)
        old = start or {}
        e.result = res
        # the fitted values become the next start, keeping fixed / bounds
        e.start = {n: M.Spec(v, old[n].vary if n in old else True,
                             old[n].min if n in old else -np.inf,
                             old[n].max if n in old else np.inf)
                   for n, v in res.values.items()}
        if not old:                              # guessed inside fit(): take its bounds
            g = M.guess(e.curve.x, e.y(), e.setup, hand=res.hand)
            for n, sp in e.start.items():
                sp.min, sp.max = g[n].min, g[n].max
        return res

    def fit(self):
        e = self.current()
        if e is None:
            return
        e.setup = self._setup_from_controls(e)
        e.reference = self._reference_from_controls(e)
        try:
            res = self._fit_entry(e, e.start)
        except Exception as exc:
            self.say(f"fit failed: {exc}", error=True)
            return
        self._remember(e)
        self._after_fit([e])
        hand = f", hand {res.hand:+d}" if e.setup.mode == "complex" else ""
        warn = M.suspicious(res)
        self.say((f"CHECK: {warn}. " if warn else "fitted: ")
                 + f"{res.message}  (χ²_red = {res.redchi:.4g}{hand})", error=bool(warn))

    def fit_all(self):
        if self.request_stop():
            return
        cur = self.current()
        if cur is None:
            return
        template = self._setup_from_controls(cur)
        ref = self._reference_from_controls(cur)
        bad = []
        # the reference sweep is not fitted with itself taken out of it
        targets = [e for e in self.targets() if ref is None or e is not ref[0]]
        self._remember(cur)
        done = []
        self._batch_begin([self.fit_all_btn])
        try:
            for k, e in enumerate(targets, 1):
                if not self._fit_one_of_all(e, cur, template, ref, bad):
                    continue
                done.append(e)
                if not self._batch_step(k, len(targets), e):
                    break
        finally:
            self._batch_end()
        stopped = len(done) < len(targets) and self._stop
        targets = done
        self._after_fit(targets)
        n = len(targets) - len(bad)
        check = [e.curve.label for e in targets if e.result and M.suspicious(e.result)]
        msg = f"fitted {n} of {len(targets)}"
        if stopped:
            msg = f"STOPPED: fitted {n} (the rest not touched)"
        elif len(targets) < len(self.entries):
            msg += " selected"
        if check:
            msg += f"; check {', '.join(check[:6])}{' ...' if len(check) > 6 else ''} (marked ⚠)"
        if bad:
            msg += " -- " + "; ".join(bad[:3])
        self.say(msg, error=bool(bad or check or stopped))

    def _fit_one_of_all(self, e, cur, template, ref, bad) -> bool:
        """One curve of Fit all, with the model of the one on screen."""
        e.reference = ref
        e.located = None
        e.touched = True
        mode = template.mode if (template.mode == "real" or e.curve.z is not None) else "real"
        # the model is shared, the RANGE is not: each curve keeps its own band
        # or gets the carried one around its own peak (never cur's field values)
        e.setup = M.Setup(**{**template.__dict__, "mode": mode,
                             "xmin": e.setup.xmin, "xmax": e.setup.xmax})
        self._apply_rule(e)
        try:
            start = cur.start if e is cur else None
            self._fit_entry(e, start)
        except Exception as exc:
            e.result = None
            bad.append(f"{e.curve.label}: {exc}")
        return True

    def _update_row(self, e):
        """The curve list's row of one entry: its fit in one line."""
        it = self.curve_list.topLevelItem(self.entries.index(e))
        r = e.result
        if r is None:
            it.setText(1, "failed")
            return
        w, we = r.fwhm(1)
        warn = M.suspicious(r)
        it.setText(1, ("⚠ check  " if warn else "")
                   + f"x0 = {M.fmt_pm(r.values['p1_center'], r.errors['p1_center'])}, "
                     f"FWHM = {M.fmt_pm(w, we)} {e.curve.x_unit}")
        it.setToolTip(1, warn or f"χ²_red = {r.redchi:.4g}")
        it.setForeground(1, QtGui.QColor(C["danger"] if warn else C["text"]))

    def _after_fit(self, entries):
        for e in entries:
            self._update_row(e)
        self.curve_list.resizeColumnToContents(1)
        self._fill_table()
        self._redraw()
        self._fill_results()

    # ── drawing ────────────────────────────────────────────────────────────
    def _redraw(self, show_start: bool = False):
        for plot, it in self._items:
            plot.removeItem(it)
        self._items = []
        self.legend.clear()
        e = self.current()
        if e is None:
            return
        c = e.curve
        try:
            y = e.y()
            # what the fit sees: derivative-divided, reference taken out
            shown = M.displayed(c.x, y, e.setup)
        except Exception as exc:
            self.say(str(exc), error=True)
            return
        parts = ([("Re", np.real, TAB10[0]), ("Im", np.imag, TAB10[3])]
                 if e.setup.mode == "complex" else [(c.y_name, np.real, TAB10[0])])
        # with the delay option: drawn with the cable's phase taken out (fitted
        # tau, or its start value), or Re and Im spin and look like noise
        un = None
        if e.result is not None and not show_start:
            un = M.unwinder(e.setup, e.result.values, e.result.xc)
        elif e.start is not None:
            try:
                xc = M.prepare(c.x, y, e.setup).xc
                un = M.unwinder(e.setup, {n: sp.value for n, sp in e.start.items()}, xc)
            except Exception:
                un = None

        def frame(x, v):
            return v * un(x) if un is not None else v

        # Re left, Im right (each scaled to itself) -- complex data, box ticked
        two = self.two_axes.isChecked() and len(parts) == 2
        self.two_axes.setEnabled(len(parts) == 2)
        view = {"Re": self.plot, "Im": self.vb2 if two else self.plot}

        for name, fn, col in parts:
            self._add(view.get(name, self.plot),
                      _points(shown.x, fn(frame(shown.x, shown.y)), col, 4,
                              name=f"{name} data"))
        model_vals = None
        xs = None
        if e.result is not None and not show_start:
            xs = np.linspace(*e.result.xrange, 800)
            model_vals = frame(xs, e.result.evaluate(xs))
            label = "fit"
            rx, rr = M.residuals(e.result, c.x, y)
            rr = frame(rx, rr)
            for name, fn, col in parts:
                self._add(self.rplot, _points(rx, fn(rr), col, 3))
            self._add(self.rplot, pg.InfiniteLine(pos=0, angle=0,
                                                  pen=pg.mkPen(C["muted"], width=1)))
        elif e.start is not None:
            try:
                d = M.prepare(c.x, y, e.setup)
                xs = np.linspace(d.x[0], d.x[-1], 800)
                model_vals = frame(xs, M.evaluate({n: s.value for n, s in e.start.items()},
                                                  xs, e.setup, e.setup.hand or 1, d.xc,
                                                  dd_half=d.dd_half))
                label = "start"
            except Exception:
                model_vals = None
        if model_vals is not None:
            style = QtCore.Qt.SolidLine if label == "fit" else QtCore.Qt.DashLine
            for name, fn, col in parts:
                self._add(view.get(name, self.plot), pg.PlotDataItem(
                    xs, fn(model_vals), pen=pg.mkPen(C["accent"] if len(parts) == 1 else col,
                                                     width=2, style=style),
                    name=f"{name} {label}"))
        self.plot.setLabel("bottom", axis_title(c.x_name, c.x_unit))
        title = M.y_title(c, e.setup, unwound=un is not None)
        if two:
            self.plot.showAxis("right")
            self.plot.setLabel("left", f"Re {title}", color=TAB10[0])
            self.plot.setLabel("right", f"Im {title}", color=TAB10[3])
            self.vb2.setGeometry(self.plot.vb.sceneBoundingRect())
            self.vb2.enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            self.plot.hideAxis("right")
            self.plot.setLabel("left", title, color=C["text"])
        self.rplot.setLabel("bottom", axis_title(c.x_name, c.x_unit))
        self._thin()

    def _add(self, plot, item):
        plot.addItem(item)
        self._items.append((plot, item))
        if plot is self.vb2:
            # a bare ViewBox: no legend entry and no clip-to-view of its own
            name = item.opts.get("name") if hasattr(item, "opts") else None
            if name:
                self.legend.addItem(item, name)
            if hasattr(item, "setClipToView"):
                item.setClipToView(True)

    def _thin(self, *_):
        """Every k-th symbol when more than MAX_SYMBOLS would be in view."""
        lo, hi = self.plot.viewRange()[0]
        for _, item in self._items:
            if getattr(item, "thinned", False) and item.xData is not None:
                k = thin_step(item.xData, lo, hi)
                if item.opts.get("downsample") != k:
                    item.setDownsampling(ds=k, auto=False, method="subsample")

    # ── results out ────────────────────────────────────────────────────────
    def fitted(self) -> list[tuple[Curve, M.Result]]:
        return [(e.curve, e.result) for e in self.entries if e.result is not None]

    def columns(self) -> list[M.Column]:
        rows = self.fitted()
        if not rows:
            raise ValueError("nothing fitted yet -- press Fit or Fit all")
        return M.results_table(rows)

    def _fill_results(self):
        rows = self.fitted()
        cols = M.results_table(rows) if rows else []
        self.results.clear()
        self.results.setColumnCount(len(cols))
        self.results.setRowCount(len(rows))
        self.results.setHorizontalHeaderLabels(
            [f"{c.name}\n{c.unit}" if c.unit else c.name for c in cols])
        for j, col in enumerate(cols):
            self.results.horizontalHeaderItem(j).setToolTip(col.comment)
            for i, v in enumerate(col.values):
                text = v if isinstance(v, str) else ("" if not np.isfinite(v) else f"{v:.6g}")
                self.results.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        self.results.resizeColumnsToContents()

    def _guard(self, what, fn):
        try:
            return fn()
        except Exception as exc:
            self.say(f"{what} failed: {exc}", error=True)
            return None

    def _stem(self) -> str:
        e = next((e for e in self.entries if e.curve.source), None)
        return (Path(e.curve.source).stem if e else "unsaved_run") + "_fmr_fit"

    def _dir(self) -> Path:
        e = next((e for e in self.entries if e.curve.source), None)
        for d in (self.last_dir, Path(e.curve.source).parent if e else None):
            if d and Path(d).is_dir():
                return Path(d)
        return Path.home()

    def copy_results(self):
        cols = self._guard("copy", self.columns)
        if cols is None:
            return
        with tempfile.TemporaryDirectory() as tmp:
            p = M.write_results(Path(tmp) / "r.dat", cols)
            QtWidgets.QApplication.clipboard().setText(p.read_text(encoding="utf-8"))
        self.say("results copied to the clipboard (tab-separated)")

    def save_results(self):
        cols = self._guard("save", self.columns)
        if cols is None:
            return
        fn, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save results", str(self._dir() / f"{self._stem()}.csv"),
            "Comma-separated (*.csv);;Tab-separated (*.dat)")
        if not fn:
            return
        path = Path(fn)
        if path.suffix.lower() not in (".csv", ".dat", ".txt"):
            path = path.with_suffix(".dat" if "Tab" in chosen else ".csv")
        self.last_dir = path.parent
        if self._guard("save", lambda: M.write_results(path, cols)):
            self.say(f"saved {path}")

    def save_image(self):
        e = self.current()
        if e is None or e.result is None:
            self.say("fit the curve on screen first", error=True)
            return
        fig = self._guard("image", lambda: M.figure_fit(e.curve, e.result, e.y()))
        if fig is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", str(self._dir() / f"{self._stem()}.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if fn:
            self.last_dir = Path(fn).parent
            if self._guard("image", lambda: save_figure(fig, fn)):
                self.say(f"saved {fn}")

    def send_to_origin(self):
        """Same route as the viewer: a child process (see aaltoview/origin.py)."""
        from aaltoview.origin import available, save_payload
        reason = available()
        if reason:
            self.say(f"Origin: {reason}", error=True)
            return
        if self._origin_proc is not None:
            self.say("Origin: still sending the previous one…")
            return
        cols = self._guard("Origin", self.columns)
        if cols is None:
            return
        tmp = Path(tempfile.mkdtemp(prefix="fmr_fit_origin_"))
        payload = self._guard("Origin", lambda: save_payload(
            tmp / "payload.npz", table=[c.__dict__ for c in cols], name=self._stem()))
        if payload is None:
            return
        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(str(tmp))
        proc.setProgram(sys.executable)
        proc.setArguments(["-m", "aaltoview.origin", str(payload)])
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.finished.connect(lambda *_: self._origin_done(proc, tmp))
        self._origin_proc = proc
        self.say("Origin: sending…")
        proc.start()

    def _origin_done(self, proc, tmp: Path):
        out = bytes(proc.readAll()).decode(errors="replace").strip().splitlines()
        last = out[-1] if out else "no answer"
        self._origin_proc = None
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()
        if last.startswith("OK"):
            self.say(f"Origin: sent as {last[3:]}")
        else:
            self.say(f"Origin: {last.removeprefix('ERROR ')}", error=True)
