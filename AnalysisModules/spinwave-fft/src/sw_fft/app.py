"""app.py -- the Spin-wave FFT window.

Two tabs, the two stages of the analysis:

* FFT: the input (a map from the viewer, or curves) transformed line by line,
  shown as |FFT| over (k, the input's y), with the line under the cursor below
  it; then the peaks of every line, drawn on both.
* Dispersion: the peaks as points (k, f, B), where f and B come from the
  input's axes or are typed; the Kalinikos-Slavin dispersion of a stripe
  (Guslienko pinning) fitted through them; the fit drawn back on the FFT map.

The maths is in fft.py / waveguide.py / inputs.py / results.py (no Qt).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from aaltoview.apps.theme import C
from aaltoview.apps.viewer import _float, _pg_cmap, _plain_axes
from aaltoview.export import Curve, MapData, axis_title, edges, save_figure, write_map

from . import fft as F
from . import inputs as I
from . import results as R
from . import waveguide as W


def _tag(text):
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("tag")
    return lab


def _combo(items, tip=""):
    c = QtWidgets.QComboBox()
    for text, data in items:
        c.addItem(text, data)
    c.setToolTip(tip)
    return c


class FFTWindow(QtWidgets.QWidget):
    PCOLS = ("Use", "Line", "k (rad/µm)", "f (GHz)", "B (mT)", "|FFT|")
    MCOLS = ("Parameter", "Value", "±", "Unit", "Fit", "Min", "Max")

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.resize(1500, 920)
        self.inputs: list[I.Input] = []
        self.spectrum: F.Spectrum | None = None
        self.peaks: list[F.Peak] = []
        self.points: list[W.Point] = []
        self.specs = W.default_specs()
        self.result: W.Result | None = None
        self.line = 0
        self._filling = False
        self._origin_proc: QtCore.QProcess | None = None
        self.last_dir: Path | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        title = QtWidgets.QLabel("SPIN-WAVE FFT")
        title.setObjectName("title")
        outer.addWidget(title)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_fft_tab(), "FFT")
        self.tabs.addTab(self._build_dispersion_tab(), "Dispersion")
        self.status = QtWidgets.QLabel("Waiting for data: in AaltoView, Map → Analysis "
                                       "(or 1D plots → Analysis)")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self._fill_params()
        self._set_enabled(False)

    # ── building ───────────────────────────────────────────────────────────
    def _build_fft_tab(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(6)

        lv.addWidget(_tag("INPUTS"))
        self.input_list = QtWidgets.QTreeWidget()
        self.input_list.setHeaderLabels(["Input", "Lines", "From"])
        self.input_list.setRootIsDecorated(False)
        self.input_list.setMaximumHeight(120)
        self.input_list.currentItemChanged.connect(lambda *_: self._input_changed())
        lv.addWidget(self.input_list)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Remove"); b.clicked.connect(self.remove_input)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Clear all"); b.setObjectName("danger")
        b.clicked.connect(self.clear)
        row.addWidget(b)
        row.addStretch(1)
        lv.addLayout(row)

        lv.addWidget(_tag("TRANSFORM  (along x, every line)"))
        g = QtWidgets.QGridLayout()
        self.part = _combo([("complex (X + iY)", "complex"), ("Re", "real"), ("Im", "imag"),
                            ("|z|", "abs")],
                           "complex: a wave towards +x at +k, towards -x at -k.\n"
                           "A real part is its own mirror image: the sign of k means nothing.")
        self.window_ = _combo([(w, w) for w in F.WINDOWS],
                              "The taper against leakage from the ends of the scan:\n"
                              "none = sharpest peak, highest side lobes; blackman = the reverse.")
        self.window_.setCurrentText("hann")
        self.tukey = QtWidgets.QDoubleSpinBox(); self.tukey.setRange(0, 1)
        self.tukey.setSingleStep(0.1); self.tukey.setValue(0.5)
        self.tukey.setToolTip("Tukey: the fraction of the line that is tapered (0 = none, "
                              "1 = Hann).")
        self.detrend = _combo([("subtract the mean", "mean"), ("subtract a line", "linear"),
                               ("none", "none")],
                              "The offset (and slope) of a line would sit at k = 0 and leak.")
        self.pad = _combo([(f"x{p}", p) for p in (1, 2, 4, 8, 16)],
                          "Zero-padding: a smoother spectrum and a better peak position. It "
                          "does NOT\nseparate two close peaks: the resolution is 2π / (length "
                          "scanned).")
        self.pad.setCurrentIndex(2)
        self.k_unit = _combo([("k  (rad/µm)", "rad/um"), ("1/λ  (1/µm)", "1/um")],
                             "rad/µm: k = 2π/λ;  1/µm: 1/λ.")
        self.show_ = _combo([(R.SHOW_TEXT[s], s) for s in R.SHOW])
        self.cmap = _combo([(c, c) for c in ("magma", "viridis", "inferno", "grey")])
        items = [("part", self.part), ("window", self.window_), ("Tukey α", self.tukey),
                 ("offset", self.detrend), ("zero-padding", self.pad), ("axis", self.k_unit),
                 ("show", self.show_), ("colours", self.cmap)]
        for i, (lab, w) in enumerate(items):
            g.addWidget(QtWidgets.QLabel(lab), i // 2, 2 * (i % 2))
            g.addWidget(w, i // 2, 2 * (i % 2) + 1)
        lv.addLayout(g)
        for w in (self.part, self.window_, self.detrend, self.pad, self.k_unit):
            w.currentIndexChanged.connect(self.transform)
        self.tukey.valueChanged.connect(self.transform)
        for w in (self.show_, self.cmap):
            w.currentIndexChanged.connect(self._redraw)
        # TR-MOKE: the laser aliases f to f - n f_rep; a negative alias arrives
        # conjugated, and the complex FFT's branch jumps between +k and -k
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("TR-MOKE unfold"))
        self.unfold = _combo([("off", 0.0)] + [(f"{r:g} MHz laser", r) for r in F.REP_RATES],
                             "Stroboscopic detection aliases f to f − n·f_rep; where that is "
                             "negative the lock-in\nrecords the complex conjugate (a wave "
                             "running the other way): the branch jumps between\n+k and −k "
                             "every f_rep/2. Unfold conjugates those lines back. Complex data "
                             "only;\nthe frequency of each line comes from the Dispersion "
                             "tab's 'frequency f'.")
        self.unfold_invert = QtWidgets.QCheckBox("other half")
        self.unfold_invert.setToolTip("Conjugate the lines with a POSITIVE alias instead: "
                                      "the lock-in's sign\nconvention decides which half. "
                                      "Try it if the branch comes out at −k.")
        row.addWidget(self.unfold, 1)
        row.addWidget(self.unfold_invert)
        lv.addLayout(row)
        self.unfold.currentIndexChanged.connect(self.transform)
        self.unfold_invert.toggled.connect(self.transform)
        self.x_note = QtWidgets.QLabel("")
        self.x_note.setWordWrap(True)
        self.x_note.setStyleSheet(f"color:{C['muted']}; font-size:11px;")
        lv.addWidget(self.x_note)

        lv.addWidget(_tag("PEAKS  (in every line)"))
        g = QtWidgets.QGridLayout()
        self.n_peaks = QtWidgets.QSpinBox(); self.n_peaks.setRange(1, 6)
        self.side = _combo([("both signs", "both"), ("k > 0", "positive"), ("k < 0", "negative")],
                           "k > 0: waves going towards +x (complex data).")
        self.kmin = QtWidgets.QLineEdit("0.3")
        self.kmin.setToolTip("|k| below this is left out: what is left of the offset.")
        self.kmax = QtWidgets.QLineEdit("")
        self.kmax.setPlaceholderText("no limit")
        self.min_rel = QtWidgets.QDoubleSpinBox(); self.min_rel.setRange(0, 1)
        self.min_rel.setSingleStep(0.05); self.min_rel.setValue(0.1)
        self.min_rel.setToolTip(
            "A peak must be at least this fraction of the line's own highest point (in the\n"
            "allowed |k| range and side): 0.2 = at least 20 % of it. It keeps small side\n"
            "bumps out, but on a line with NO wave the highest noise bump still passes --\n"
            "for that, use '≥ × noise'.")
        self.min_snr = QtWidgets.QDoubleSpinBox(); self.min_snr.setRange(0, 1000)
        self.min_snr.setSingleStep(0.5); self.min_snr.setValue(0.0)
        self.min_snr.setSpecialValueText("off")
        self.min_snr.setToolTip(
            "Signal to noise: a peak must be at least this many times the line's noise --\n"
            "the median |FFT| over the allowed |k| range (a peak is a few bins, the rest is\n"
            "the noise floor). A line with no wave then gets no peak. 3-5 is a good start;\n"
            "the peak table lists each peak's SNR. 0 = off.")
        for i, (lab, w) in enumerate([("peaks per line", self.n_peaks), ("side", self.side),
                                      ("|k| from", self.kmin), ("|k| to", self.kmax),
                                      ("≥ × highest", self.min_rel), ("≥ × noise", self.min_snr)]):
            g.addWidget(QtWidgets.QLabel(lab), i // 2, 2 * (i % 2))
            g.addWidget(w, i // 2, 2 * (i % 2) + 1)
        lv.addLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.follow = QtWidgets.QCheckBox("follow the last peak, within ±")
        self.follow.setToolTip(
            "Follow a branch: on each line, look only within ± this bandwidth (in the k unit)\n"
            "of the peak found on the line before. It starts on the line shown below the\n"
            "map -- click a line with a clear peak first -- and walks out in both directions.\n"
            "A line with nothing in the window keeps the last position.")
        self.follow_w = QtWidgets.QLineEdit("0.5")
        self.follow_w.setToolTip("The bandwidth, in the axis unit (rad/µm or 1/µm).")
        self.follow_w.setMaximumWidth(80)
        row.addWidget(self.follow)
        row.addWidget(self.follow_w)
        row.addStretch(1)
        lv.addLayout(row)
        hint = QtWidgets.QLabel("click a peak on the map to leave it out (again: take it back)")
        hint.setStyleSheet(f"color:{C['muted']}; font-size:11px;")
        lv.addWidget(hint)
        self.peaks_btn = QtWidgets.QPushButton("Find peaks")
        self.peaks_btn.setObjectName("primary")
        self.peaks_btn.clicked.connect(self.find_peaks)
        lv.addWidget(self.peaks_btn)

        lv.addWidget(_tag("EXPORT"))
        g = QtWidgets.QGridLayout()
        spec = [("Save FFT map…", self.save_fft, "The spectra as text: a matrix (k across, "
                 "the lines down)\nor XYZ columns."),
                ("Copy FFT map", self.copy_fft, "The matrix, tab-separated, onto the clipboard."),
                ("FFT to Origin", self.fft_to_origin, "A matrix + colour map in Origin."),
                ("Save peaks…", self.save_peaks, "One row per peak: line, k, |k|, λ, "
                 "amplitude, width."),
                ("Copy peaks", self.copy_peaks, "The peak table onto the clipboard."),
                ("Save image…", self.save_image, "The FFT map with the peaks and the fit: "
                 "PNG, PDF or SVG.")]
        for i, (text, fn, tip) in enumerate(spec):
            b = QtWidgets.QPushButton(text); b.setToolTip(tip); b.clicked.connect(fn)
            g.addWidget(b, i // 2, i % 2)
        lv.addLayout(g)
        lv.addStretch(1)
        split.addWidget(left)

        self.glw = pg.GraphicsLayoutWidget()
        self.mplot = self.glw.addPlot(row=0, col=0)
        self.img = pg.ImageItem(autoDownsample=True)
        self.mplot.addItem(self.img)
        self.cbar = pg.ColorBarItem(colorMap=_pg_cmap("magma", False), interactive=True)
        self.cbar.setImageItem(self.img, insert_in=self.mplot)
        _plain_axes(self.mplot, self.cbar.axis)
        self.peak_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(C["accent"], width=1.2),
                                               brush=None, symbol="o")
        self.mplot.addItem(self.peak_scatter)
        self.model_items: list = []
        self.hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(C["accent"], width=1,
                                                           style=QtCore.Qt.DashLine))
        self.mplot.addItem(self.hline, ignoreBounds=True)
        self.mplot.scene().sigMouseClicked.connect(self._map_clicked)
        self.lplot = self.glw.addPlot(row=1, col=0)
        self.lplot.setXLink(self.mplot)
        _plain_axes(self.lplot)
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        rv.addWidget(self.glw, 1)
        self.readout = QtWidgets.QLabel("click the map to show that line below")
        self.readout.setStyleSheet(f"color:{C['accent']};")
        rv.addWidget(self.readout)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setSizes([430, 1050])
        return split

    def _build_dispersion_tab(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(6)

        lv.addWidget(_tag("WHERE EACH LINE WAS MEASURED"))
        g = QtWidgets.QGridLayout()
        self.f_src = QtWidgets.QComboBox()
        self.f_const = QtWidgets.QLineEdit(); self.f_const.setPlaceholderText("GHz")
        self.b_src = QtWidgets.QComboBox()
        self.b_const = QtWidgets.QLineEdit(); self.b_const.setPlaceholderText("mT")
        self.b_const.setToolTip("μ0H when it is not a scan axis -- e.g. from the file's "
                                "comment. Its sign does not matter.")
        g.addWidget(QtWidgets.QLabel("frequency f"), 0, 0)
        g.addWidget(self.f_src, 0, 1)
        g.addWidget(self.f_const, 0, 2)
        g.addWidget(QtWidgets.QLabel("field μ0H"), 1, 0)
        g.addWidget(self.b_src, 1, 1)
        g.addWidget(self.b_const, 1, 2)
        lv.addLayout(g)
        for w in (self.f_src, self.b_src):
            w.currentIndexChanged.connect(self._sources_changed)
        for w in (self.f_const, self.b_const):
            w.editingFinished.connect(self._sources_changed)

        lv.addWidget(_tag("POINTS  (one per peak; untick to leave out)"))
        self.ptable = QtWidgets.QTableWidget(0, len(self.PCOLS))
        self.ptable.setHorizontalHeaderLabels(self.PCOLS)
        self.ptable.verticalHeader().setVisible(False)
        self.ptable.itemChanged.connect(self._point_toggled)
        lv.addWidget(self.ptable, 1)

        lv.addWidget(_tag("MODEL  Kalinikos-Slavin, lowest thickness mode, in-plane M"))
        g = QtWidgets.QGridLayout()
        self.pinning = _combo([("Guslienko (dipolar pinning: w_eff)", "guslienko"),
                               ("unpinned edges (w)", "unpinned"),
                               ("none: infinite film (k_y = 0)", "none")],
                              "k_y = n π / w_eff across the stripe. Guslienko et al., PRB 66, "
                              "132402 (2002):\nw_eff = w D/(D − 2), D = 2π / (p (1 + 2 ln 1/p)),"
                              " p = d / w.\nnone: no width mode -- w and n are not used or "
                              "fitted.")
        self.pinning.currentIndexChanged.connect(self._fill_params)
        self.robust = QtWidgets.QCheckBox("robust (a stray peak counts less)")
        g.addWidget(QtWidgets.QLabel("width modes"), 0, 0)
        g.addWidget(self.pinning, 0, 1)
        g.addWidget(self.robust, 1, 0, 1, 2)
        lv.addLayout(g)
        self.mtable = QtWidgets.QTableWidget(0, len(self.MCOLS))
        self.mtable.setHorizontalHeaderLabels(self.MCOLS)
        self.mtable.verticalHeader().setVisible(False)
        self.mtable.itemChanged.connect(self._param_edited)
        lv.addWidget(self.mtable, 1)
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.clicked.connect(self.fit)
        lv.addWidget(self.fit_btn)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.dplot = pg.PlotWidget()
        _plain_axes(self.dplot.getPlotItem())
        self.dplot.setLabel("bottom", "k (rad/µm)")
        self.dplot.setLabel("left", "f (GHz)")
        self.dplot.addLegend(offset=(10, 10))
        rv.addWidget(self.dplot, 3)
        rv.addWidget(_tag("RESULTS"))
        self.rtable = QtWidgets.QTableWidget(0, 5)
        self.rtable.setHorizontalHeaderLabels(["Parameter", "Value", "±", "Unit", "Meaning"])
        self.rtable.verticalHeader().setVisible(False)
        self.rtable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        rv.addWidget(self.rtable, 2)
        eb = QtWidgets.QHBoxLayout()
        for text, fn in (("Copy results", self.copy_results), ("Save results…", self.save_results),
                         ("Results to Origin", self.results_to_origin),
                         ("Save image…", self.save_dispersion_image)):
            b = QtWidgets.QPushButton(text); b.clicked.connect(fn)
            eb.addWidget(b)
        eb.addStretch(1)
        rv.addLayout(eb)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        return split

    # ── small helpers ──────────────────────────────────────────────────────
    def say(self, text: str, error: bool = False):
        self.status.setStyleSheet(f"color:{C['danger'] if error else C['muted']};"
                                  " font-size:11px;")
        self.status.setText(text)

    def _set_enabled(self, on: bool):
        for w in (self.peaks_btn, self.fit_btn):
            w.setEnabled(on)

    def current(self) -> I.Input | None:
        i = self.input_list.indexOfTopLevelItem(self.input_list.currentItem())
        return self.inputs[i] if 0 <= i < len(self.inputs) else None

    def settings(self) -> F.Settings:
        return F.Settings(window=self.window_.currentData(), tukey_alpha=self.tukey.value(),
                          detrend=self.detrend.currentData(), pad=self.pad.currentData(),
                          k_unit=self.k_unit.currentData(), part=self.part.currentData())

    def peak_settings(self) -> F.PeakSettings:
        return F.PeakSettings(n_peaks=self.n_peaks.value(), side=self.side.currentData(),
                              k_min=_float(self.kmin.text(), 0.0),
                              k_max=_float(self.kmax.text(), np.inf) if self.kmax.text().strip()
                              else np.inf,
                              min_rel=self.min_rel.value(), min_snr=self.min_snr.value())

    def _guard(self, what, fn):
        try:
            return fn()
        except Exception as exc:
            self.say(f"{what} failed: {exc}", error=True)
            return None

    # ── data in ────────────────────────────────────────────────────────────
    def add_maps(self, maps: list[MapData]):
        self._add([I.from_map(m) for m in maps], "map")

    def add_curves(self, curves: list[Curve]):
        self._add(I.from_curves(curves), "curves")

    def _add(self, new: list[I.Input], kind: str):
        for inp in new:
            self.inputs.append(inp)
            it = QtWidgets.QTreeWidgetItem([inp.label, str(len(inp.y)),
                                            Path(inp.source).name if inp.source else "unsaved"])
            it.setToolTip(0, f"{inp.label}\n{inp.source or 'unsaved run'}\n"
                             f"{inp.z_name} along {inp.x_name}, one line per {inp.y_name}")
            self.input_list.addTopLevelItem(it)
        self.input_list.resizeColumnToContents(0)
        self.input_list.setCurrentItem(self.input_list.topLevelItem(len(self.inputs) - 1))
        inp = new[-1]
        self.say(f"received {len(new)} {kind if len(new) == 1 else kind + 's'}: "
                 f"{inp.z_name} along {inp.x_name} ({inp.x_unit}), {len(inp.y)} lines "
                 f"along {inp.y_name}"
                 + (f"; FFT resolution 2π/L = {self.spectrum.resolution:.3g} "
                    f"{R.k_title(self.spectrum.k_unit)[1]}" if self.spectrum is not None else ""))

    def remove_input(self):
        i = self.input_list.indexOfTopLevelItem(self.input_list.currentItem())
        if 0 <= i < len(self.inputs):
            self.inputs.pop(i)
            self.input_list.takeTopLevelItem(i)
        if not self.inputs:
            self.clear()

    def clear(self):
        self.inputs.clear()
        self.input_list.clear()
        self.spectrum, self.peaks, self.points, self.result = None, [], [], None
        self.img.clear()
        self.peak_scatter.clear()
        self.lplot.clear()
        self._clear_model()
        self.ptable.setRowCount(0)
        self.rtable.setRowCount(0)
        self.dplot.clear()
        self._set_enabled(False)

    def _input_changed(self):
        inp = self.current()
        if inp is None:
            return
        # a real input has no second quadrature to transform
        model = self.part.model()
        for i in range(self.part.count()):
            item = model.item(i)
            item.setEnabled(inp.complex or self.part.itemData(i) != "imag")
        if not inp.complex and self.part.currentData() in ("complex", "imag"):
            self.part.setCurrentIndex(self.part.findData("real"))
        sc = F.length_scale(inp.x_unit)
        self.x_note.setText(
            f"x = {inp.x_name} in {inp.x_unit or '(no unit)'}"
            + ("" if sc is not None else ": NOT a length unit -- taken as µm") +
            ("" if inp.complex else ".  Real data: ±k are mirror images."))
        self._fill_sources(inp)
        self.peaks, self.points, self.result = [], [], None
        self.line = 0
        self.transform()
        self._set_enabled(True)

    # ── transform, peaks ───────────────────────────────────────────────────
    def transform(self, *_):
        inp = self.current()
        if inp is None:
            return
        rows, note = self._unfolded(inp)
        sp = self._guard("FFT", lambda: F.spectra(inp.x, rows, self.settings(), inp.x_unit))
        if sp is None:
            return
        old_unit = self.spectrum.k_unit if self.spectrum is not None else sp.k_unit
        self.spectrum = sp
        if old_unit != sp.k_unit:            # the |k| range is typed in the axis unit
            f = 1 / (2 * np.pi) if sp.k_unit == "1/um" else 2 * np.pi
            for edit in (self.kmin, self.kmax, self.follow_w):
                if edit.text().strip():
                    edit.setText(f"{_float(edit.text(), 0.0) * f:.4g}")
        if self.peaks:                       # settings changed: find them again
            self.find_peaks(quiet=True)
            if note:
                self.say(f"peaks found again{note}", error=note.startswith(";  unfold OFF"))
        else:
            self._redraw()
            self.say(f"FFT of {len(inp.y)} lines: {sp.F.shape[1]} points in k, resolution "
                     f"2π/L = {sp.resolution:.3g} {R.k_title(sp.k_unit)[1]}{note}",
                     error=note.startswith(";  unfold OFF"))

    def _unfolded(self, inp: I.Input):
        """The rows to transform: TR-MOKE-unfolded when asked (and possible)."""
        rep = self.unfold.currentData() or 0.0
        if not rep:
            return inp.rows, ""
        if not inp.complex or self.part.currentData() != "complex":
            return inp.rows, ";  unfold OFF: it needs the complex (X + iY) signal"
        fs = I.Source(self.f_src.currentData() or "constant", self._const(self.f_const))
        f = np.array([I.quantity(inp, i, fs) for i in range(len(inp.y))])
        if fs.how == "constant" or not np.isfinite(f).any():
            return inp.rows, (";  unfold OFF: the lines have no frequency of their own "
                              "(Dispersion tab: frequency f)")
        rows, flipped = F.unfold(inp.rows, f, rep, self.unfold_invert.isChecked())
        return rows, (f";  unfolded for {rep:g} MHz: {int(flipped.sum())} of {len(f)} "
                      f"lines conjugated")

    def find_peaks(self, *_, quiet=False):
        inp, sp = self.current(), self.spectrum
        if inp is None or sp is None:
            return
        ps = self.peak_settings()
        width = _float(self.follow_w.text(), float("nan"))
        follow = self.follow.isChecked()
        if follow and not width > 0:
            self.say("follow the last peak: type a bandwidth > 0 (in the k unit)", error=True)
            return
        seed = min(max(self.line, 0), len(inp.y) - 1)
        if follow:
            self.peaks = F.track_peaks(sp, ps, width, seed=seed,
                                       order=np.argsort(np.asarray(inp.y), kind="stable"))
        else:
            self.peaks = F.all_peaks(sp, ps)
        self._make_points()
        self._redraw()
        n_lines = len({p.line for p in self.peaks})
        if not quiet:
            how = (f", following from {inp.line_label(seed)} within ±{width:g} "
                   f"{R.k_title(sp.k_unit)[1]}" if follow else "")
            self.say(f"{len(self.peaks)} peaks in {n_lines} of {len(inp.y)} lines{how}"
                     + ("" if n_lines == len(inp.y) else
                        " (lines without a peak above the threshold have none)"))

    # ── drawing ────────────────────────────────────────────────────────────
    def _clear_model(self):
        for it in self.model_items:
            self.mplot.removeItem(it)
        self.model_items = []

    def _redraw(self, *_):
        inp, sp = self.current(), self.spectrum
        if inp is None or sp is None:
            return
        m = R.fft_map(inp, sp, self.show_.currentData())
        self.img.setImage(m.z, autoLevels=False)
        xe, ye = edges(m.x), edges(m.y)
        self.img.setRect(QtCore.QRectF(xe[0], ye[0], xe[-1] - xe[0], ye[-1] - ye[0]))
        self.cbar.setColorMap(_pg_cmap(self.cmap.currentData(), False))
        lo, hi = m.levels
        span = abs(hi - lo) or abs(hi) or 1.0
        self.cbar.rounding = 10.0 ** np.floor(np.log10(span / 1000.0))
        self.cbar.setLevels((lo, hi))
        self.mplot.setLabel("bottom", axis_title(m.x_name, m.x_unit))
        self.mplot.setLabel("left", axis_title(m.y_name, m.y_unit))
        self.mplot.setTitle(m.z_name, size="9pt")
        self._draw_peaks(inp)
        self._draw_model_on_map(inp, sp)
        self._draw_line()
        self._draw_dispersion()

    def _used(self, i: int) -> bool:
        return self.points[i].use if i < len(self.points) else True

    def _draw_peaks(self, inp):
        """Used peaks as circles, left-out ones as grey crosses."""
        used = [self._used(i) for i in range(len(self.peaks))]
        self.peak_scatter.setData(
            [p.k for p in self.peaks], [inp.y[p.line] for p in self.peaks],
            symbol=["o" if u else "x" for u in used],
            pen=[pg.mkPen(C["accent"] if u else C["muted"], width=1.2) for u in used],
            brush=[None] * len(used), size=[6 if u else 8 for u in used])

    def toggle_peak(self, i: int):
        """Leave peak i out of the dispersion -- or take it back."""
        if not 0 <= i < len(self.points):
            return
        p = self.points[i]
        p.use = not p.use
        self._fill_points()
        self._draw_peaks(self.current())
        self._draw_line()
        self._draw_dispersion()
        self.say(f"{p.label}, k = {p.k:.4g} rad/µm: {'used' if p.use else 'left out'}  "
                 f"({sum(q.use for q in self.points)} of {len(self.points)} peaks used)")

    def _nearest_peak(self, plot, scene_pos, xy_of, candidates, px: float = 8.0):
        """The peak index whose marker is within px pixels of a click, or None."""
        best, dist = None, px
        for i in candidates:
            x, y = xy_of(i)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            sp_ = plot.vb.mapViewToScene(QtCore.QPointF(x, y))
            d = float(np.hypot(sp_.x() - scene_pos.x(), sp_.y() - scene_pos.y()))
            if d < dist:
                best, dist = i, d
        return best

    def _draw_model_on_map(self, inp, sp):
        """The fitted f(k) on the FFT map -- when its y axis is the frequency."""
        self._clear_model()
        if self.result is None or self.f_src.currentData() != "y":
            return
        kind, to_ghz = I.unit_kind(inp.y_unit)
        if kind != "freq":
            return
        per = 2 * np.pi if sp.k_unit == "1/um" else 1.0      # the map's k -> rad/um
        kmax = float(np.nanmax(np.abs(sp.k))) * per
        B = self._const(self.b_const) if self.b_src.currentData() == "constant" else None
        k, f = R.model_curve(self.result, kmax, 600, B=B)
        item = pg.PlotDataItem(k / per, f / to_ghz, pen=pg.mkPen(C["ok"], width=2))
        self.mplot.addItem(item, ignoreBounds=True)       # the view stays on the data
        self.model_items.append(item)

    def _draw_line(self):
        inp, sp = self.current(), self.spectrum
        self.lplot.clear()
        if inp is None or sp is None:
            return
        i = min(max(self.line, 0), len(inp.y) - 1)
        mag = R.shown(sp, self.show_.currentData())[i]
        self.lplot.plot(sp.k, mag, pen=pg.mkPen(C["accent"], width=1.5))
        idx = [j for j, p in enumerate(self.peaks) if p.line == i]
        mine = [self.peaks[j] for j in idx]
        if mine:
            vals = [self._shown_amp(p) for p in mine]
            used = [self._used(j) for j in idx]
            self.lplot.addItem(pg.ScatterPlotItem(
                [p.k for p in mine], vals, size=[9 if u else 10 for u in used],
                symbol=["o" if u else "x" for u in used],
                pen=[pg.mkPen(C["text"] if u else C["muted"]) for u in used], brush=None))
        self.hline.setPos(inp.y[i])
        kn, ku = R.k_title(sp.k_unit)
        self.lplot.setLabel("bottom", axis_title(kn, ku))
        self.lplot.setLabel("left", R.SHOW_TEXT[self.show_.currentData()])
        txt = ", ".join(f"k = {p.k:.4g} (λ = {self._lam(p.k, sp):.3g} µm)" for p in mine)
        self.readout.setText(f"{inp.line_label(i)}:  {txt or 'no peak'}")

    def _shown_amp(self, p) -> float:
        show = self.show_.currentData()
        return (p.amplitude if show == "abs" else
                p.amplitude ** 2 if show == "power" else np.log10(p.amplitude))

    @staticmethod
    def _lam(k, sp):
        full = 2 * np.pi if sp.k_unit == "rad/um" else 1.0
        return full / abs(k) if k else float("nan")

    def _map_clicked(self, ev):
        """On a peak (map or line plot): leave it out / take it back. Elsewhere
        on the map: show that line below."""
        inp = self.current()
        if inp is None:
            return
        pos = ev.scenePos()
        if self.mplot.sceneBoundingRect().contains(pos):
            hit = self._nearest_peak(self.mplot, pos,
                                     lambda j: (self.peaks[j].k, inp.y[self.peaks[j].line]),
                                     range(len(self.peaks)))
            y = self.mplot.vb.mapSceneToView(pos).y()
            self.line = (self.peaks[hit].line if hit is not None
                         else int(np.nanargmin(np.abs(np.asarray(inp.y) - y))))
            if hit is not None:
                self.toggle_peak(hit)            # redraws the line too
            else:
                self._draw_line()
        elif self.lplot.sceneBoundingRect().contains(pos):
            idx = [j for j, p in enumerate(self.peaks) if p.line == self.line]
            hit = self._nearest_peak(self.lplot, pos,
                                     lambda j: (self.peaks[j].k,
                                                self._shown_amp(self.peaks[j])), idx)
            if hit is not None:
                self.toggle_peak(hit)

    # ── dispersion ─────────────────────────────────────────────────────────
    def _fill_sources(self, inp: I.Input):
        self._filling = True
        for combo, kind in ((self.f_src, "freq"), (self.b_src, "field")):
            combo.blockSignals(True)
            combo.clear()
            if I.unit_kind(inp.y_unit)[0] == kind:
                combo.addItem(f"the lines' {inp.y_name} ({inp.y_unit})", "y")
            for d, (v, u) in inp.held.items():
                if I.unit_kind(u)[0] == kind:
                    combo.addItem(f"{d} = {v:g} {u}", d)
            combo.addItem("typed value →", "constant")
            combo.setCurrentIndex(max(combo.findData(I.guess_source(inp, kind).how), 0))
            combo.blockSignals(False)
        self._filling = False
        self._const_enable()

    def _const_enable(self):
        self.f_const.setEnabled(self.f_src.currentData() == "constant")
        self.b_const.setEnabled(self.b_src.currentData() == "constant")

    @staticmethod
    def _const(edit) -> float:
        return _float(edit.text(), float("nan")) if edit.text().strip() else float("nan")

    def _sources_changed(self, *_):
        if self._filling:
            return
        self._const_enable()
        if self.unfold.currentData():        # the unfold needs each line's frequency
            self.transform()
        self._make_points()
        self._draw_dispersion()

    def _make_points(self):
        inp, sp = self.current(), self.spectrum
        if inp is None or sp is None:
            return
        fs = I.Source(self.f_src.currentData() or "constant", self._const(self.f_const))
        bs = I.Source(self.b_src.currentData() or "constant", self._const(self.b_const))
        per = 2 * np.pi if sp.k_unit == "1/um" else 1.0
        old = {(p.line, round(p.k, 6)): p.use for p in self.points}
        self.points = []
        for p in self.peaks:
            k = p.k * per
            self.points.append(W.Point(k=k, f=I.quantity(inp, p.line, fs),
                                       B=abs(I.quantity(inp, p.line, bs)), line=p.line,
                                       label=inp.line_label(p.line), amplitude=p.amplitude,
                                       use=old.get((p.line, round(k, 6)), True)))
        self._fill_points()

    def _fill_points(self):
        self._filling = True
        self.ptable.setRowCount(len(self.points))
        for r, p in enumerate(self.points):
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            use.setCheckState(QtCore.Qt.Checked if p.use else QtCore.Qt.Unchecked)
            self.ptable.setItem(r, 0, use)
            for c, text in enumerate([p.label, f"{p.k:.5g}", f"{p.f:.5g}", f"{p.B:.4g}",
                                      f"{p.amplitude:.3g}"], 1):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.ptable.setItem(r, c, it)
        self.ptable.resizeColumnsToContents()
        self._filling = False

    def _point_toggled(self, item):
        if self._filling or item.column() != 0:
            return
        self.points[item.row()].use = item.checkState() == QtCore.Qt.Checked
        self._draw_dispersion()

    def _fill_params(self):
        self._filling = True
        self.mtable.setRowCount(len(W.PARAMS))
        unused = self.pinning.currentData() == "none"
        for r, (n, p) in enumerate(W.PARAMS.items()):
            sp = self.specs[n]
            err = self.result.errors.get(n) if self.result else None
            off = unused and n in W.WIDTH_PARAMS    # no width mode: greyed, not fitted
            cells = [n, f"{sp.value:.6g}", "" if err is None else f"{err:.2g}", p[0], None,
                     "" if not np.isfinite(sp.min) else f"{sp.min:g}",
                     "" if not np.isfinite(sp.max) else f"{sp.max:g}"]
            for c, text in enumerate(cells):
                if c == 4:
                    it = QtWidgets.QTableWidgetItem()
                    it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    it.setCheckState(QtCore.Qt.Checked if sp.vary else QtCore.Qt.Unchecked)
                else:
                    it = QtWidgets.QTableWidgetItem(text)
                    if c in (0, 2, 3):
                        it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                if c == 0:
                    it.setToolTip(p[1] + ("  -- not used: width modes = none" if off else ""))
                if off:
                    it.setFlags(QtCore.Qt.ItemFlag(0))
                self.mtable.setItem(r, c, it)
        self.mtable.resizeColumnsToContents()
        self._filling = False

    def _param_edited(self, item):
        if self._filling:
            return
        n = list(W.PARAMS)[item.row()]
        sp = self.specs[n]
        c = item.column()
        if c == 1:
            sp.value = _float(item.text(), sp.value)
        elif c == 4:
            sp.vary = item.checkState() == QtCore.Qt.Checked
        elif c == 5:
            sp.min = _float(item.text(), -np.inf) if item.text().strip() else -np.inf
        elif c == 6:
            sp.max = _float(item.text(), np.inf) if item.text().strip() else np.inf

    def fit(self):
        if not self.points:
            self.say("find the peaks first (FFT tab)", error=True)
            return
        if any(not np.isfinite(p.f) for p in self.points if p.use):
            self.say("the points have no frequency: choose where f comes from, or type it",
                     error=True)
            return
        if self.b_src.currentData() == "constant" and not np.isfinite(self._const(self.b_const)):
            for p in self.points:
                p.B = float("nan")           # the parameter B then (fixed or fitted)
        res = self._guard("fit", lambda: W.fit(self.points, self.specs,
                                               self.pinning.currentData(),
                                               self.robust.isChecked()))
        if res is None:
            return
        self.result = res
        for n, v in res.values.items():
            self.specs[n].value = v
        self._fill_params()
        rows = res.rows()
        self.rtable.setRowCount(len(rows))
        for r, (n, v, e, unit, meaning) in enumerate(rows):
            for c, text in enumerate([n, f"{v:.6g}", "" if e is None else f"{e:.2g}", unit,
                                      meaning]):
                self.rtable.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self.rtable.resizeColumnsToContents()
        self._redraw()
        weak = [n for n, e in res.errors.items() if e is not None and res.vary[n]
                and abs(e) > abs(res.values[n])]
        msg = (f"{'fitted' if res.success else 'NOT converged'}: rms {res.rms * 1e3:.3g} MHz "
               f"from {res.n_points} points")
        if weak:
            msg += f".  Not determined by these points: {', '.join(weak)} (error > value)"
        self.say(msg, error=bool(weak) or not res.success)

    def _draw_dispersion(self):
        self.dplot.clear()
        used = [p for p in self.points if p.use]
        off = [p for p in self.points if not p.use]
        if used:
            self.dplot.plot([p.k for p in used], [p.f for p in used], pen=None, symbol="o",
                            symbolSize=6, symbolBrush=C["accent"], symbolPen=None, name="peaks")
        if off:
            self.dplot.plot([p.k for p in off], [p.f for p in off], pen=None, symbol="x",
                            symbolSize=7, symbolPen=C["muted"], name="left out")
        if self.result is not None and self.points:
            kmax = max(abs(p.k) for p in self.points) * 1.15
            Bs = [p.B for p in used if np.isfinite(p.B)]
            k, f = R.model_curve(self.result, kmax, B=Bs[0] if Bs else None)
            self.dplot.plot(k, f, pen=pg.mkPen(C["ok"], width=2),
                            name=f"Kalinikos-Slavin ({self.result.pinning})")

    # ── out ────────────────────────────────────────────────────────────────
    def _stem(self, what: str) -> str:
        inp = self.current()
        base = Path(inp.source).stem if inp and inp.source else "unsaved_run"
        return f"{base}_{what}"

    def _dir(self) -> Path:
        inp = self.current()
        for d in (self.last_dir, Path(inp.source).parent if inp and inp.source else None):
            if d and Path(d).is_dir():
                return Path(d)
        return Path.home()

    def fft_map(self):
        inp, sp = self.current(), self.spectrum
        if inp is None or sp is None:
            raise ValueError("no FFT yet -- send a map from AaltoView")
        return R.fft_map(inp, sp, self.show_.currentData())

    def peak_columns(self):
        inp, sp = self.current(), self.spectrum
        if inp is None or sp is None:
            raise ValueError("no FFT yet")
        return R.peak_columns(inp, sp, self.peaks, self.points if self.points else None)

    def _save_text(self, what, stem, filters, write):
        fn, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Save {what}", str(self._dir() / f"{self._stem(stem)}.csv"), filters)
        if not fn:
            return
        path = Path(fn)
        if path.suffix.lower() not in (".csv", ".dat", ".txt"):
            path = path.with_suffix(".dat" if "Tab" in chosen or "tab" in chosen else ".csv")
        self.last_dir = path.parent
        if self._guard("save", lambda: write(path, chosen)):
            self.say(f"saved {path}")

    def save_fft(self):
        m = self._guard("save", self.fft_map)
        if m is not None:
            self._save_text("FFT map", "fft", "Matrix, comma-separated (*.csv);;Matrix, "
                            "tab-separated (*.dat);;XYZ columns, comma-separated (*.csv)",
                            lambda p, ch: write_map(p, m, "xyz" if "XYZ" in ch else "matrix"))

    def _copy(self, write):
        with tempfile.TemporaryDirectory() as tmp:
            p = write(Path(tmp) / "c.dat")
            QtWidgets.QApplication.clipboard().setText(p.read_text(encoding="utf-8"))

    def copy_fft(self):
        m = self._guard("copy", self.fft_map)
        if m is not None:
            self._copy(lambda p: write_map(p, m))
            self.say("FFT map copied to the clipboard (tab-separated matrix)")

    def save_peaks(self):
        cols = self._guard("save", self.peak_columns)
        if cols is not None:
            self._save_text("peaks", "peaks", "Comma-separated (*.csv);;Tab-separated (*.dat)",
                            lambda p, ch: R.write_columns(p, cols))

    def copy_peaks(self):
        cols = self._guard("copy", self.peak_columns)
        if cols is not None:
            self._copy(lambda p: R.write_columns(p, cols))
            self.say("peaks copied to the clipboard (tab-separated)")

    def result_columns(self):
        if self.result is None:
            raise ValueError("nothing fitted yet -- press Fit")
        return R.result_columns(self.result)

    def copy_results(self):
        cols = self._guard("copy", self.result_columns)
        if cols is not None:
            self._copy(lambda p: R.write_columns(p, cols))
            self.say("results copied to the clipboard (tab-separated)")

    def save_results(self):
        cols = self._guard("save", self.result_columns)
        if cols is not None:
            self._save_text("results", "dispersion", "Comma-separated (*.csv);;Tab-separated "
                            "(*.dat)", lambda p, ch: R.write_columns(p, cols))

    def figure(self):
        inp, sp = self.current(), self.spectrum
        m = self.fft_map()
        peaks = ([p.k for p in self.peaks], [inp.y[p.line] for p in self.peaks])
        model = None
        if self.result is not None and self.f_src.currentData() == "y":
            kind, to_ghz = I.unit_kind(inp.y_unit)
            if kind == "freq":
                per = 2 * np.pi if sp.k_unit == "1/um" else 1.0
                B = self._const(self.b_const) if self.b_src.currentData() == "constant" else None
                k, f = R.model_curve(self.result, float(np.nanmax(np.abs(sp.k))) * per, B=B)
                model = (k / per, f / to_ghz)
        return R.figure_fft(m, peaks, model, title=inp.label)

    def _save_fig(self, make, stem):
        fig = self._guard("image", make)
        if fig is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", str(self._dir() / f"{self._stem(stem)}.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if fn:
            self.last_dir = Path(fn).parent
            if self._guard("image", lambda: save_figure(fig, fn)):
                self.say(f"saved {fn}")

    def save_image(self):
        self._save_fig(self.figure, "fft")

    def save_dispersion_image(self):
        self._save_fig(lambda: R.figure_dispersion(self.points, self.result), "dispersion")

    def fft_to_origin(self):
        m = self._guard("Origin", self.fft_map)
        if m is not None:
            self._origin(lambda path: __import__("aaltoview.origin", fromlist=["x"]).save_payload(
                path, m=m, name=self._stem("fft")))

    def results_to_origin(self):
        cols = self._guard("Origin", self.result_columns)
        if cols is not None:
            self._origin(lambda path: __import__("aaltoview.origin", fromlist=["x"]).save_payload(
                path, table=[c.__dict__ for c in cols], name=self._stem("dispersion")))

    def _origin(self, make_payload):
        """Same route as the viewer: a child process (see aaltoview/origin.py)."""
        from aaltoview.origin import available
        reason = available()
        if reason:
            self.say(f"Origin: {reason}", error=True)
            return
        if self._origin_proc is not None:
            self.say("Origin: still sending the previous one…")
            return
        tmp = Path(tempfile.mkdtemp(prefix="sw_fft_origin_"))
        payload = self._guard("Origin", lambda: make_payload(tmp / "payload.npz"))
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
