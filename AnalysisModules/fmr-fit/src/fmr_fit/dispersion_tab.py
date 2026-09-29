"""dispersion_tab.py -- the Dispersion tab: resonances -> material parameters.

    +------------------------------------+-----------------------------------+
    | POINTS  (one per fitted peak)      |  resonance vs angle / Kittel plot |
    |  use | curve | role | x0 | FWHM ...|  + model (uniform, PSSW n)        |
    | COORDINATES  field / f / angles    |  FWHM + damping model             |
    | MODEL  uniaxial 4-fold 6-fold PSSW +-----------------------------------+
    | PARAMETERS  start, fixed, bounds   |  RESULTS  value ± error           |
    | Fit | Reset start values           |  Copy | Save… | Origin | Image    |
    +------------------------------------+-----------------------------------+

The maths is dispersion.py; this is the widgets. The points come from the
Resonances tab's fits; a peak's ROLE (uniform, PSSW n, ignore) is set here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from aaltoview.apps.theme import C
from aaltoview.apps.viewer import TAB10, _float, _plain_axes
from aaltoview.export import save_figure

from . import dispersion as D
from . import model as M

ROLE_CHOICES = [(D.ROLE_UNIFORM, "uniform (Kittel)")] + \
    [(n, f"PSSW n = {n}") for n in (1, 2, 3, 4)] + [(D.ROLE_IGNORE, "ignore")]
PRETTY = {"gamma": "γ/2π", "g": "g-factor", "Meff": "μ0 M_eff", "Bu": "B_u (uniaxial)",
          "phi_u": "φ_u (uniaxial axis)", "B4": "B_4 (4-fold)", "phi_4": "φ_4 (4-fold axis)",
          "B6": "B_6 (6-fold)", "phi_6": "φ_6 (6-fold axis)", "A": "A (exchange stiffness)",
          "alpha": "α (Gilbert damping)", "dH0": "ΔH0 (inhomogeneous FWHM)"}
ROLE_COLORS = {D.ROLE_UNIFORM: TAB10[0], 1: TAB10[3], 2: TAB10[2], 3: TAB10[4], 4: TAB10[1]}


def pretty(name: str) -> str:
    if name.startswith("Hex"):
        return f"μ0 H_ex, PSSW n = {name[3:]}"
    if name.startswith("A") and name[1:].isdigit():
        return f"A from PSSW n = {name[1:]}"
    return PRETTY.get(name, name)


class DispersionTab(QtWidgets.QWidget):
    PCOLS = ("Use", "Curve", "Role", "Sweep", "x0", "FWHM", "μ0H (mT)", "f (GHz)",
             "φ_H (°)", "θ_H (°)")
    COLS = ("Parameter", "Value", "±", "Unit", "Fixed", "Min", "Max")

    def __init__(self, owner):
        super().__init__()
        self.owner = owner                       # the FitWindow: entries and their fits
        self.roles: dict[tuple[int, int], int] = {}     # (id(entry), peak) -> role
        self.unused: set[tuple[int, int]] = set()
        self.sources: dict | None = None
        self.specs: dict[str, M.Spec] | None = None
        self.points: list[D.Point] = []
        self.entry_ids: list[int] = []
        self.dres: D.DResult | None = None
        self.damp: D.DampResult | None = None
        self._filling = False
        self._origin_proc = None

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.addWidget(split)

        # ── left ───────────────────────────────────────────────────────────
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(6)
        lv.addWidget(self._tag("POINTS  (one per fitted peak, from the Resonances tab)"))
        self.ptable = QtWidgets.QTableWidget(0, len(self.PCOLS))
        self.ptable.setHorizontalHeaderLabels(self.PCOLS)
        self.ptable.verticalHeader().setVisible(False)
        self.ptable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.ptable.setToolTip("Untick a point to leave it out. Role: which mode the peak is "
                               "-- flag a second peak as PSSW n = 1, 2, ...")
        self.ptable.itemChanged.connect(self._point_edited)
        lv.addWidget(self.ptable, 3)
        pb = QtWidgets.QHBoxLayout()
        for text, use in (("Use all", True), ("Use none", False)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, u=use: self._use_all(u))
            pb.addWidget(b)
        self.order_btn = QtWidgets.QPushButton("Assign PSSW by order")
        self.order_btn.setToolTip(
            "Per curve: the strongest peak = uniform; the peaks on the PSSW side (lower\n"
            "field in a field sweep, higher frequency in a frequency sweep) = PSSW n = 1,\n"
            "2, ... by their distance from it; others ignored. Assumes no mode is missing\n"
            "in between -- check the plot: each PSSW order should lie on its own line.")
        self.order_btn.clicked.connect(self.assign_pssw)
        pb.addWidget(self.order_btn)
        pb.addStretch(1)
        lv.addLayout(pb)

        lv.addWidget(self._tag("COORDINATES  (where each resonance was measured)"))
        self.coord_rows: dict[str, tuple[QtWidgets.QComboBox, QtWidgets.QLineEdit]] = {}
        cg = QtWidgets.QGridLayout()
        for r, (k, text) in enumerate((("H", "field μ0H (mT)"), ("f", "frequency f (GHz)"),
                                       ("phi", "in-plane angle φ_H (°)"),
                                       ("theta", "polar angle θ_H (°)"))):
            cg.addWidget(QtWidgets.QLabel(text), r, 0)
            combo = QtWidgets.QComboBox()
            edit = QtWidgets.QLineEdit()
            edit.setFixedWidth(80)
            edit.setToolTip("With 'constant': the value for every curve.\n"
                            "With a dimension: the value for curves whose file does NOT have "
                            "that dimension\n(a field sweep at one fixed angle next to an "
                            "angle series). Empty = leave such curves out.")
            combo.currentIndexChanged.connect(self._coords_changed)
            edit.editingFinished.connect(self._coords_changed)
            cg.addWidget(combo, r, 1)
            cg.addWidget(edit, r, 2)
            self.coord_rows[k] = (combo, edit)
        self.elev = QtWidgets.QCheckBox("polar angle measured from the PLANE (elevation)")
        self.elev.setToolTip("Ticked: θ_H = 90° − the value. Unticked: from the film normal "
                             "(90° = in-plane).")
        self.elev.toggled.connect(self._coords_changed)
        cg.addWidget(self.elev, 4, 0, 1, 3)
        lv.addLayout(cg)

        lv.addWidget(self._tag("MODEL"))
        mg = QtWidgets.QGridLayout()
        self.uni = QtWidgets.QCheckBox("in-plane uniaxial")
        self.four = QtWidgets.QCheckBox("4-fold")
        self.six = QtWidgets.QCheckBox("6-fold")
        for i, w in enumerate((self.uni, self.four, self.six)):
            w.setToolTip("Unticked = 0. Perpendicular anisotropy is always in, through M_eff.")
            w.toggled.connect(self._model_changed)
            mg.addWidget(w, 0, i)
        mg.addWidget(QtWidgets.QLabel("PSSW"), 1, 0)
        self.pssw = QtWidgets.QComboBox()
        self.pssw.addItem("exchange field per mode", "free")
        self.pssw.addItem("exchange stiffness A", "A")
        self.pssw.setToolTip("A: H_ex,n = 2A (nπ/d)² / Ms, unpinned surfaces. Needs d and "
                             "μ0Ms (NOT M_eff).\nWith 'per mode', A is still shown per mode "
                             "when d and μ0Ms are given.")
        self.pssw.currentIndexChanged.connect(self._model_changed)
        mg.addWidget(self.pssw, 1, 1, 1, 2)
        mg.addWidget(QtWidgets.QLabel("d (nm)"), 2, 0)
        self.d_edit = QtWidgets.QLineEdit()
        mg.addWidget(self.d_edit, 2, 1)
        mg.addWidget(QtWidgets.QLabel("μ0Ms (mT)"), 3, 0)
        self.ms_edit = QtWidgets.QLineEdit()
        mg.addWidget(self.ms_edit, 3, 1)
        for w in (self.d_edit, self.ms_edit):
            w.editingFinished.connect(self._model_changed)
        lv.addLayout(mg)

        lv.addWidget(self._tag("PARAMETERS  (start values; Fixed = held)"))
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i in range(1, len(self.COLS)):
            hh.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setToolTip("γ and M_eff are only separable with more than one frequency\n"
                              "(or field): with one, fix γ (g = 2: 27.99 GHz/T).\n"
                              "Fixing an axis angle fits the strength along that axis.")
        self.table.itemChanged.connect(self._param_edited)
        lv.addWidget(self.table, 3)
        fb = QtWidgets.QHBoxLayout()
        self.fit_btn = QtWidgets.QPushButton("Fit"); self.fit_btn.setObjectName("primary")
        self.fit_btn.setToolTip("Positions first (γ, M_eff, anisotropy, PSSW), then the "
                                "linewidths of the uniform mode (α, ΔH0).")
        self.fit_btn.clicked.connect(self.fit)
        self.reset_btn = QtWidgets.QPushButton("Reset start values")
        self.reset_btn.clicked.connect(self.reset_specs)
        fb.addWidget(self.fit_btn); fb.addWidget(self.reset_btn)
        lv.addLayout(fb)
        split.addWidget(left)

        # ── right ──────────────────────────────────────────────────────────
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        rv.setSpacing(6)
        self.glw = pg.GraphicsLayoutWidget()
        self.pplot = self.glw.addPlot(row=0, col=0)
        self.plegend = self.pplot.addLegend(offset=(-10, 10))
        self.wplot = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        _plain_axes(self.pplot)
        _plain_axes(self.wplot)
        rv.addWidget(self.glw, 3)
        rv.addWidget(self._tag("RESULTS"))
        self.results = QtWidgets.QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(["Parameter", "Value", "±", "Unit", ""])
        self.results.verticalHeader().setVisible(False)
        self.results.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        rv.addWidget(self.results, 2)
        eb = QtWidgets.QHBoxLayout()
        for text, fn, tip in (
                ("Copy results", self.copy_results, "Parameters, tab-separated, to the clipboard."),
                ("Save results…", self.save_results, "Parameters as .csv / .dat."),
                ("Send to Origin", self.send_to_origin, "Parameters into a running Origin."),
                ("Save image…", self.save_image, "Both plots: PNG, PDF or SVG.")):
            b = QtWidgets.QPushButton(text); b.setToolTip(tip); b.clicked.connect(fn)
            eb.addWidget(b)
        eb.addStretch(1)
        rv.addLayout(eb)
        self.status = QtWidgets.QLabel("Fit resonances first (Resonances tab), then come back.")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([640, 900])

    @staticmethod
    def _tag(text):
        t = QtWidgets.QLabel(text); t.setObjectName("tag")
        return t

    def say(self, text: str, error: bool = False):
        self.status.setStyleSheet(f"color:{C['danger'] if error else C['muted']};"
                                  " font-size:11px;")
        self.status.setText(text)

    # ── the points ─────────────────────────────────────────────────────────
    def _rows(self):
        entries = [e for e in self.owner.entries if e.result is not None]
        return entries, [(e.curve, e.result) for e in entries]

    def refresh(self):
        """Rebuild the points from the Resonances tab's current fits."""
        entries, rows = self._rows()
        self.entry_ids = [id(e) for e in entries]
        if not rows:
            self.points = []
            self._fill_points()
            self._redraw()
            self.say("Fit resonances first (Resonances tab), then come back.")
            return
        guessed = D.guess_sources(rows)
        self._fill_coord_widgets(rows, self.sources or guessed)
        sources = self._read_sources() if self.sources is not None else guessed
        roles = {(i, k): self.roles[(eid, k)] for i, eid in enumerate(self.entry_ids)
                 for k in range(1, 7) if (eid, k) in self.roles}
        notes: list[str] = []
        self.points, problems = D.points_from_fits(rows, sources, roles, notes)
        for p in self.points:
            p.use = self._key(p) not in self.unused
        self._fill_points()
        if self.specs is None:
            self.reset_specs(quiet=True)
        else:
            self._fill_params()
        self._redraw()
        n = sum(p.use and p.role != D.ROLE_IGNORE for p in self.points)
        msg = f"{len(self.points)} resonances, {n} used"
        if notes:
            msg += "; used " + "; ".join(notes)
        if self.specs is not None and not self.specs["gamma"].vary and D.one_kittel_point(
                self.points):
            msg += "; γ fixed (one frequency: γ and M_eff are not separable)"
        if problems:
            msg += " -- left out: " + "; ".join(problems[:3]) + (" …" if len(problems) > 3 else "")
        self.say(msg, error=bool(problems))

    def _key(self, p: D.Point) -> tuple[int, int]:
        return self.entry_ids[p.key[0]], p.key[1]

    def _fill_points(self):
        self._filling = True
        try:
            # emptied first: a role drop-down of the old rows was left standing
            # over the new first row
            self.ptable.setRowCount(0)
            self.ptable.setRowCount(len(self.points))
            for r, p in enumerate(self.points):
                unit = D.INTERNAL[p.swept]
                cells = ["", p.label, "", {"H": "field", "f": "frequency", "phi": "φ_H",
                                           "theta": "θ_H"}[p.swept],
                         f"{M.fmt_pm(getattr(p, p.swept), p.err)} {unit}",
                         f"{M.fmt_pm(p.fwhm, p.fwhm_err)} {unit}",
                         f"{p.H:.4g}", f"{p.f:.4g}", f"{p.phi:.4g}", f"{p.theta:.4g}"]
                for c, text in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(text)
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    if c == 0:
                        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                        it.setCheckState(QtCore.Qt.Checked if p.use else QtCore.Qt.Unchecked)
                    self.ptable.setItem(r, c, it)
                combo = QtWidgets.QComboBox()
                for role, text in ROLE_CHOICES:
                    combo.addItem(text, role)
                combo.setCurrentIndex(combo.findData(p.role))
                combo.currentIndexChanged.connect(lambda _=0, row=r, cb=combo:
                                                  self._role_changed(row, cb.currentData()))
                self.ptable.setCellWidget(r, 2, combo)
        finally:
            self._filling = False

    def _point_edited(self, item):
        if self._filling or item.column() != 0:
            return
        p = self.points[item.row()]
        p.use = item.checkState() == QtCore.Qt.Checked
        (self.unused.discard if p.use else self.unused.add)(self._key(p))
        self._redraw()

    def _use_all(self, use: bool):
        for p in self.points:
            p.use = use
            (self.unused.discard if use else self.unused.add)(self._key(p))
        self._fill_points()
        self._redraw()

    def assign_pssw(self):
        entries, rows = self._rows()
        if not rows:
            return
        sources = self._read_sources() if self.sources is not None else D.guess_sources(rows)
        by_index = D.pssw_roles(rows, sources)
        for (i, k), role in by_index.items():
            self.roles[(id(entries[i]), k)] = role
        self.refresh()
        orders = D.pssw_orders(self.points)
        if self.specs is not None:
            fresh = D.default_specs(self.points, self._settings())
            for n in orders:                    # start values from where they sit
                if self.dres is None or f"Hex{n}" not in self.dres.values:
                    self.specs[f"Hex{n}"] = fresh[f"Hex{n}"]
        self._fill_params()
        self.say(f"roles by order: uniform + PSSW n = {', '.join(map(str, orders)) or '-'}"
                 " -- check that each order lies on its own line")

    def _role_changed(self, row: int, role: int):
        p = self.points[row]
        p.role = role
        self.roles[self._key(p)] = role
        if role > 0 and self.specs is not None and f"Hex{role}" in self.specs:
            # a newly flagged PSSW: its exchange field guessed from where it sits
            guess = D.default_specs(self.points, self._settings())[f"Hex{role}"]
            if self.dres is None or f"Hex{role}" not in self.dres.values:
                self.specs[f"Hex{role}"] = guess
        self._fill_params()
        self._redraw()

    # ── coordinates ────────────────────────────────────────────────────────
    def _fill_coord_widgets(self, rows, sources):
        dims = []
        for c, _ in rows:
            for d in [c.x_name] + list(c.held):
                if d not in dims:
                    dims.append(d)
        self._filling = True
        try:
            for k, (combo, edit) in self.coord_rows.items():
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(dims + ["constant"])
                src = sources.get(k) or (sources.get("elev") if k == "theta" else None)
                if src and src[0] == "dim":
                    combo.setCurrentText(src[1])
                    fb = src[2] if len(src) > 2 else None
                    edit.setText("" if fb is None else f"{fb:g}")
                    edit.setPlaceholderText("if missing")
                else:
                    combo.setCurrentText("constant")
                    edit.setText(f"{src[1]:g}" if src else "0")
                    edit.setPlaceholderText("")
                combo.blockSignals(False)
            self.elev.blockSignals(True)
            self.elev.setChecked("elev" in sources)
            self.elev.blockSignals(False)
        finally:
            self._filling = False

    def _read_sources(self) -> dict:
        out = {}
        for k, (combo, edit) in self.coord_rows.items():
            key = "elev" if (k == "theta" and self.elev.isChecked()) else k
            if combo.currentText() == "constant":
                out[key] = ("const", _float(edit.text(), 90.0 if k == "theta" else 0.0))
            else:
                t = edit.text().strip()
                out[key] = ("dim", combo.currentText(), _float(t, 0.0) if t else None)
        return out

    def _coords_changed(self, *_):
        if self._filling:
            return
        for k, (combo, edit) in self.coord_rows.items():
            edit.setPlaceholderText("" if combo.currentText() == "constant" else "if missing")
        self.sources = self._read_sources()
        self.dres = self.damp = None
        self.refresh()

    # ── model and parameters ───────────────────────────────────────────────
    def _settings(self) -> D.Settings:
        d = _float(self.d_edit.text(), 0.0) or None
        ms = _float(self.ms_edit.text(), 0.0) or None
        return D.Settings(uniaxial=self.uni.isChecked(), fourfold=self.four.isChecked(),
                          sixfold=self.six.isChecked(), pssw=self.pssw.currentData(),
                          d_nm=d, Ms_mT=ms)

    def _names(self) -> list[str]:
        s = self._settings()
        names = ["gamma", "Meff"]
        for key, on in (("u", s.uniaxial), ("4", s.fourfold), ("6", s.sixfold)):
            if on:
                names += [f"B{key}", f"phi_{key}"]
        orders = D.pssw_orders(self.points)
        if orders:
            names += ["A"] if D._use_A(s) else [f"Hex{n}" for n in orders]
        return names + ["alpha", "dH0"]

    def reset_specs(self, quiet=False):
        self.specs = D.default_specs(self.points, self._settings()) if self.points else None
        self.dres = self.damp = None
        self._fill_params()
        self._fill_results()
        self._redraw()
        fixed = self.specs is not None and not self.specs["gamma"].vary
        if fixed:
            self.say("γ FIXED at g = 2.0023: all resonances are at one frequency (or one "
                     "field), so γ and M_eff cannot both be fitted. Untick Fixed if the data "
                     "has more.")
        elif not quiet:
            self.say("start values from the data (γ of g = 2.0023, M_eff from the in-plane "
                     "Kittel formula, anisotropy 0)")

    def _model_changed(self, *_):
        if self.specs is not None:
            fresh = D.default_specs(self.points, self._settings())
            for k, v in fresh.items():
                self.specs.setdefault(k, v)
        self._fill_params()

    def _fill_params(self):
        self._filling = True
        try:
            self.table.setRowCount(0)
            if self.specs is None:
                return
            names = self._names()
            self.table.setRowCount(len(names))
            for r, n in enumerate(names):
                sp = self.specs[n]
                val, err = sp.value, None
                if self.dres is not None and n in self.dres.values:
                    val, err = self.dres.values[n], self.dres.errors.get(n)
                if self.damp is not None and n in ("alpha", "dH0"):
                    val = getattr(self.damp, n)
                    err = getattr(self.damp, f"{n}_err")
                cells = [pretty(n), f"{val:.6g}", "" if err is None else f"{err:.2g}",
                         D.UNITS.get(n, "mT" if n.startswith("Hex") else ""), "",
                         "" if not np.isfinite(sp.min) else f"{sp.min:.6g}",
                         "" if not np.isfinite(sp.max) else f"{sp.max:.6g}"]
                for c, text in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(text)
                    if c in (0, 2, 3):
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    if c == 0:
                        it.setData(QtCore.Qt.UserRole, n)
                    if c == 4:
                        it.setFlags((it.flags() | QtCore.Qt.ItemIsUserCheckable)
                                    & ~QtCore.Qt.ItemIsEditable)
                        it.setCheckState(QtCore.Qt.Unchecked if sp.vary else QtCore.Qt.Checked)
                    self.table.setItem(r, c, it)
        finally:
            self._filling = False

    def _param_edited(self, item):
        if self._filling or self.specs is None:
            return
        n = self.table.item(item.row(), 0).data(QtCore.Qt.UserRole)
        sp = self.specs[n]
        t, col = item.text(), item.column()
        if col == 1:
            sp.value = _float(t, sp.value)
        elif col == 4:
            sp.vary = item.checkState() != QtCore.Qt.Checked
        elif col == 5:
            sp.min = _float(t, -np.inf) if t.strip() else -np.inf
        elif col == 6:
            sp.max = _float(t, np.inf) if t.strip() else np.inf

    # ── fitting ────────────────────────────────────────────────────────────
    def fit(self):
        if not self.points:
            self.say("no resonances -- fit curves in the Resonances tab first", error=True)
            return
        s = self._settings()
        if s.pssw == "A" and not (s.d_nm and s.Ms_mT):
            self.say("exchange stiffness A needs d and μ0Ms", error=True)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.dres = D.fit_positions(self.points, s, self.specs)
            try:
                self.damp = D.fit_damping(self.points, self.dres,
                                          {k: self.specs[k] for k in ("alpha", "dH0")})
                note = ""
            except ValueError as exc:
                self.damp, note = None, f" (no damping: {exc})"
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.say(f"fit failed: {exc}", error=True)
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        # the result becomes the next start, keeping fixed / bounds
        for n, v in self.dres.values.items():
            if n in self.specs:
                self.specs[n].value = v
        if self.damp is not None:
            self.specs["alpha"].value, self.specs["dH0"].value = self.damp.alpha, self.damp.dH0
        self._fill_params()
        self._fill_results()
        self._redraw()
        weak = [pretty(n) for n, *_ in D.result_rows(self.dres)
                if self.dres.errors.get(n) is not None and self.dres.vary.get(n)
                and abs(self.dres.errors[n]) > abs(self.dres.values[n])
                and not n.startswith("phi")]
        msg = (f"{'fitted' if self.dres.success else 'NOT converged'}: χ²_red = "
               f"{self.dres.redchi:.3g} from {self.dres.ndata} resonances{note}")
        if weak:
            msg += f".  Not determined by these data: {', '.join(weak)} (error > value)"
        self.say(msg, error=bool(weak) or not self.dres.success)

    def _fill_results(self):
        rows = D.result_rows(self.dres, self.damp) if self.dres else []
        self.results.setRowCount(len(rows))
        for r, (n, v, e, unit, meaning) in enumerate(rows):
            for c, text in enumerate([pretty(n), f"{v:.6g}", "" if e is None else f"{e:.2g}",
                                      unit, meaning]):
                self.results.setItem(r, c, QtWidgets.QTableWidgetItem(text))

    # ── drawing ────────────────────────────────────────────────────────────
    def _redraw(self):
        for plot in (self.pplot, self.wplot):
            plot.clear()
        self.plegend.clear()
        if not self.points:
            return
        pos = D.position_plot(self.points, self.dres)
        self._draw(self.pplot, pos, legend=True)
        wid = D.width_plot(self.points, self.dres, self.damp)
        self.wplot.setVisible(wid is not None)
        if wid is not None:
            self._draw(self.wplot, wid, legend=False)

    def _draw(self, plot, p: D.Plot, legend: bool):
        for s in p.data:
            col = ROLE_COLORS.get(s.role, TAB10[5])
            item = pg.ScatterPlotItem(s.x, s.y, size=7, pen=None,
                                      brush=pg.mkBrush(QtGui.QColor(col)))
            plot.addItem(item)
            if s.yerr is not None and np.isfinite(s.yerr).any():
                e = np.where(np.isfinite(s.yerr), s.yerr, 0.0)
                plot.addItem(pg.ErrorBarItem(x=s.x, y=s.y, height=2 * e,
                                             pen=pg.mkPen(col, width=1)))
            if legend:
                self.plegend.addItem(item, s.label)
        for s in p.lines:
            col = ROLE_COLORS.get(s.role, C["accent"]) if legend else C["accent"]
            plot.addItem(pg.PlotDataItem(s.x, s.y, pen=pg.mkPen(col, width=2),
                                         connect="finite"))
        plot.setLabel("bottom", p.x_title)
        plot.setLabel("left", p.y_title)

    # ── results out ────────────────────────────────────────────────────────
    def columns(self) -> list[M.Column]:
        if self.dres is None:
            raise ValueError("nothing fitted yet -- press Fit")
        rows = D.result_rows(self.dres, self.damp)
        return [M.Column("parameter", "", "", [r[0] for r in rows], kind="L"),
                M.Column("value", "", "", [r[1] for r in rows]),
                M.Column("error", "", "1 sigma",
                         [np.nan if r[2] is None else r[2] for r in rows], kind="E"),
                M.Column("unit", "", "", [r[3] for r in rows], kind="L"),
                M.Column("meaning", "", "", [r[4] for r in rows], kind="L")]

    def _guard(self, what, fn):
        try:
            return fn()
        except Exception as exc:
            self.say(f"{what} failed: {exc}", error=True)
            return None

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
        base = self.owner._dir() / f"{self.owner._stem()}_dispersion.csv"
        fn, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save results", str(base), "Comma-separated (*.csv);;Tab-separated (*.dat)")
        if not fn:
            return
        path = Path(fn)
        if path.suffix.lower() not in (".csv", ".dat", ".txt"):
            path = path.with_suffix(".dat" if "Tab" in chosen else ".csv")
        if self._guard("save", lambda: M.write_results(path, cols)):
            self.say(f"saved {path}")

    def figure(self):
        if not self.points:
            raise ValueError("nothing to draw")
        pos = D.position_plot(self.points, self.dres)
        return D.figure_dispersion(pos, D.width_plot(self.points, self.dres, self.damp))

    def save_image(self):
        fig = self._guard("image", self.figure)
        if fig is None:
            return
        base = self.owner._dir() / f"{self.owner._stem()}_dispersion.png"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", str(base),
                                                      "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if fn and self._guard("image", lambda: save_figure(fig, fn)):
            self.say(f"saved {fn}")

    def send_to_origin(self):
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
        tmp = Path(tempfile.mkdtemp(prefix="fmr_dispersion_origin_"))
        payload = self._guard("Origin", lambda: save_payload(
            tmp / "payload.npz", table=[c.__dict__ for c in cols],
            name=self.owner._stem() + "_dispersion"))
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
        self.say(f"Origin: sent as {last[3:]}" if last.startswith("OK")
                 else f"Origin: {last.removeprefix('ERROR ')}", error=not last.startswith("OK"))
