"""viewer.py -- the AaltoView, successor of the LabVIEW AaltoView.

Needs no instruments: it reads measurement files (.nc, as every scan autosaves
them) and nothing else.

    +-------------------+---------------------------------------------------+
    | FILES             |  Map | 1D plots                                   |
    | data folder       |                                                   |
    | name  measured    |  show [detector] [|z|]   X [..]  Y [..]           |
    | axes  shape       |  one row per other dim: hold at / mean all / mean |
    | ...               |  colour map, limits, symmetric, log, line norm    |
    |                   |  the picture + cursor + "row/column -> 1D"        |
    | selected file's   |  Save image | Copy image | Copy data | Save data  |
    | metadata          |  Send to Origin | Notebook                        |
    +-------------------+---------------------------------------------------+

What is the same as AaltoView, and what is not:
* The loop names are gone. AaltoView knew "topmost loop" and "loop level 1"
  and special-cased Position XY; a scan here is N named dims and the controls
  are built from whatever the file has.
* "Add plot" with the parameter list is `1D plots`: pick the dim to step
  through, select several values, Add selected. Curves are FROZEN copies with
  their origin attached, so curves from different files overlay (5 um away
  against 2 um away).
* Export to Origin is a live push over COM (aaltoview/origin.py), plus plain
  files. New: a Jupyter notebook that recomputes the view from the files.
* Not yet: the TR-MOKE corrections (laser repetition rate, harmonic, demod
  frequency folding, correction file, phase autocorrect). Deliberately left out
  until the maths is written down -- guessing at them would give wrong numbers
  that look right.

The arithmetic is not here: view.py reduces the cube, export.py builds
curves/maps and writes files, origin.py talks to Origin. All three are tested without a screen.

Run:
    uv run aaltoview [file.nc] [--folder DIR] [--theme light]
    (or: python -m aaltoview)
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import xarray as xr
from PySide6 import QtCore, QtGui, QtWidgets

from .. import export as E
from .. import view as V
from ..data import find_measurements, load, summarize
from .theme import C, DEFAULT_THEME, apply, apply_window_icon, set_theme
from .widgets import DimRow
NONE_TEXT = "— none —"
PART_TEXT = {"|z|": "abs", "arg z": "arg", "Re z": "real", "Im z": "imag"}
#: matplotlib's default colour cycle, so a curve has the same colour on screen,
#: in a saved figure and in the notebook.
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _float(text: str, default: float) -> float:
    """Parse a typed number with float(), NOT a Qt validator: under the lab PC's
    locale a validator expects a decimal comma (gotcha #18)."""
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        return default


def _pg_cmap(name: str, invert: bool) -> pg.ColorMap:
    cmap = pg.colormap.get(E.CMAPS.get(name, name), source="matplotlib")
    if invert:
        cmap.reverse()
    return cmap


def _plain_axes(plot: pg.PlotItem, *extra: pg.AxisItem) -> None:
    """No automatic SI prefixes. pyqtgraph would relabel a volt axis
    "V (x0.001)" and a frequency already in MHz "kMHz"; the numbers should be
    the ones in the file, in the unit the file states."""
    for ax in [plot.getAxis(n) for n in ("bottom", "left")] + list(extra):
        ax.enableAutoSIPrefix(False)


# ─────────────────────────────── the file list ────────────────────────────────

class FileBrowser(QtWidgets.QWidget):
    """A data folder, its measurements newest first, and the selected one's header."""

    fileChosen = QtCore.Signal(str)
    COLS = ("Measurement", "Measured", "Axes (outer → inner)", "Shape", "Detectors")

    def __init__(self, folder: Path | None = None):
        super().__init__()
        self.folder = Path(folder) if folder else None
        self._cache: dict[Path, tuple[float, object]] = {}    # path -> (mtime, Summary)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        tag = QtWidgets.QLabel("FILES"); tag.setObjectName("tag")
        v.addWidget(tag)

        row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit(str(self.folder or ""))
        self.folder_edit.setToolTip("The data folder. Sub-folders (one per day) are included.")
        self.folder_edit.editingFinished.connect(
            lambda: self.set_folder(self.folder_edit.text(), remember=True))
        row.addWidget(self.folder_edit, 1)
        browse = QtWidgets.QPushButton("…"); browse.setFixedWidth(32)
        browse.setToolTip("Choose the data folder")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        v.addLayout(row)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("filter: name, axis or detector…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        v.addWidget(self.filter_edit)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(self.COLS)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemActivated.connect(self._activated)          # double-click / Enter
        self.tree.currentItemChanged.connect(self._show_header)
        v.addWidget(self.tree, 3)

        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setPlaceholderText("Select a file to see what is in it. "
                                     "Double-click to open it.")
        v.addWidget(self.info, 2)

    # ---- folder -----------------------------------------------------------
    def set_folder(self, folder, refresh: bool = True, remember: bool = False) -> None:
        folder = Path(str(folder).strip()) if str(folder).strip() else None
        self.folder = folder
        # remember only a folder the OPERATOR chose (Browse, typing), never one
        # a host or a test set -- that would overwrite the operator's choice
        if remember and folder is not None and folder.is_dir():
            remember_folder(folder)
        self.folder_edit.setText(str(folder or ""))
        if refresh:
            self.refresh()

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder", str(self.folder or ""))
        if d:
            self.set_folder(d, remember=True)

    def refresh(self) -> None:
        """Re-list the folder. Headers are cached by modification time, so a
        refresh of a folder with hundreds of scans only opens the new ones."""
        self.tree.clear()
        if self.folder is None or not self.folder.is_dir():
            self.info.setPlainText(f"No such folder: {self.folder}" if self.folder else "")
            return
        for p in find_measurements(self.folder):
            mtime = p.stat().st_mtime
            hit = self._cache.get(p)
            if hit is None or hit[0] != mtime:
                hit = (mtime, summarize(p))
                self._cache[p] = hit
            s = hit[1]
            try:
                rel = p.relative_to(self.folder)
            except ValueError:
                rel = p
            item = QtWidgets.QTreeWidgetItem([
                s.name, s.measured.strftime("%Y-%m-%d %H:%M:%S"),
                "  ›  ".join(s.dims) if not s.error else "unreadable",
                s.shape_text, ", ".join(s.detectors)])
            item.setToolTip(0, str(rel))
            item.setData(0, QtCore.Qt.UserRole, str(p))
            if s.error:
                item.setForeground(2, QtGui.QColor(C["danger"]))
            self.tree.addTopLevelItem(item)
        for col in range(len(self.COLS) - 1):
            self.tree.resizeColumnToContents(col)
        self._apply_filter()

    def _apply_filter(self):
        words = self.filter_edit.text().lower().split()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            hay = " ".join(it.text(c) for c in range(len(self.COLS))).lower()
            it.setHidden(not all(w in hay for w in words))

    def _activated(self, item, _col=0):
        self.fileChosen.emit(item.data(0, QtCore.Qt.UserRole))

    def _show_header(self, item, _prev=None):
        if item is None:
            return
        p = Path(item.data(0, QtCore.Qt.UserRole))
        s = self._cache.get(p, (0, summarize(p)))[1]
        if s.error:
            self.info.setPlainText(f"{p}\n\nCould not read this file:\n{s.error}")
            return
        lines = [str(p), "", f"name       {s.name}",
                 f"measured   {s.measured:%Y-%m-%d %H:%M:%S}"]
        if s.comment:
            lines.append(f"comment    {s.comment}")
        if s.n_points is not None:
            dur = f", {s.seconds / 60:.1f} min" if s.seconds else ""
            lines.append(f"points     {s.n_points}{dur}")
        lines += ["", "axes (outer -> inner)"]
        lines += [f"  {d}  ({s.sizes[d]})" for d in s.dims]
        lines += ["", "detectors"] + [f"  {d}" for d in s.detectors]
        self.info.setPlainText("\n".join(lines))


# ───────────────────────── choosing a piece of the cube ───────────────────────

class CubeControls(QtWidgets.QWidget):
    """Detector (+ complex part), X, optionally Y, and one row per other dim.

    The same controls drive the map (with Y) and the 1-D plot (without). Rows
    are rebuilt only when the set of leftover dims changes, so a slider being
    dragged is never replaced under the mouse.
    """

    changed = QtCore.Signal()

    def __init__(self, with_y: bool):
        super().__init__()
        self.with_y = with_y
        self.ds: xr.Dataset | None = None
        self.rows: list[DimRow] = []
        self._signature = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(5)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("show"))
        self.det_combo = QtWidgets.QComboBox(); self.det_combo.setMinimumWidth(180)
        self.det_combo.currentIndexChanged.connect(self._det_changed)
        top.addWidget(self.det_combo)
        self.part_combo = QtWidgets.QComboBox()
        self.part_combo.addItems(list(PART_TEXT))
        self.part_combo.setFixedWidth(72)
        self.part_combo.setVisible(False)
        self.part_combo.setToolTip("Which part of a complex detector. Averages are taken on\n"
                                   "the complex value first (coherent), then this is applied.")
        self.part_combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        top.addWidget(self.part_combo)
        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("X"))
        self.x_combo = QtWidgets.QComboBox(); self.x_combo.setMinimumWidth(160)
        self.x_combo.currentIndexChanged.connect(self._axes_changed)
        top.addWidget(self.x_combo)
        self.y_combo = QtWidgets.QComboBox(); self.y_combo.setMinimumWidth(160)
        if with_y:
            top.addSpacing(10)
            top.addWidget(QtWidgets.QLabel("Y"))
            self.y_combo.currentIndexChanged.connect(self._axes_changed)
            top.addWidget(self.y_combo)
        top.addStretch(1)
        v.addLayout(top)
        self.rows_box = QtWidgets.QVBoxLayout()
        self.rows_box.setSpacing(3)
        v.addLayout(self.rows_box)

    # ---- data -------------------------------------------------------------
    def set_dataset(self, ds: xr.Dataset | None) -> None:
        self.ds = ds
        if ds is None:
            return
        names = V.detector_names(ds)
        if names != [self.det_combo.itemText(i) for i in range(self.det_combo.count())]:
            self._fill(self.det_combo, names, keep=self.det_combo.currentText())
        self._rebuild()

    def current_da(self) -> xr.DataArray | None:
        if self.ds is None or not self.det_combo.currentText():
            return None
        try:
            return V.detector(self.ds, self.det_combo.currentText())
        except Exception:
            return None

    def selection(self) -> E.Selection | None:
        da = self.current_da()
        x = self.x_combo.currentText()
        if da is None or x not in da.dims:
            return None
        y = self.y_combo.currentText() if self.with_y else None
        y = None if y in ("", NONE_TEXT) else y
        return E.Selection(detector=self.det_combo.currentText(), x=x, y=y,
                           slices={r.dim: r.slice_() for r in self.rows},
                           part=PART_TEXT.get(self.part_combo.currentText(), "abs"))

    def dims(self) -> list[str]:
        da = self.current_da()
        return list(da.dims) if da is not None else []

    # ---- internals --------------------------------------------------------
    @staticmethod
    def _fill(combo, items, keep=None):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if keep and keep in items:
            combo.setCurrentText(keep)
        combo.blockSignals(False)

    def _rebuild(self):
        da = self.current_da()
        if da is None:
            return
        dims = list(da.dims)
        sig = tuple((d, int(da.sizes[d])) for d in dims)
        if sig != self._signature:
            self._signature = sig
            self.part_combo.setVisible(bool(np.iscomplexobj(da.values)))
            # keep the operator's axes when the new detector has them (switching
            # R to Phi should not throw the view away); otherwise innermost two
            dx, dy = V.default_axes(da)
            x_old, y_old = self.x_combo.currentText(), self.y_combo.currentText()
            x = x_old if x_old in dims else dx
            self._fill(self.x_combo, dims, keep=x)
            if self.with_y:
                y = next((c for c in (y_old, dy, dx) if c in dims and c != x), NONE_TEXT)
                self._fill(self.y_combo, [NONE_TEXT] + dims, keep=y)
        plotted = {self.x_combo.currentText()}
        if self.with_y:
            plotted.add(self.y_combo.currentText())
        want = [d for d in dims if d not in plotted]
        if [r.dim for r in self.rows] == want and all(r.n == da.sizes[r.dim] for r in self.rows):
            return
        keep = {r.dim: r.state() for r in self.rows}
        for r in self.rows:
            self.rows_box.removeWidget(r)
            r.setParent(None)            # see DataView: removeWidget alone leaves it painting
            r.deleteLater()
        self.rows = []
        for d in want:
            coords = (np.asarray(self.ds[d].values) if d in self.ds.coords
                      else np.arange(da.sizes[d]))
            unit = self.ds[d].attrs.get("units", "") if d in self.ds.coords else ""
            row = DimRow(d, coords, unit)
            if d in keep:
                row.restore(keep[d])
            row.changed.connect(self.changed.emit)
            self.rows_box.addWidget(row)
            self.rows.append(row)

    def _det_changed(self):
        self._rebuild()
        self.changed.emit()

    def _axes_changed(self):
        self._rebuild()
        self.changed.emit()


# ───────────────────────────── shared export bar ──────────────────────────────

class ExportBar(QtWidgets.QWidget):
    """The same six ways out, for the map and for the curves."""

    def __init__(self, panel: "_Panel"):
        super().__init__()
        self.panel = panel
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        tag = QtWidgets.QLabel("EXPORT"); tag.setObjectName("tag")
        h.addWidget(tag)
        spec = [("Save image…", panel.save_image,
                 "A publication-style figure (white background): PNG, PDF or SVG."),
                ("Copy image", panel.copy_image,
                 "The same figure onto the clipboard -- paste into PowerPoint or a lab book."),
                ("Copy data", panel.copy_data,
                 "The numbers, tab-separated, onto the clipboard -- paste into Excel or Origin."),
                ("Save data…", panel.save_data,
                 "The numbers as .dat (tab) or .csv, with name / unit / label header rows."),
                ("Send to Origin", panel.send_to_origin,
                 "Push into a running Origin (starts one if needed): worksheet or matrix + graph."),
                ("Notebook…", panel.save_notebook,
                 "A Jupyter notebook that recomputes this view from the measurement file(s).")]
        self.buttons = {}
        for text, fn, tip in spec:
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
            self.buttons[text] = b
        h.addStretch(1)


class _Panel(QtWidgets.QWidget):
    """What the map and the 1-D panel share: a host, a status line, the exports."""

    kind = "view"

    def __init__(self, host: "ViewerWidget"):
        super().__init__()
        self.host = host
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{C['muted']}; font-size:11px;")
        self._origin_proc: QtCore.QProcess | None = None

    # ---- to be provided ---------------------------------------------------
    def figure(self):
        raise NotImplementedError

    def write_data(self, path: Path, fmt: str) -> Path:
        raise NotImplementedError

    def data_filters(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def origin_payload(self, path: Path) -> Path:
        raise NotImplementedError

    def notebook_kwargs(self) -> dict:
        raise NotImplementedError

    def export_stem(self) -> str:
        raise NotImplementedError

    # ---- shared -----------------------------------------------------------
    def say(self, text: str, error: bool = False) -> None:
        self.status.setStyleSheet(
            f"color:{C['danger'] if error else C['muted']}; font-size:11px;")
        self.status.setText(text)

    def _suggest(self, ext: str) -> str:
        return str(self.host.export_dir() / f"{self.export_stem()}{ext}")

    def _ask_path(self, title: str, ext: str, filters: str) -> Path | None:
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, self._suggest(ext), filters)
        if not fn:
            return None
        self.host.last_export_dir = Path(fn).parent
        return Path(fn)

    def _guard(self, what: str, fn):
        """Run an export; a failure is reported on the status line, not raised
        into Qt's event loop (where it would only reach a console nobody reads)."""
        try:
            return fn()
        except Exception as exc:
            self.say(f"{what} failed: {exc}", error=True)
            return None

    def save_image(self):
        fig = self._guard("image", self.figure)
        if fig is None:
            return
        path = self._ask_path("Save image", ".png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path and self._guard("image", lambda: E.save_figure(fig, path)):
            self.say(f"saved {path}")

    def copy_image(self):
        fig = self._guard("image", self.figure)
        if fig is None:
            return
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, facecolor="white")
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)
        self.say("figure copied to the clipboard")

    def copy_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._guard("copy", lambda: self.write_data(Path(tmp) / "clip.dat",
                                                             self.data_filters()[0][1]))
            if p is None:
                return
            QtWidgets.QApplication.clipboard().setText(Path(p).read_text(encoding="utf-8"))
        self.say("data copied to the clipboard (tab-separated)")

    def save_data(self):
        filters = self.data_filters()
        fn, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save data", self._suggest(".dat"), ";;".join(f for f, _ in filters))
        if not fn:
            return
        self.host.last_export_dir = Path(fn).parent
        fmt = dict(filters).get(chosen, filters[0][1])
        path = Path(fn)
        if path.suffix.lower() not in (".dat", ".csv", ".txt"):
            path = path.with_suffix(".csv" if "csv" in chosen else ".dat")
        if self._guard("save", lambda: self.write_data(path, fmt)):
            self.say(f"saved {path}")

    def save_notebook(self):
        kw = self._guard("notebook", self.notebook_kwargs)
        if kw is None:
            return
        path = self._ask_path("Save notebook", ".ipynb", "Jupyter notebook (*.ipynb)")
        if path and self._guard("notebook", lambda: E.write_notebook(path, **kw)):
            self.say(f"saved {path} -- open it in Jupyter or VS Code and run all cells")

    def send_to_origin(self):
        """Hand the view to a child process (see aaltoview/origin.py for why)."""
        from ..origin import available
        reason = available()
        if reason:
            self.say(f"Origin: {reason}", error=True)
            return
        if self._origin_proc is not None:
            self.say("Origin: still sending the previous one…")
            return
        tmp = Path(tempfile.mkdtemp(prefix="aaltoview_origin_"))
        payload = self._guard("Origin", lambda: self.origin_payload(tmp / "payload.npz"))
        if payload is None:
            return
        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(str(tmp))
        proc.setProgram(sys.executable)
        proc.setArguments(["-m", "aaltoview.origin", str(payload)])
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.finished.connect(lambda code, _st, p=proc, t=tmp: self._origin_done(p, t))
        self._origin_proc = proc
        self.say("Origin: sending… (the first time, Origin takes a few seconds to start)")
        proc.start()

    def _origin_done(self, proc: QtCore.QProcess, tmp: Path):
        out = bytes(proc.readAll()).decode(errors="replace").strip().splitlines()
        last = out[-1] if out else "no answer"
        self._origin_proc = None
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()
        if last.startswith("OK"):
            self.say(f"Origin: sent as {last[3:]} (with a graph)")
        else:
            self.say(f"Origin: {last.removeprefix('ERROR ')}", error=True)


# ──────────────────────────────── the map ─────────────────────────────────────

class MapPanel(_Panel):
    """A 2-D picture of any two dims, AaltoView's intensity graph."""

    kind = "map"
    cutsRequested = QtCore.Signal(object)          # list[Curve]

    def __init__(self, host):
        super().__init__(host)
        self._cursor: tuple[int, int] | None = None
        self._map: E.Map | None = None
        self._setting_levels = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        self.controls = CubeControls(with_y=True)
        self.controls.changed.connect(self.refresh)
        v.addWidget(self.controls)

        st = QtWidgets.QHBoxLayout()
        st.addWidget(QtWidgets.QLabel("colours"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(list(E.CMAPS))
        self.cmap_combo.currentIndexChanged.connect(self.refresh)
        st.addWidget(self.cmap_combo)
        self.invert = QtWidgets.QCheckBox("inverse")
        self.symmetric = QtWidgets.QCheckBox("symmetric")
        self.symmetric.setToolTip("Limits ±max, centred on zero. Use with red-blue for a signal "
                                  "that changes sign.")
        self.auto = QtWidgets.QCheckBox("auto limits"); self.auto.setChecked(True)
        self.auto.setToolTip("1st to 99th percentile, so a single hot pixel does not flatten "
                             "the map. Untick to type limits, or drag the colour bar.")
        self.lo_edit = QtWidgets.QLineEdit(); self.hi_edit = QtWidgets.QLineEdit()
        for w, tip in ((self.lo_edit, "lower limit"), (self.hi_edit, "upper limit")):
            w.setFixedWidth(90); w.setToolTip(tip)
            w.editingFinished.connect(self._limits_typed)
        self.log = QtWidgets.QCheckBox("log |z|")
        self.norm_combo = QtWidgets.QComboBox()
        self.norm_combo.addItems(["no line norm.", "norm. each row", "norm. each column"])
        self.norm_combo.setToolTip("Scale every line of the map to its own peak -- follows a "
                                   "resonance whose strength changes along the other axis.")
        for w in (self.invert, self.symmetric, self.auto, self.log):
            w.toggled.connect(self.refresh)
            st.addWidget(w)
        st.insertWidget(st.indexOf(self.auto) + 1, self.lo_edit)
        st.insertWidget(st.indexOf(self.lo_edit) + 1, self.hi_edit)
        self.norm_combo.currentIndexChanged.connect(self.refresh)
        st.addWidget(self.norm_combo)
        st.addStretch(1)
        v.addLayout(st)

        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.cbar = pg.ColorBarItem(colorMap=_pg_cmap("magma", False), interactive=True)
        self.cbar.setImageItem(self.img, insert_in=self.plot)
        self.cbar.sigLevelsChangeFinished.connect(self._bar_dragged)
        _plain_axes(self.plot, self.cbar.axis)
        pen = pg.mkPen(C["accent"], width=1, style=QtCore.Qt.DashLine)
        self.vline = pg.InfiniteLine(angle=90, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, pen=pen)
        for ln in (self.vline, self.hline):
            ln.setVisible(False)
            self.plot.addItem(ln, ignoreBounds=True)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.plot.scene().sigMouseMoved.connect(self._hover)
        v.addWidget(self.glw, 1)

        cur = QtWidgets.QHBoxLayout()
        self.readout = QtWidgets.QLabel("click the map to place the cursor")
        self.readout.setStyleSheet(f"color:{C['accent']};")
        cur.addWidget(self.readout, 1)
        self.row_btn = QtWidgets.QPushButton("Row → 1D")
        self.row_btn.setToolTip("The line through the cursor along X, added to 1D plots.")
        self.row_btn.clicked.connect(lambda: self._cut("row"))
        self.col_btn = QtWidgets.QPushButton("Column → 1D")
        self.col_btn.setToolTip("The line through the cursor along Y, added to 1D plots.")
        self.col_btn.clicked.connect(lambda: self._cut("column"))
        cur.addWidget(self.row_btn); cur.addWidget(self.col_btn)
        v.addLayout(cur)

        self.exports = ExportBar(self)
        v.addWidget(self.exports)
        v.addWidget(self.status)

    # ---- state ------------------------------------------------------------
    def map_style(self) -> E.MapStyle:
        return E.MapStyle(cmap=self.cmap_combo.currentText(), invert=self.invert.isChecked(),
                          symmetric=self.symmetric.isChecked(), auto=self.auto.isChecked(),
                          lo=_float(self.lo_edit.text(), 0.0),
                          hi=_float(self.hi_edit.text(), 1.0), log=self.log.isChecked(),
                          norm=E.MAP_NORMS[self.norm_combo.currentIndex()])

    def set_dataset(self, ds):
        self.controls.set_dataset(ds)
        self.refresh()

    def current_map(self) -> tuple[E.Map, V.Reduced] | None:
        ds = self.host.ds
        sel = self.controls.selection()
        if ds is None or sel is None:
            return None
        if sel.y is None:
            raise ValueError("choose a Y dimension -- or use 1D plots for a line")
        if sel.y == sel.x:
            raise ValueError("X and Y must be different dimensions")
        return E.make_map(ds, sel, self.map_style(), self.host.path)

    # ---- drawing ----------------------------------------------------------
    def refresh(self):
        ds = self.host.ds
        if ds is None:
            return
        if len(self.controls.dims()) < 2:
            self.img.setVisible(False)
            self.say("This detector has one dimension -- see the 1D plots tab.")
            return
        try:
            got = self.current_map()
        except Exception as exc:
            self.say(str(exc), error=True)
            return
        if got is None:
            return
        m, red = got
        self._map = m
        z = m.z
        style = self.map_style()
        self.img.setImage(z, autoLevels=False)
        xe, ye = E.edges(m.x), E.edges(m.y)
        self.img.setRect(QtCore.QRectF(xe[0], ye[0], xe[-1] - xe[0], ye[-1] - ye[0]))
        self.img.setVisible(True)
        self._setting_levels = True
        try:
            self.cbar.setColorMap(_pg_cmap(style.cmap, style.invert))
            # pyqtgraph ROUNDS dragged levels to multiples of `rounding`,
            # default 1: on a 0.004..0.014 map the first drag of a handle threw
            # the range out to whole units. ~1/1000 of the span instead.
            span = abs(m.levels[1] - m.levels[0]) or abs(m.levels[1]) or 1.0
            self.cbar.rounding = 10.0 ** np.floor(np.log10(span / 1000.0))
            self.cbar.setLevels(m.levels)
        finally:
            self._setting_levels = False
        if style.auto:
            self.lo_edit.setText(f"{m.levels[0]:.6g}")
            self.hi_edit.setText(f"{m.levels[1]:.6g}")
        self.lo_edit.setEnabled(not style.auto)
        self.hi_edit.setEnabled(not style.auto)
        self.plot.setLabel("bottom", E.axis_title(m.x_name, m.x_unit))
        self.plot.setLabel("left", E.axis_title(m.y_name, m.y_unit))
        self.cbar.axis.setLabel(E.axis_title(m.z_name, m.z_unit))
        if self._cursor:
            i, j = self._cursor
            if i < len(m.x) and j < len(m.y):
                self._place_cursor(i, j)
            else:
                self._cursor = None
                for ln in (self.vline, self.hline):
                    ln.setVisible(False)

        bits = []
        if red.averaged > 1:
            bits.append(f"each pixel is a mean of {red.averaged}")
        if red.coverage < 0.999:
            bits.append(f"{red.coverage * 100:.0f} % measured (missing points are skipped)")
        if red.note:
            bits.append(red.note)
        if not (E.is_uniform(m.x) and E.is_uniform(m.y)):
            bits.append("uneven axis spacing: drawn evenly here; saved figures use the "
                        "true positions")
        what = E.describe_slices(ds, m.selection)
        if what:
            bits.append(what)
        self.say(" · ".join(bits))

    def _limits_typed(self):
        if self.auto.isChecked():
            return
        self.refresh()

    def _bar_dragged(self, bar):
        """Dragging the colour bar is a manual limit, the LabVIEW slider's job."""
        if self._setting_levels:
            return
        lo, hi = bar.levels()
        self.auto.blockSignals(True)
        self.auto.setChecked(False)
        self.auto.blockSignals(False)
        self.symmetric.blockSignals(True)
        self.symmetric.setChecked(False)
        self.symmetric.blockSignals(False)
        self.lo_edit.setText(f"{lo:.6g}")
        self.hi_edit.setText(f"{hi:.6g}")
        self.refresh()

    # ---- cursor -----------------------------------------------------------
    def _index_at(self, scene_pos) -> tuple[int, int, float, float] | None:
        if self._map is None or not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        p = self.plot.vb.mapSceneToView(scene_pos)
        m = self._map
        xe, ye = E.edges(m.x), E.edges(m.y)
        if not (min(xe[0], xe[-1]) <= p.x() <= max(xe[0], xe[-1])
                and min(ye[0], ye[-1]) <= p.y() <= max(ye[0], ye[-1])):
            return None
        i = int(np.argmin(np.abs(m.x - p.x())))
        j = int(np.argmin(np.abs(m.y - p.y())))
        return i, j, p.x(), p.y()

    def _hover(self, pos):
        hit = self._index_at(pos)
        if hit is None or self._cursor is not None:
            return
        i, j, _, _ = hit
        self.readout.setText(self._point_text(i, j) + "   (click to hold the cursor)")

    def _clicked(self, ev):
        hit = self._index_at(ev.scenePos())
        if hit is None:
            return
        self._place_cursor(hit[0], hit[1])

    def _place_cursor(self, i: int, j: int):
        m = self._map
        self._cursor = (i, j)
        self.vline.setPos(float(m.x[i])); self.hline.setPos(float(m.y[j]))
        for ln in (self.vline, self.hline):
            ln.setVisible(True)
        self.readout.setText(self._point_text(i, j))

    def _point_text(self, i: int, j: int) -> str:
        m = self._map
        z = m.z[j, i]
        zt = "not measured" if not np.isfinite(z) else f"{z:.6g} {m.z_unit}".strip()
        return (f"{m.x_name} = {m.x[i]:g} {m.x_unit}   {m.y_name} = {m.y[j]:g} {m.y_unit}"
                f"   →   {zt}")

    def _cut(self, which: str):
        if self._map is None or self._cursor is None:
            self.say("Click the map first: the cut goes through the cursor.")
            return
        sel = self._map.selection
        i, j = self._cursor
        s = E.Selection.from_dict(sel.to_dict())
        if which == "row":
            s.slices[sel.y] = V.Slice("at", j)
            s.y = None
        else:
            s.x, s.y = sel.y, None
            s.slices[sel.x] = V.Slice("at", i)
        try:
            curve = E.make_curve(self.host.ds, s, self.host.path)
        except Exception as exc:
            self.say(f"cut failed: {exc}", error=True)
            return
        self.cutsRequested.emit([curve])
        self.say(f"added to 1D plots: {curve.label}")

    # ---- exports ----------------------------------------------------------
    def _need_map(self) -> E.Map:
        got = self.current_map()
        if got is None:
            raise ValueError("nothing to export -- open a measurement first")
        return got[0]

    def export_stem(self) -> str:
        m = self._map
        det = (m.selection.detector if m else "map").replace(".", "_")
        return f"{self.host.stem()}_{det}_map"

    def figure(self):
        m = self._need_map()
        return E.figure_map(m, self.map_style(), title=self.host.title_text(m.selection, self.host.ds))

    def data_filters(self):
        return [("Matrix, tab-separated (*.dat)", "matrix"),
                ("Matrix, comma-separated (*.csv)", "matrix"),
                ("XYZ columns, tab-separated (*.dat)", "xyz"),
                ("XYZ columns, comma-separated (*.csv)", "xyz")]

    def write_data(self, path, fmt):
        return E.write_map(path, self._need_map(), fmt)

    def origin_payload(self, path):
        from ..origin import save_payload
        return save_payload(path, m=self._need_map(), name=self.export_stem())

    def notebook_kwargs(self):
        return {"m": self._need_map(), "style": self.map_style()}


# ─────────────────────────────── the curves ───────────────────────────────────

class LinePanel(_Panel):
    """1-D plots: a live preview of the current selection plus frozen curves."""

    kind = "curves"

    def __init__(self, host):
        super().__init__(host)
        self.curves: list[E.Curve] = []

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        self.controls = CubeControls(with_y=False)
        self.controls.changed.connect(self._selection_changed)
        v.addWidget(self.controls)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # left: the plot
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.legend = self.plot.addLegend(offset=(-10, 10))
        _plain_axes(self.plot)
        self.preview = pg.PlotDataItem(pen=pg.mkPen(C["muted"], width=1.5,
                                                    style=QtCore.Qt.DashLine))
        self.plot.addItem(self.preview)
        self._items: list[pg.PlotDataItem] = []
        self.plot.scene().sigMouseMoved.connect(self._hover)
        split.addWidget(self.glw)

        # right: adding curves, and the list of them
        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(8, 0, 0, 0)
        sv.setSpacing(6)
        tag = QtWidgets.QLabel("ADD CURVES"); tag.setObjectName("tag")
        sv.addWidget(tag)
        self.add_btn = QtWidgets.QPushButton("Add current")
        self.add_btn.setObjectName("primary")
        self.add_btn.setToolTip("Freeze the dashed preview as a curve.")
        self.add_btn.clicked.connect(self.add_current)
        sv.addWidget(self.add_btn)
        along = QtWidgets.QHBoxLayout()
        along.addWidget(QtWidgets.QLabel("one per value of"))
        self.along_combo = QtWidgets.QComboBox()
        self.along_combo.currentIndexChanged.connect(self._fill_values)
        along.addWidget(self.along_combo, 1)
        sv.addLayout(along)
        self.values = QtWidgets.QListWidget()
        self.values.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.values.setToolTip("Select several (Ctrl/Shift-click), then Add selected.\n"
                               "Every other dimension keeps what its row says.")
        sv.addWidget(self.values, 2)
        self.add_sel_btn = QtWidgets.QPushButton("Add selected")
        self.add_sel_btn.clicked.connect(self.add_selected)
        sv.addWidget(self.add_sel_btn)

        tag2 = QtWidgets.QLabel("CURVES"); tag2.setObjectName("tag")
        sv.addWidget(tag2)
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["Label (double-click to edit)", "From"])
        self.table.setRootIsDecorated(False)
        self.table.itemChanged.connect(self._item_edited)
        sv.addWidget(self.table, 3)
        rb = QtWidgets.QHBoxLayout()
        rm = QtWidgets.QPushButton("Remove"); rm.clicked.connect(self.remove_selected)
        clr = QtWidgets.QPushButton("Clear all"); clr.setObjectName("danger")
        clr.clicked.connect(self.clear)
        rb.addWidget(rm); rb.addWidget(clr)
        sv.addLayout(rb)

        tag3 = QtWidgets.QLabel("DISPLAY"); tag3.setObjectName("tag")
        sv.addWidget(tag3)
        disp = QtWidgets.QGridLayout()
        disp.addWidget(QtWidgets.QLabel("normalise"), 0, 0)
        self.norm_combo = QtWidgets.QComboBox()
        for k, text in E.NORMS.items():
            self.norm_combo.addItem(text, k)
        self.norm_combo.currentIndexChanged.connect(self.redraw)
        disp.addWidget(self.norm_combo, 0, 1)
        disp.addWidget(QtWidgets.QLabel("stack by"), 1, 0)
        self.offset_edit = QtWidgets.QLineEdit("0")
        self.offset_edit.setToolTip("Waterfall: curve k is shifted up by k x this, after "
                                    "normalisation.")
        self.offset_edit.editingFinished.connect(self.redraw)
        disp.addWidget(self.offset_edit, 1, 1)
        self.logy = QtWidgets.QCheckBox("log Y")
        self.logy.toggled.connect(self.redraw)
        disp.addWidget(self.logy, 2, 0)
        self.show_preview = QtWidgets.QCheckBox("preview")
        self.show_preview.setChecked(True)
        self.show_preview.toggled.connect(self.redraw)
        disp.addWidget(self.show_preview, 2, 1)
        sv.addLayout(disp)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([900, 320])
        v.addWidget(split, 1)

        self.readout = QtWidgets.QLabel("")
        self.readout.setStyleSheet(f"color:{C['accent']};")
        v.addWidget(self.readout)
        self.exports = ExportBar(self)
        v.addWidget(self.exports)
        v.addWidget(self.status)

    # ---- state ------------------------------------------------------------
    def norm(self) -> str:
        return self.norm_combo.currentData() or "none"

    def offset(self) -> float:
        return _float(self.offset_edit.text(), 0.0)

    def set_dataset(self, ds):
        self.controls.set_dataset(ds)
        self._selection_changed()

    def _selection_changed(self):
        dims = [d for d in self.controls.dims() if d != self.controls.x_combo.currentText()]
        current = [self.along_combo.itemText(i) for i in range(self.along_combo.count())]
        if dims != current:
            keep = self.along_combo.currentText()
            self.along_combo.blockSignals(True)
            self.along_combo.clear()
            self.along_combo.addItems(dims)
            if keep in dims:
                self.along_combo.setCurrentText(keep)
            self.along_combo.blockSignals(False)
            self._fill_values()
        self.redraw()

    def _fill_values(self):
        self.values.clear()
        ds, d = self.host.ds, self.along_combo.currentText()
        if ds is None or not d:
            self.add_sel_btn.setEnabled(False)
            return
        n = self.controls.current_da().sizes[d]
        for i in range(n):
            self.values.addItem(f"{d} = {V.coord_text(ds, d, i)}")
        self.add_sel_btn.setEnabled(True)

    # ---- curves -----------------------------------------------------------
    def add_curves(self, curves: list[E.Curve]):
        self.table.blockSignals(True)
        for c in curves:
            self.curves.append(c)
            it = QtWidgets.QTreeWidgetItem([c.label, Path(c.source).name if c.source
                                            else "unsaved run"])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(0, QtCore.Qt.Checked)
            it.setToolTip(1, c.source or "a run that has not been saved to a file")
            self.table.addTopLevelItem(it)
        self.table.blockSignals(False)
        self.redraw()

    def add_current(self):
        sel = self.controls.selection()
        if self.host.ds is None or sel is None:
            self.say("Open a measurement first.")
            return
        c = self._guard("add", lambda: E.make_curve(self.host.ds, sel, self.host.path))
        if c:
            self.add_curves([c])
            self.say(f"added: {c.label}")

    def add_selected(self):
        sel = self.controls.selection()
        rows = sorted(self.values.row(it) for it in self.values.selectedItems())
        if self.host.ds is None or sel is None or not rows:
            self.say("Select one or more values in the list first.")
            return
        cs = self._guard("add", lambda: E.curves_along(
            self.host.ds, sel, self.along_combo.currentText(), rows, self.host.path))
        if cs:
            self.add_curves(cs)
            self.say(f"added {len(cs)} curves")

    def remove_selected(self):
        idx = sorted({self.table.indexOfTopLevelItem(it) for it in self.table.selectedItems()},
                     reverse=True)
        for i in idx:
            self.table.takeTopLevelItem(i)
            del self.curves[i]
        self.redraw()

    def clear(self):
        self.table.clear()
        self.curves = []
        self.redraw()

    def _item_edited(self, item, col):
        i = self.table.indexOfTopLevelItem(item)
        if 0 <= i < len(self.curves):
            self.curves[i].label = item.text(0)
            self.curves[i].visible = item.checkState(0) == QtCore.Qt.Checked
            self.redraw()

    # ---- drawing ----------------------------------------------------------
    def redraw(self):
        for it in self._items:
            self.plot.removeItem(it)
        self._items = []
        self.legend.clear()
        norm, off = self.norm(), self.offset()
        shown = [c for c in self.curves if c.visible]
        for k, (c, y) in enumerate(zip(shown, E.displayed_y(self.curves, norm, off))):
            col = TAB10[k % len(TAB10)]
            item = pg.PlotDataItem(c.x, y, pen=pg.mkPen(col, width=2), symbol="o",
                                   symbolSize=4, symbolBrush=col, symbolPen=None,
                                   name=c.label, connect="finite")
            self.plot.addItem(item)
            self._items.append(item)

        preview = None
        if self.show_preview.isChecked() and self.host.ds is not None:
            sel = self.controls.selection()
            if sel is not None:
                try:
                    preview = E.make_curve(self.host.ds, sel, self.host.path)
                except Exception as exc:
                    self.say(str(exc), error=True)
        if preview is not None:
            y = E.normalize(preview.y, norm) + len(shown) * off
            self.preview.setData(preview.x, y, connect="finite")
            self.legend.addItem(self.preview, f"preview: {preview.label}")
        else:
            self.preview.setData([], [])
        ref = shown[0] if shown else preview
        if ref is not None:
            self.plot.setLabel("bottom", E.axis_title(ref.x_name, ref.x_unit))
            self.plot.setLabel("left", E.axis_title(ref.y_name, ref.y_unit if norm == "none"
                                                    else E.NORMS[norm]))
        self.plot.setLogMode(y=self.logy.isChecked())

    def _hover(self, pos):
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = self.plot.vb.mapSceneToView(pos)
        y = 10 ** p.y() if self.logy.isChecked() else p.y()
        self.readout.setText(f"x = {p.x():.6g}   y = {y:.6g}")

    # ---- exports ----------------------------------------------------------
    def _need_curves(self) -> list[E.Curve]:
        if not any(c.visible for c in self.curves):
            raise ValueError("no curves yet -- press Add current or Add selected")
        return self.curves

    def export_stem(self) -> str:
        return f"{self.host.stem()}_curves"

    def figure(self):
        return E.figure_curves(self._need_curves(), self.norm(), self.offset(),
                               self.logy.isChecked())

    def data_filters(self):
        return [("Columns, tab-separated (*.dat)", "columns"),
                ("Columns, comma-separated (*.csv)", "columns")]

    def write_data(self, path, fmt):
        return E.write_curves(path, self._need_curves(), self.norm(), self.offset())

    def origin_payload(self, path):
        from ..origin import save_payload
        return save_payload(path, curves=self._need_curves(), norm=self.norm(),
                            offset=self.offset(), name=self.export_stem())

    def notebook_kwargs(self):
        return {"curves": self._need_curves(), "norm": self.norm(),
                "offset": self.offset(), "logy": self.logy.isChecked()}


# ─────────────────────────────── the whole thing ──────────────────────────────

class ViewerWidget(QtWidgets.QWidget):
    """Files on the left, Map and 1D plots on the right. Embeddable (the suite's
    Data tab) or the central widget of its own window (`main`)."""

    def __init__(self, folder: Path | None = None):
        super().__init__()
        self.ds: xr.Dataset | None = None
        self.path: Path | None = None
        self.last_export_dir: Path | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        self.top = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open file…")
        self.open_btn.clicked.connect(self._open_dialog)
        self.top.addWidget(self.open_btn)
        self._action_slot = self.top.count()
        self.current = QtWidgets.QLabel("no measurement open")
        self.current.setStyleSheet(f"color:{C['accent']}; font-weight:700;")
        self.top.addSpacing(10)
        self.top.addWidget(self.current, 1)
        outer.addLayout(self.top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.browser = FileBrowser(folder)
        self.browser.fileChosen.connect(self.load_file)
        split.addWidget(self.browser)
        self.tabs = QtWidgets.QTabWidget()
        self.map = MapPanel(self)
        self.lines = LinePanel(self)
        self.map.cutsRequested.connect(self.lines.add_curves)
        self.tabs.addTab(self.map, "Map")
        self.tabs.addTab(self.lines, "1D plots")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([430, 1100])
        outer.addWidget(split, 1)

        #: the most recent message of either panel, for a host that wants one line
        self.status = self.map.status
        if folder:
            QtCore.QTimer.singleShot(0, self.browser.refresh)

    def add_action(self, w: QtWidgets.QWidget) -> None:
        """A host's own button, in this toolbar (the suite's "Show the last run")."""
        self.top.insertWidget(self._action_slot, w)
        self._action_slot += 1

    # ---- where things are -------------------------------------------------
    @property
    def default_dir(self) -> Path | None:
        return self.browser.folder

    @default_dir.setter
    def default_dir(self, folder) -> None:
        # listed when shown, not now: a host sets this at start-up, and reading
        # a few hundred headers then would slow the whole suite's start.
        self.browser.set_folder(folder, refresh=False)
        if self.isVisible():
            self.browser.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self.browser.tree.topLevelItemCount() == 0 and self.browser.folder:
            self.browser.refresh()

    def export_dir(self) -> Path:
        for d in (self.last_export_dir, self.path.parent if self.path else None,
                  self.default_dir):
            if d and Path(d).is_dir():
                return Path(d)
        return Path.home()

    def stem(self) -> str:
        return self.path.stem if self.path else "unsaved_run"

    @staticmethod
    def title_text(sel: E.Selection, ds) -> str:
        return E.describe_slices(ds, sel) if ds is not None else ""

    # ---- data in ----------------------------------------------------------
    def set_dataset(self, ds: xr.Dataset | None, path: Path | None = None) -> None:
        self.ds = ds
        self.path = Path(path) if path else None
        if ds is None:
            return
        self.current.setText(self.path.name if self.path else "unsaved run (this session)")
        self.current.setToolTip(str(self.path or ""))
        self.map.set_dataset(ds)
        self.lines.set_dataset(ds)
        # a 1-D scan has no map to look at
        if max((len(V.detector(ds, n).dims) for n in V.detector_names(ds)), default=0) < 2:
            self.tabs.setCurrentWidget(self.lines)

    def load_file(self, path) -> None:
        try:
            ds = load(path)
            data = ds.load()           # read it all now; do not hold the file open
            ds.close()
        except Exception as exc:
            self.map.say(f"Could not read {Path(path).name}: {exc}", error=True)
            return
        self.set_dataset(data, Path(path))

    def _open_dialog(self):
        start = str(self.path.parent if self.path else (self.default_dir or ""))
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open measurement", start, "netCDF (*.nc);;All files (*)")
        if fn:
            self.load_file(fn)


def window_title() -> str:
    """'TR-MOKE · AaltoView' when installed with an AaltoFlow suite whose
    setup has a name (the installer asks for it); plain 'AaltoView' alone."""
    try:
        from suite_common import setup_name           # the AaltoFlow suite, if present
        name = setup_name()
    except Exception:
        name = ""
    return f"{name} · AaltoView" if name else "AaltoView"


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, folder: Path | None = None, file: Path | None = None):
        super().__init__()
        self.setWindowTitle(window_title())
        self.resize(1600, 960)
        root = QtWidgets.QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)
        title = QtWidgets.QLabel(window_title().upper()); title.setObjectName("title")
        v.addWidget(title)
        self.viewer = ViewerWidget(folder)
        v.addWidget(self.viewer, 1)
        if file:
            QtCore.QTimer.singleShot(0, lambda: self.viewer.load_file(file))


def _settings() -> QtCore.QSettings:
    # per Windows user, in the registry (HKCU\Software\AaltoFlow\AaltoView).
    # The viewer was "TRMOKE DataViewer" until 2026-09-24: copy what the old
    # name remembered (the last folder) the first time, so the rename does not
    # make the viewer forget where the data is.
    new = QtCore.QSettings("AaltoFlow", "AaltoView")
    if not new.allKeys():
        old = QtCore.QSettings("TRMOKE", "DataViewer")
        for key in old.allKeys():
            new.setValue(key, old.value(key))
    return new


def remember_folder(folder: Path) -> None:
    _settings().setValue("folder", str(folder))


def start_folder(explicit: str | None) -> Path:
    """Which folder to list at start, most specific first:
    --folder; the folder chosen here last time; the measurement suite's data
    directory (only when the AaltoFlow suite is installed alongside); home."""
    if explicit:
        return Path(explicit)
    last = _settings().value("folder")
    if last and Path(str(last)).is_dir():
        return Path(str(last))
    try:
        from suite_common import get_setting          # the AaltoFlow suite, if present
        suite_dir = get_setting("data_dir")
        if suite_dir and Path(suite_dir).is_dir():
            return Path(suite_dir)
    except Exception:
        pass
    return Path.home()


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major",
                        background=C["code_bg"], foreground=C["text"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AaltoView")
    ap.add_argument("file", nargs="?", help="a measurement (.nc) to open")
    ap.add_argument("--folder", default=None,
                    help="data folder to list (default: the one used last time)")
    ap.add_argument("--theme", choices=["dark", "light"], default=None)
    args = ap.parse_args(argv)

    set_theme(args.theme or DEFAULT_THEME)        # BEFORE any widget is built
    configure_pyqtgraph()
    # Numbers in C locale: under the lab PC's locale 0.5 would show as "0,5"
    # and 10 ms as "10,000" (gotcha #18).
    loc = QtCore.QLocale.c()
    loc.setNumberOptions(QtCore.QLocale.OmitGroupSeparator)
    QtCore.QLocale.setDefault(loc)

    folder = start_folder(args.folder)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply(app)
    apply_window_icon(app)
    win = ViewerWindow(folder, Path(args.file) if args.file else None)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
