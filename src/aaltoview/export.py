"""export.py -- getting a view OUT of the viewer: curves, maps, files, notebooks.

The viewer (aaltoview/apps/viewer.py) is the successor of the LabVIEW AaltoView. What it
shows is always one of two things, and both are described here as DATA, with
no Qt anywhere, so every export can be tested without a screen:

* a MAP  -- `Selection(detector, x, y, slices)` reduced to (y, x), drawn with a
  `MapStyle` (reference, colour map, limits, symmetric, log, per-line
  normalisation);
* CURVES -- each a `Curve`: the numbers AND the `Selection` + file they came
  from, so a set of overlaid curves can come from several measurement files and
  can still be regenerated from scratch.

Where a view can go:

    write_curves(path, curves, norm)    .csv / .dat / .txt  (Origin, Excel, pandas)
    write_map(path, m, fmt)             matrix or XYZ columns
    figure_curves / figure_map          a matplotlib Figure -> PNG / PDF / SVG
    write_notebook(path, ...)           a Jupyter notebook that RECOMPUTES the view
                                        from the .nc files (see its docstring)
    origin.py                           straight into a running Origin

Why a notebook and not only a CSV: a CSV is the answer, a notebook is the
answer plus how it was obtained. Averaging over 12 frequencies and normalising
to the peak is a decision; six months later the notebook still says so, and a
reviewer can change the range and run it again.
"""

from __future__ import annotations

import csv
import json
import os
import pprint
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from .segments import INDEX, LINES, Segments, segments_of, valid_choice
from .view import Reduced, Slice, apply_part, coord_text, detector, reduce_cube

#: How a curve is scaled for display/export. The stored numbers never change;
#: normalisation is applied on the way out, so switching it back is lossless.
NORMS = {
    "none": "as measured",
    "peak": "divide by max |y|",
    "minmax": "scale to 0 ... 1",
    "first": "divide by the first point",
    "zero_mean": "subtract the mean",
}

#: Per-line normalisation of a MAP (AaltoView's "1D norm. by axis"). An FMR map
#: whose signal strength changes with frequency shows the resonance line far
#: better when every line along the field is scaled to itself.
MAP_NORMS = ("none", "rows", "columns")

#: A map against a reference (see `reference_values`): one line, the median
#: line, or the derivative-divide along an axis. Applied to the complex values.
MAP_REFS = ("none", "row", "column", "median_rows", "median_columns", "dd_y", "dd_x")
MAP_REF_OPS = ("divide", "subtract")

#: Colour maps offered by the viewer -> the matplotlib name behind each. The
#: viewer draws with these same tables (pyqtgraph loads them from matplotlib),
#: so an exported figure has the colours that were on screen. "red-blue" is
#: AaltoView's map for signed signals: use it with "symmetric".
CMAPS = {"magma": "magma", "viridis": "viridis", "inferno": "inferno",
         "cividis": "cividis", "grey": "gray", "red-blue": "RdBu_r",
         "coolwarm": "coolwarm", "seismic": "seismic"}

PART_LABEL = {"abs": "|{}|", "arg": "arg {}", "real": "Re {}", "imag": "Im {}"}


# ─────────────────────────────── what is shown ────────────────────────────────

@dataclass
class Selection:
    """Which piece of which detector: the complete, replayable description."""
    detector: str
    x: str
    y: str | None = None
    slices: dict[str, Slice] = field(default_factory=dict)
    part: str = "abs"
    #: a CUT of a referenced map: {"x", "y" (the map's axes), "mode", "op", "i"}
    #: -- the line is taken from the map AFTER the reference (see ref_line)
    ref: dict | None = None
    #: an image axis that goes up AND down: which run(s) to show, {dim:
    #: "lines" | "index" | run number} (segments.py). Missing = the default:
    #: every run as its own line on a 1-D plot, the first run on a map.
    runs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"detector": self.detector, "x": self.x, "y": self.y, "part": self.part,
             "slices": {d: {"mode": s.mode, "i0": int(s.i0),
                            "i1": None if s.i1 is None else int(s.i1)}
                        for d, s in self.slices.items()}}
        if self.ref:
            d["ref"] = dict(self.ref)
        if self.runs:
            d["runs"] = {k: (v if isinstance(v, str) else int(v))
                         for k, v in self.runs.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        return cls(detector=d["detector"], x=d["x"], y=d.get("y"),
                   part=d.get("part", "abs"),
                   slices={k: Slice(v["mode"], v["i0"], v.get("i1"))
                           for k, v in d.get("slices", {}).items()},
                   ref=dict(d["ref"]) if d.get("ref") else None,
                   runs=dict(d.get("runs") or {}))


@dataclass
class Curve:
    """One line on the 1-D plot, with enough attached to make it again."""
    x: np.ndarray
    y: np.ndarray
    label: str
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    selection: Selection
    source: str | None = None       # the .nc file; None = unsaved (a live run)
    visible: bool = True
    #: the COMPLEX values behind y when the detector is complex (y is only the
    #: part on screen). An FMR fit wants both quadratures; see analysis_link.py.
    z: np.ndarray | None = None
    #: the dims held at one value, {dim: (value, unit)}: "this sweep was at
    #: 8 GHz" as a number, so a fit result can be plotted against it
    held: dict[str, tuple[float, str]] = field(default_factory=dict)
    #: X goes up AND down (a hysteresis loop) and every run is shown: the run
    #: number of each point and one info dict per run (segments_json). The
    #: curve is then DRAWN and EXPORTED as one line per run (curve_runs).
    segment: np.ndarray | None = None
    runs: list[dict] | None = None


def curve_runs(c: "Curve") -> list[tuple[str | None, np.ndarray]]:
    """[(run name, indices to draw)] of a curve: one entry per run when its X
    goes up and down (a joined run starts at the previous run's last point,
    so the loop is closed as measured), else one entry with every point."""
    if c.segment is None or not c.runs:
        return [(None, np.arange(np.asarray(c.x).size))]
    segs = Segments(np.asarray(c.segment, dtype=int), list(c.runs))
    return [(segs.name(k), segs.line_indices(k)) for k in range(segs.n)]


def run_label(label: str, run: str | None) -> str:
    """'field = 8 GHz · down' -- a label with its run, when it has one."""
    return label if not run else f"{label} · {run}"


def split_runs(c: "Curve") -> list["Curve"]:
    """A curve whose X goes up AND down as one curve PER RUN (labels "· up",
    "· down"), each with its own selection: what an analysis module gets --
    a fit of a loop wants the branches apart, and its label says which."""
    if c.segment is None or not c.runs:
        return [c]
    out = []
    segs = Segments(np.asarray(c.segment, dtype=int), list(c.runs))
    for k in range(segs.n):
        idx = segs.indices(k)
        sel = Selection.from_dict(c.selection.to_dict())
        sel.runs[c.x_name] = k
        out.append(Curve(x=np.asarray(c.x)[idx].copy(), y=np.asarray(c.y)[idx].copy(),
                         label=run_label(c.label, segs.name(k)), x_name=c.x_name,
                         x_unit=c.x_unit, y_name=c.y_name, y_unit=c.y_unit,
                         selection=sel, source=c.source, visible=c.visible,
                         z=None if c.z is None else np.asarray(c.z)[idx].copy(),
                         held=dict(c.held)))
    return out


@dataclass
class MapStyle:
    cmap: str = "magma"
    invert: bool = False            # flip the colour map
    symmetric: bool = False         # limits +-max, centred on zero (Kerr signals)
    auto: bool = True               # limits from the 0.1st/99.9th percentile
    lo: float = 0.0                 # used when auto is False
    hi: float = 1.0
    log: bool = False               # log10 |value|
    norm: str = "none"              # MAP_NORMS
    ref: str = "none"               # MAP_REFS, applied first (complex values)
    ref_op: str = "divide"          # MAP_REF_OPS (row/column/median only)
    ref_i: int = -1                 # the reference line's index (-1 = the last)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Map:
    """A map ready to draw or write: the values as they will appear."""
    z: np.ndarray                   # (ny, nx), styling already applied
    x: np.ndarray
    y: np.ndarray
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    z_name: str
    z_unit: str
    levels: tuple[float, float]
    selection: Selection
    source: str | None = None
    #: an axis that goes up AND down: the FILE index of each column / row
    #: shown (None = 0, 1, 2 ...) and its run's name ("down"; None = no runs)
    x_index: np.ndarray | None = None
    y_index: np.ndarray | None = None
    x_tags: list[str] | None = None
    y_tags: list[str] | None = None


# ───────────────────────────── building them ──────────────────────────────────

def units_of(ds: xr.Dataset, name: str) -> str:
    if name in ds.coords or name in ds.data_vars:
        return str(ds[name].attrs.get("units", ""))
    if f"{name}_real" in ds.data_vars:
        return str(ds[f"{name}_real"].attrs.get("units", ""))
    return ""


def coords_of(ds: xr.Dataset, dim: str, n: int) -> np.ndarray:
    """A dimension's coordinate values -- or 0..n-1 when the file has none."""
    if dim in ds.coords:
        return np.asarray(ds[dim].values, dtype=float)
    return np.arange(n, dtype=float)


def shown_coords(ds: xr.Dataset, red: Reduced, dim: str, n: int) -> np.ndarray:
    """The coordinate of an image axis AS SHOWN: one run of an up-and-down
    axis, or point numbers -- whatever the reduction left on it."""
    if dim in red.index and dim in red.data.coords:
        return np.asarray(red.data[dim].values, dtype=float)
    return coords_of(ds, dim, n)


def shown_indices(ds: xr.Dataset, sel: Selection, dim: str) -> np.ndarray:
    """The FILE indices of the points an image axis of `sel` shows, in the
    order shown: one run of an up-and-down axis (ascending), else all."""
    n = detector(ds, sel.detector).sizes[dim]
    segs = segments_of(ds, dim)
    if segs is None or segs.seg.size != n:
        return np.arange(n)
    choice = valid_choice(segs, sel.runs.get(dim), sel.y is None and dim == sel.x)
    if isinstance(choice, int):
        return segs.ascending(choice, coords_of(ds, dim, n))
    return np.arange(n)


def _axis_runs(ds: xr.Dataset, red: Reduced, dim: str | None, sel: Selection):
    """(file indices shown, run name of each) for a segmented image axis;
    (None, None) for an ordinary one."""
    if dim is None or dim not in red.index:
        return None, None
    segs = segments_of(ds, dim)
    if segs is None:
        return None, None
    idx = np.asarray(red.index[dim], dtype=int)
    if axis_unit(ds, sel, dim) == "point #":
        # shown by point number: the tag carries the value too ("50 mT · down")
        from .view import coord_texts
        return idx, coord_texts(ds, dim, idx.tolist())
    return idx, [segs.name_at(i) for i in idx]


def axis_unit(ds: xr.Dataset, sel: Selection, dim: str) -> str:
    """The unit of an image axis as shown: "point #" when an up-and-down axis
    is drawn against its point number, else the coordinate's own."""
    segs = segments_of(ds, dim)
    if segs is not None and valid_choice(
            segs, sel.runs.get(dim), sel.y is None and dim == sel.x) == INDEX:
        return "point #"
    return units_of(ds, dim)


def runs_text(ds: xr.Dataset, sel: Selection) -> str:
    """'field down' -- which run of each up-and-down image axis is shown
    (nothing for a line plot that draws every run: its legend names them)."""
    bits = []
    for dim in [d for d in (sel.x, sel.y) if d]:
        segs = segments_of(ds, dim)
        if segs is None:
            continue
        choice = valid_choice(segs, sel.runs.get(dim), sel.y is None and dim == sel.x)
        if choice == LINES:
            continue
        bits.append(f"{dim} by point #" if choice == INDEX else
                    f"{dim} {segs.name(int(choice))}")
    return ", ".join(bits)


def quantity_name(sel: Selection, is_complex: bool) -> str:
    return PART_LABEL[sel.part].format(sel.detector) if is_complex else sel.detector


def describe_slices(ds: xr.Dataset, sel: Selection) -> str:
    """'rf_freq = 501.1 MHz, x mean 0 ... 20 um' -- what was done to the other dims.

    Dims of length 1 are left out: holding a one-point axis is not information.
    """
    da = detector(ds, sel.detector)
    bits = []
    for d in da.dims:
        if d in (sel.x, sel.y) or da.sizes[d] <= 1:
            continue
        s = sel.slices.get(d, Slice())
        i0, i1 = s.span(da.sizes[d])
        if s.mode == "at":
            bits.append(f"{d} = {coord_text(ds, d, i0)}")
            continue
        segs = segments_of(ds, d)
        run = None
        if segs is not None:
            run = next((k for k in range(segs.n) if segs.span(k) == (i0, i1)), None)
        if run is not None:
            bits.append(f"{d} mean of the {segs.name(run)} run")
        elif i0 == 0 and i1 == da.sizes[d] - 1:
            bits.append(f"{d} mean")
        else:
            bits.append(f"{d} mean {coord_text(ds, d, i0)} ... {coord_text(ds, d, i1)}")
        if segs is not None and run is None and len(set(segs.seg[i0:i1 + 1].tolist())) > 1:
            # never silently: an average across a turning point mixes branches
            bits[-1] += " (up and down mixed)"
    return ", ".join(bits)


def reduce(ds: xr.Dataset, sel: Selection) -> Reduced:
    return reduce_cube(detector(ds, sel.detector), sel.x, sel.y, sel.slices, sel.part,
                       runs=sel.runs, ds=ds)


def make_curve(ds: xr.Dataset, sel: Selection, source: str | Path | None = None,
               label: str | None = None, da: xr.DataArray | None = None) -> Curve:
    """Freeze the current 1-D selection into a Curve (the numbers are copied).

    da: the detector, when the caller already has it (curves_along: recombining
    a complex pair per curve was most of the 4.6 s for 174 VNA sweeps)."""
    sel = Selection.from_dict(sel.to_dict())        # detach from the caller's dict
    sel.y = None
    da = detector(ds, sel.detector) if da is None else da
    cplx = np.iscomplexobj(da.values)
    y_name = quantity_name(sel, cplx)
    y_unit = "rad" if (sel.part == "arg" and cplx) else units_of(ds, sel.detector)
    if sel.ref:
        zl, x = ref_line(ds, sel, with_x=True)
        y = np.asarray(apply_part(xr.DataArray(zl), sel.part).values, dtype=float).copy()
        style, msel = _ref_style(sel)
        y_name, y_unit = _ref_name_unit(ds, msel, style, y_name, y_unit)
        x = np.asarray(x, dtype=float).copy()
    else:
        red = reduce_cube(da, sel.x, None, sel.slices, sel.part, runs=sel.runs, ds=ds)
        y = np.asarray(red.data.values, dtype=float).copy()
        x = shown_coords(ds, red, sel.x, y.size).copy()
    if label is None:
        label = describe_slices(ds, sel) or sel.detector
        shown = runs_text(ds, Selection(sel.detector, sel.x, None, runs=sel.runs))
        if shown:
            label += f" ({shown})"
        if sel.ref:
            label += f" ({reference_text(ds, msel, style)})"
    z = None
    if cplx:
        z = (zl.astype(complex).copy() if sel.ref else
             np.asarray(reduce_cube(da, sel.x, None, sel.slices, "complex",
                                    runs=sel.runs, ds=ds).data.values,
                        dtype=complex).copy())
    # every run of an up-and-down X: the curve remembers which point is on
    # which run, so it is drawn and written one line per run
    segment = runs = None
    segs = segments_of(ds, sel.x)
    if (segs is not None and segs.seg.size == x.size
            and valid_choice(segs, sel.runs.get(sel.x), True) == LINES):
        segment, runs = segs.seg.copy(), [dict(r) for r in segs.info]
    return Curve(x=x, y=y, label=label, x_name=sel.x, x_unit=axis_unit(ds, sel, sel.x),
                 y_name=y_name, y_unit=y_unit,
                 selection=sel, source=str(source) if source else None,
                 z=z, held=held_values(ds, sel, da), segment=segment, runs=runs)


def _ref_style(sel: Selection) -> tuple["MapStyle", Selection]:
    """The map a referenced cut came from: its style (reference only) and axes."""
    r = sel.ref
    # An up-and-down axis of that map: the run the cut was taken from when one
    # was chosen; otherwise EVERY point in measured order ("index"), so a
    # held point of any run can be found and a line keeps all its runs.
    runs = dict(sel.runs)
    for d in (r["x"], r["y"]):
        if runs.get(d) in (None, LINES):
            runs[d] = INDEX
    return (MapStyle(ref=r["mode"], ref_op=r.get("op", "divide"), ref_i=int(r.get("i", -1))),
            Selection(sel.detector, x=r["x"], y=r["y"], slices=sel.slices, part=sel.part,
                      runs=runs))


def ref_line(ds: xr.Dataset, sel: Selection, with_x: bool = False):
    """A cut of a referenced map, as COMPLEX values along sel.x: the map (its
    two axes, everything else as sel says) referenced as a whole, then the line
    at the held index. So a column divided by the 800 mT column, a row divided
    by its own value at 800 mT, a derivative-divide across the lines -- each
    exactly what the map shows through the cursor."""
    style, msel = _ref_style(sel)
    held = msel.y if sel.x == msel.x else msel.x
    if sel.x not in (msel.x, msel.y) or held == sel.x:
        raise ValueError(f"a referenced cut runs along {msel.x} or {msel.y}")
    da = detector(ds, sel.detector)
    slices = {d: s for d, s in sel.slices.items() if d != held}
    red = reduce_cube(da, msel.x, msel.y, slices, "complex", runs=msel.runs, ds=ds)
    ny, nx = red.data.shape

    def axis(dim, n):
        # all points in measured order: their TRUE values, not point numbers
        if sel.runs.get(dim) in (None, LINES):
            return coords_of(ds, dim, n)
        return shown_coords(ds, red, dim, n)
    xs, ys = axis(msel.x, nx), axis(msel.y, ny)
    zc = reference_values(red.data.values, style, xs, ys)
    s = sel.slices.get(held, Slice())
    if s.mode != "at":
        raise ValueError(f"hold {held} at one value to reference along it "
                         f"(it is averaged now)")
    k = s.span(da.sizes[held])[0]
    if held in red.index:
        # the held axis goes up and down: the map shows one run; find the
        # held point (a FILE index) among the lines it shows
        pos = np.flatnonzero(np.asarray(red.index[held]) == k)
        if pos.size == 0:
            raise ValueError(f"{held} = {coord_text(ds, held, k)} is not on the "
                             f"run the map shows")
        k = int(pos[0])
    line = np.asarray(zc[k] if held == msel.y else zc[:, k])
    if with_x:
        return line, (xs if held == msel.y else ys)
    return line


def _ref_name_unit(ds, sel: Selection, style: "MapStyle", name: str, unit: str):
    """'|s21| (÷ field = 800 mT)', and the unit a reference leaves."""
    name = f"{name} ({reference_text(ds, sel, style)})"
    if style.ref in ("dd_y", "dd_x") and sel.part != "arg":
        dim = sel.y if style.ref == "dd_y" else sel.x
        unit = f"1/{units_of(ds, dim)}" if units_of(ds, dim) else ""
    elif style.ref_op == "divide" and sel.part != "arg":
        unit = ""
    return name, unit


def held_values(ds: xr.Dataset, sel: Selection,
                da: xr.DataArray | None = None) -> dict[str, tuple[float, str]]:
    """{dim: (value, unit)} for every dim held at ONE index -- a dim of length 1
    too: "this sweep was at 10 GHz" is what an analysis needs, even when the
    label leaves it out as obvious."""
    da = detector(ds, sel.detector) if da is None else da
    out = {}
    for d in da.dims:
        if d in (sel.x, sel.y):
            continue
        s = sel.slices.get(d, Slice())
        i0, i1 = s.span(da.sizes[d])
        if i0 == i1:
            out[d] = (float(coords_of(ds, d, da.sizes[d])[i0]), units_of(ds, d))
    return out


def curves_along(ds: xr.Dataset, sel: Selection, dim: str, indices,
                 source: str | Path | None = None) -> list[Curve]:
    """One curve per chosen value of `dim` -- AaltoView's multi-select parameter list.

    Every other dim keeps what `sel` says about it; `dim` is held at each index
    in turn.
    """
    da = detector(ds, sel.detector)                # once, not per curve
    out = []
    for i in indices:
        s = Selection.from_dict(sel.to_dict())
        s.slices[dim] = Slice("at", int(i))
        out.append(make_curve(ds, s, source, da=da))
    return out


def normalize(y: np.ndarray, mode: str) -> np.ndarray:
    """Scale a curve for display. NaN-aware: a hole stays a hole, never a zero."""
    y = np.asarray(y, dtype=float)
    fin = y[np.isfinite(y)]
    if mode == "none" or fin.size == 0:
        return y
    if mode == "peak":
        m = np.max(np.abs(fin))
        return y / m if m > 0 else y
    if mode == "minmax":
        lo, hi = np.min(fin), np.max(fin)
        return (y - lo) / (hi - lo) if hi > lo else y - lo
    if mode == "first":
        first = y[np.isfinite(y)][0]
        return y / first if first != 0 else y
    if mode == "zero_mean":
        return y - np.mean(fin)
    raise ValueError(f"unknown normalisation '{mode}' (have: {', '.join(NORMS)})")


def displayed_y(curves: list[Curve], norm: str = "none", offset: float = 0.0) -> list[np.ndarray]:
    """What each VISIBLE curve looks like on screen: normalised, then stacked.

    The offset is a waterfall: curve k is shifted by k * offset, so overlapping
    spectra can be read one above the other. Applied after normalisation,
    because "stack by 1" only means something once the curves share a scale.
    """
    shown = [c for c in curves if c.visible]
    return [normalize(c.y, norm) + k * offset for k, c in enumerate(shown)]


def make_map(ds: xr.Dataset, sel: Selection, style: MapStyle | None = None,
             source: str | Path | None = None) -> tuple[Map, Reduced]:
    """Reduce to (y, x) and apply the style, so a file gets what the screen shows."""
    style = style or MapStyle()
    if sel.y is None:
        raise ValueError("a map needs a Y dimension")
    da = detector(ds, sel.detector)
    cplx = np.iscomplexobj(da.values)
    name = quantity_name(sel, cplx)
    unit = "rad" if (sel.part == "arg" and cplx) else units_of(ds, sel.detector)
    if style.ref == "none":
        red = reduce_cube(da, sel.x, sel.y, sel.slices, sel.part, runs=sel.runs, ds=ds)
        z = np.asarray(red.data.values, dtype=float)
    else:
        # reference BEFORE |z| / arg: the ratio of two complex numbers
        red = reduce_cube(da, sel.x, sel.y, sel.slices, "complex", runs=sel.runs, ds=ds)
        ny, nx = red.data.shape
        zc = reference_values(red.data.values, style, shown_coords(ds, red, sel.x, nx),
                              shown_coords(ds, red, sel.y, ny))
        red.data = apply_part(red.data.copy(data=zc), sel.part)
        if red.averaged > 1 and cplx and sel.part in ("abs", "arg"):
            red.note = "coherent average (complex averaged, then " + sel.part + ")"
        z = np.asarray(red.data.values, dtype=float)
        name, unit = _ref_name_unit(ds, sel, style, name, unit)
    z = style_values(z, style)
    ny, nx = z.shape
    if style.norm != "none":
        name, unit = f"{name} (normalised per {style.norm[:-1]})", ""
    if style.log:
        name, unit = f"log10 {name}", ""
    xi, xt = _axis_runs(ds, red, sel.x, sel)
    yi, yt = _axis_runs(ds, red, sel.y, sel)
    m = Map(z=z, x=shown_coords(ds, red, sel.x, nx), y=shown_coords(ds, red, sel.y, ny),
            x_name=sel.x, x_unit=axis_unit(ds, sel, sel.x),
            y_name=sel.y, y_unit=axis_unit(ds, sel, sel.y),
            z_name=name, z_unit=unit, levels=map_levels(z, style),
            selection=sel, source=str(source) if source else None,
            x_index=xi, y_index=yi, x_tags=xt, y_tags=yt)
    return m, red


@dataclass
class MapData:
    """A whole map for an ANALYSIS module: the numbers before any styling.

    `values` is the part that was on screen (|z|, Re, ...) after the reference,
    without the per-line normalisation or the log (those are for the eye);
    `z` is the COMPLEX map behind it when the detector is complex, so an FFT
    or a fit can work on both quadratures. Rows run along x: values[i, :] is
    the line at y[i]."""
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray              # (ny, nx), real
    label: str
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    z_name: str
    z_unit: str
    selection: Selection
    source: str | None = None
    z: np.ndarray | None = None     # (ny, nx), complex
    held: dict[str, tuple[float, str]] = field(default_factory=dict)
    ref: dict | None = None         # the reference applied: {"mode", "op", "i"}
    #: as on Map: the file index of each column / row (an up-and-down axis
    #: shows one run) and its run's name; None = an ordinary axis
    x_index: np.ndarray | None = None
    y_index: np.ndarray | None = None
    x_tags: list[str] | None = None
    y_tags: list[str] | None = None


def make_map_data(ds: xr.Dataset, sel: Selection, style: MapStyle | None = None,
                  source: str | Path | None = None) -> MapData:
    """The map on screen, for an analysis module: its reference applied (on the
    complex values), its colour styling NOT (normalisation, log, limits)."""
    style = style or MapStyle()
    if sel.y is None:
        raise ValueError("a map needs a Y dimension")
    sel = Selection.from_dict(sel.to_dict())
    da = detector(ds, sel.detector)
    cplx = np.iscomplexobj(da.values)
    name = quantity_name(sel, cplx)
    unit = "rad" if (sel.part == "arg" and cplx) else units_of(ds, sel.detector)
    red = reduce_cube(da, sel.x, sel.y, sel.slices, "complex" if cplx else sel.part,
                      runs=sel.runs, ds=ds)
    zc = np.asarray(red.data.values)
    ny, nx = zc.shape
    x, y = shown_coords(ds, red, sel.x, nx), shown_coords(ds, red, sel.y, ny)
    label = describe_slices(ds, sel) or sel.detector
    shown = runs_text(ds, sel)
    if shown:
        label += f" ({shown})"
    xi, xt = _axis_runs(ds, red, sel.x, sel)
    yi, yt = _axis_runs(ds, red, sel.y, sel)
    ref = None
    if style.ref != "none":
        zc = reference_values(zc, style, x, y)
        name, unit = _ref_name_unit(ds, sel, style, name, unit)
        label = f"{label} ({reference_text(ds, sel, style)})"
        ref = {"mode": style.ref, "op": style.ref_op, "i": int(style.ref_i)}
    values = np.asarray(apply_part(xr.DataArray(zc), sel.part).values, dtype=float)
    return MapData(x=x.copy(), y=y.copy(), values=values.copy(), label=label,
                   x_name=sel.x, x_unit=axis_unit(ds, sel, sel.x),
                   y_name=sel.y, y_unit=axis_unit(ds, sel, sel.y), z_name=name, z_unit=unit,
                   selection=sel, source=str(source) if source else None,
                   z=np.asarray(zc, dtype=complex).copy() if cplx else None,
                   held=held_values(ds, sel), ref=ref,
                   x_index=xi, y_index=yi, x_tags=xt, y_tags=yt)


#: a reference along one axis is along the other once the map is turned
_REF_TURNED = {"row": "column", "column": "row", "median_rows": "median_columns",
               "median_columns": "median_rows", "dd_y": "dd_x", "dd_x": "dd_y"}


def transpose_map(m: MapData) -> MapData:
    """The same map with X and Y swapped: rows become columns. A module that
    cuts maps into sweeps uses it to sweep along the axis it fits in (a
    VNA-FMR fit: frequency, whichever axis the viewer showed it on)."""
    sel = Selection.from_dict(m.selection.to_dict())
    sel.x, sel.y = m.y_name, m.x_name
    ref = None
    if m.ref is not None:
        ref = {**m.ref, "mode": _REF_TURNED.get(m.ref.get("mode"), m.ref.get("mode"))}
    return MapData(x=np.asarray(m.y).copy(), y=np.asarray(m.x).copy(),
                   values=np.asarray(m.values).T.copy(), label=m.label,
                   x_name=m.y_name, x_unit=m.y_unit, y_name=m.x_name, y_unit=m.x_unit,
                   z_name=m.z_name, z_unit=m.z_unit, selection=sel, source=m.source,
                   z=None if m.z is None else np.asarray(m.z).T.copy(),
                   held=dict(m.held), ref=ref,
                   x_index=m.y_index, y_index=m.x_index, x_tags=m.y_tags, y_tags=m.x_tags)


def pick_rows(y, count: int, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Indices of `count` lines spread evenly over the map (or over the part
    with lo <= y <= hi): when a whole map is too many sweeps to fit."""
    y = np.asarray(y, dtype=float)
    idx = np.arange(y.size)
    if lo is not None and hi is not None:
        lo, hi = min(lo, hi), max(lo, hi)
        idx = idx[(y >= lo) & (y <= hi)]
    if idx.size == 0 or count <= 0:
        return np.array([], dtype=int)
    if count >= idx.size:
        return idx
    return idx[np.unique(np.round(np.linspace(0, idx.size - 1, int(count))).astype(int))]


def map_rows(m: MapData, rows) -> MapData:
    """The same map with only these lines (rows = indices into m.y)."""
    rows = np.asarray(rows, dtype=int)
    return MapData(x=np.asarray(m.x).copy(), y=np.asarray(m.y)[rows].copy(),
                   values=np.asarray(m.values)[rows].copy(), label=m.label,
                   x_name=m.x_name, x_unit=m.x_unit, y_name=m.y_name, y_unit=m.y_unit,
                   z_name=m.z_name, z_unit=m.z_unit, selection=m.selection, source=m.source,
                   z=None if m.z is None else np.asarray(m.z)[rows].copy(),
                   held=dict(m.held), ref=None if m.ref is None else dict(m.ref),
                   x_index=m.x_index, x_tags=m.x_tags,
                   y_index=None if m.y_index is None else np.asarray(m.y_index)[rows].copy(),
                   y_tags=None if m.y_tags is None else [m.y_tags[int(r)] for r in rows])


def map_to_curves(m: MapData, columns=None) -> list[Curve]:
    """One curve per X value (per COLUMN of the map), running along Y -- how a
    module that takes curves (a fit of one sweep at a time) receives a map.
    Lukas's rule (2026-10-14): "the x axis is the parameter and the y axis is
    plotted". So a map with X = clMag.field (121 values) and Y = vna.freq (1601
    points) becomes 121 frequency sweeps, each labelled and held at its field
    ("clMag.field = -50 mT") -- ready for a dispersion. (Until 2026-10-14 it was
    one curve per ROW: 1601 field sweeps, and building/listing them was most
    of the ~30 s a send took.)

    columns: indices into m.x to make curves for (default: all). Pass them
    rather than cutting the map down first, so each curve's Selection still
    holds X at its index IN THE FILE (the notebook regenerates from it).
    """
    x = np.asarray(m.x, dtype=float)
    yax = np.asarray(m.y, dtype=float)
    # one transposed copy: each column becomes a contiguous row, copied once
    vals = np.ascontiguousarray(np.asarray(m.values, dtype=float).T)
    zt = None if m.z is None else np.ascontiguousarray(np.asarray(m.z, dtype=complex).T)
    base = m.selection.to_dict()
    cols = range(x.size) if columns is None else [int(i) for i in columns]
    out = []
    for i in cols:
        xv = float(x[i])
        sel = Selection.from_dict(base)
        # the index IN THE FILE: X may show one run of an up-and-down axis
        fi = i if m.x_index is None else int(np.asarray(m.x_index)[i])
        sel.slices[m.x_name] = Slice("at", fi)
        if m.ref:            # a cut of the referenced map (export.ref_line)
            sel.ref = {"x": m.x_name, "y": m.y_name, **m.ref}
        else:
            sel.runs.pop(m.x_name, None)     # X is held now, not shown
        sel.x, sel.y = m.y_name, None        # the curve runs along the map's Y
        if m.x_tags and m.x_unit == "point #":         # X drawn by point number
            label = f"{m.x_name} = {m.x_tags[i]}"
        else:
            label = run_label(f"{m.x_name} = {xv:g} {m.x_unit}".strip(),
                              m.x_tags[i] if m.x_tags else None)
        out.append(Curve(x=yax.copy(), y=vals[i].copy(), label=label,
                         x_name=m.y_name, x_unit=m.y_unit,
                         y_name=m.z_name, y_unit=m.z_unit, selection=sel, source=m.source,
                         z=None if zt is None else zt[i].copy(),
                         held={**m.held, m.x_name: (xv, m.x_unit)}))
    return out


def _ref_index(i: int, n: int) -> int:
    """Python-style: -1 is the last line (the highest field of a sweep)."""
    i = int(i) + n if int(i) < 0 else int(i)
    return min(max(i, 0), n - 1)


def reference_values(z: np.ndarray, style: MapStyle, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The map against a reference, on the COMPLEX values (before |z| or arg).

    A VNA's S21 carries the cables, the coupler and the amplifier as a
    frequency-dependent background far larger than the magnon line; divided by
    a line where the sample does nothing (the resonance pushed out of the band
    by a large field), only the sample is left -- and done on the complex
    numbers, arg of the ratio is the phase the SAMPLE adds.

      row / column     : every line divided by (or minus) one line, index ref_i
      median_rows / _columns : ... by the median line -- no reference measured:
                         a line that moves across the map is in few lines, so
                         the median is the background (real and imag separately)
      dd_y / dd_x      : derivative-divide (Maier-Flaig et al., PRB 2018):
                         (z[k+1] - z[k-1]) / ((c[k+1] - c[k-1]) z[k]) along the
                         axis -- the background cancels if it is slow along it;
                         units 1/axis, the first and last line are NaN
    NaN-aware; a zero in the reference gives a hole, not an infinity.
    """
    mode = style.ref
    if mode == "none":
        return z
    z = np.asarray(z)
    if not np.iscomplexobj(z):
        z = z.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode in ("dd_y", "dd_x"):
            axis = 0 if mode == "dd_y" else 1
            c = np.asarray(y if axis == 0 else x, dtype=float)
            zz = z if axis == 0 else z.T
            out = np.full(zz.shape, np.nan, dtype=zz.dtype)
            if zz.shape[0] >= 3:
                dc = (c[2:] - c[:-2])[:, None]
                out[1:-1] = (zz[2:] - zz[:-2]) / (dc * zz[1:-1])
            out[~np.isfinite(out)] = np.nan
            return out if axis == 0 else out.T
        if mode in ("row", "column"):
            n = z.shape[0 if mode == "row" else 1]
            i = _ref_index(style.ref_i, n)
            ref = z[i:i + 1, :] if mode == "row" else z[:, i:i + 1]
        elif mode in ("median_rows", "median_columns"):
            axis = 0 if mode == "median_rows" else 1
            ref = np.nanmedian(z.real, axis=axis, keepdims=True)
            if np.iscomplexobj(z):
                ref = ref + 1j * np.nanmedian(z.imag, axis=axis, keepdims=True)
        else:
            raise ValueError(f"unknown reference '{mode}' (have: {', '.join(MAP_REFS)})")
        if style.ref_op == "subtract":
            return z - ref
        out = z / ref
        out[~np.isfinite(out)] = np.nan
        return out


def reference_text(ds: xr.Dataset, sel: Selection, style: MapStyle) -> str:
    """'÷ field = 800 mT' -- what the map was referenced to, in the quantity."""
    op = "−" if style.ref_op == "subtract" else "÷"
    if style.ref in ("row", "column"):
        dim = sel.y if style.ref == "row" else sel.x
        idx = shown_indices(ds, sel, dim)
        return f"{op} {dim} = {coord_text(ds, dim, int(idx[_ref_index(style.ref_i, idx.size)]))}"
    if style.ref in ("median_rows", "median_columns"):
        return f"{op} median over {sel.y if style.ref == 'median_rows' else sel.x}"
    if style.ref in ("dd_y", "dd_x"):
        return f"d/d{sel.y if style.ref == 'dd_y' else sel.x} ÷"
    return ""


def style_values(z: np.ndarray, style: MapStyle) -> np.ndarray:
    """Per-line normalisation, then log. The ORDER matters: normalising after the
    log would scale decades, which is not what anyone means."""
    z = np.array(z, dtype=float)
    if style.norm in ("rows", "columns"):
        axis = 1 if style.norm == "rows" else 0     # a row runs along X
        # An all-NaN line (not measured yet) has no peak; it stays NaN, quietly.
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            peak = np.nanmax(np.abs(z), axis=axis, keepdims=True)
            z = np.where(peak > 0, z / peak, z)
    if style.log:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log10(np.abs(z))
        z[~np.isfinite(z)] = np.nan                 # log of 0 is a hole, not -inf
    return z


def map_levels(z: np.ndarray, style: MapStyle) -> tuple[float, float]:
    """The colour range. Percentiles, so one hot pixel does not flatten the map;
    symmetric about zero for signals that change sign (a Kerr map)."""
    fin = z[np.isfinite(z)]
    if not style.auto:
        lo, hi = float(style.lo), float(style.hi)
    elif fin.size:
        # 0.1 / 99.9, not 1 / 99: a sharp resonance on a VNA map (a 10 MHz
        # line in 16 GHz) covers < 1 % of the pixels, and 1 / 99 put the limits
        # inside the noise -- the map showed noise at full contrast and the line
        # clipped (2026-09-30). One hot pixel in 1000 is still ignored.
        lo, hi = float(np.percentile(fin, 0.1)), float(np.percentile(fin, 99.9))
    else:
        lo, hi = 0.0, 1.0
    if style.symmetric:
        a = max(abs(lo), abs(hi))
        lo, hi = -a, a
    if hi <= lo:
        hi = lo + (abs(lo) * 1e-6 or 1e-9)
    return lo, hi


def is_uniform(c: np.ndarray, rtol: float = 1e-6) -> bool:
    """Evenly spaced? A matrix (image, Origin matrix) only knows start and end."""
    c = np.asarray(c, dtype=float)
    if c.size < 3:
        return True
    d = np.diff(c)
    return bool(np.allclose(d, d[0], rtol=rtol, atol=abs(d[0]) * rtol + 1e-15))


def axis_title(name: str, unit: str) -> str:
    return f"{name} ({unit})" if unit else name


def curve_axis_titles(shown: list[Curve], norm: str = "none") -> tuple[str, str]:
    """(X title, Y title) for a set of overlaid curves.

    A unit is printed only when EVERY curve has it: a Kerr curve (mdeg) and a
    reflectivity curve (V) on one plot have no common unit, and labelling the
    axis with the first curve's unit would be wrong for the other one.
    """
    if not shown:
        return "", ""
    xnames = list(dict.fromkeys(c.x_name for c in shown))
    xunits = {c.x_unit for c in shown}
    xt = axis_title(" | ".join(xnames), xunits.pop() if len(xunits) == 1 else "")
    ynames = {c.y_name for c in shown}
    yname = ynames.pop() if len(ynames) == 1 else "signal"
    if norm != "none":
        return xt, axis_title(yname, NORMS[norm])
    yunits = {c.y_unit for c in shown}
    return xt, axis_title(yname, yunits.pop() if len(yunits) == 1 else "")


# ─────────────────────────────── text files ───────────────────────────────────

def _delimiter(path: Path) -> str:
    return "," if path.suffix.lower() == ".csv" else "\t"


def _encoding(path: Path) -> str:
    """UTF-8 with a byte-order mark for .csv, plain UTF-8 for everything else.

    A .csv is usually opened by double-clicking it into Excel, which reads a
    file WITHOUT a BOM in the Windows codepage -- so the units row turns "µm"
    into "Âµm". The BOM is how Excel learns the file is UTF-8. A .dat goes to
    numpy / gnuplot / Origin instead, some of which read the BOM as part of
    the first column name, so that one stays without.
    """
    return "utf-8-sig" if path.suffix.lower() == ".csv" else "utf-8"


def _num(v: float) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))                  # a run number, a direction: no ".0"
    return "" if not np.isfinite(v) else repr(float(v))


def _run_columns(c: Curve) -> list[tuple[np.ndarray, str, str, str]]:
    """(values, name, unit, note) of the two columns that say which run each
    point of an up-and-down X is on -- the file's own <x>_segment and
    <x>_direction, so a spreadsheet can split the loop again."""
    if c.segment is None or not c.runs:
        return []
    segs = Segments(np.asarray(c.segment, dtype=int), list(c.runs))
    return [(segs.seg.astype(int), f"{c.x_name}_segment", "", "run (0 = first)"),
            (segs.directions(), f"{c.x_name}_direction", "", "+1 up, -1 down")]


def _same_runs(a: Curve, b: Curve) -> bool:
    if a.segment is None or b.segment is None:
        return a.segment is None and b.segment is None
    return np.array_equal(a.segment, b.segment)


def share_one_x(shown: list[Curve]) -> bool:
    """Can these curves be written against ONE X column?

    Only when the X values AND the quantity they measure are the same. Equal
    numbers are not enough: a row cut (along x) and a column cut (along y) of a
    square raster both run 0 ... 20 um, and one shared column named "x" would
    label the y cut as a function of x.
    """
    first = shown[0]
    return all(c.x_name == first.x_name and c.x_unit == first.x_unit
               and c.x.shape == first.x.shape and np.array_equal(c.x, first.x)
               and _same_runs(c, first)
               for c in shown)


def write_curves(path: str | Path, curves: list[Curve], norm: str = "none",
                 offset: float = 0.0) -> Path:
    """Visible curves as columns. Row 1 = long names, row 2 = units, row 3 = the
    curve label -- the three header rows Origin's import recognises as
    Long Name / Units / Comments, and easy to skip anywhere else
    (`pandas.read_csv(path, skiprows=[1, 2])`).

    Curves that share their X values exactly get ONE X column (the usual case:
    one field sweep, several frequencies). Otherwise each curve brings its own
    X column, and shorter columns end in empty cells rather than invented zeros.

    An X that goes up AND down (a hysteresis loop) is followed by two more
    columns, `<x>_segment` (the run, 0 = first) and `<x>_direction` (+1 up,
    -1 down): the values alone repeat, the run number tells the branches apart.
    """
    path = Path(path)
    shown = [c for c in curves if c.visible]
    if not shown:
        raise ValueError("no visible curves to write")
    ys = displayed_y(curves, norm, offset)
    shared = share_one_x(shown)
    cols, names, units, notes = [], [], [], []
    if shared:
        cols.append(shown[0].x); names.append(shown[0].x_name)
        units.append(shown[0].x_unit); notes.append("")
        for v, nm, un, no in _run_columns(shown[0]):
            cols.append(v); names.append(nm); units.append(un); notes.append(no)
    for c, y in zip(shown, ys):
        if not shared:
            cols.append(c.x); names.append(c.x_name); units.append(c.x_unit); notes.append(c.label)
            for v, nm, un, no in _run_columns(c):
                cols.append(v); names.append(nm); units.append(un); notes.append(no)
        cols.append(y); names.append(c.y_name)
        units.append(c.y_unit if norm == "none" else "")
        notes.append(c.label + ("" if norm == "none" else f" [{NORMS[norm]}]"))
    n = max(len(c) for c in cols)
    with open(path, "w", newline="", encoding=_encoding(path)) as f:
        w = csv.writer(f, delimiter=_delimiter(path))
        w.writerow(names); w.writerow(units); w.writerow(notes)
        for i in range(n):
            w.writerow([_num(c[i]) if i < len(c) else "" for c in cols])
    return path


def write_map(path: str | Path, m: Map, fmt: str = "matrix") -> Path:
    """A map as text.

    fmt="matrix": first row = X values, first column = Y values, the rest is Z.
                  Compact; what you paste into a spreadsheet.
    fmt="xyz":    three columns, one row per pixel. Longer, but it is what
                  Origin/Igor/gnuplot turn into a colour map with no questions,
                  and it survives uneven spacing.
    """
    path = Path(path)
    with open(path, "w", newline="", encoding=_encoding(path)) as f:
        w = csv.writer(f, delimiter=_delimiter(path))
        if fmt == "matrix":
            corner = f"{axis_title(m.y_name, m.y_unit)} \\ {axis_title(m.x_name, m.x_unit)}"
            w.writerow([corner] + [_num(v) for v in m.x])
            for j, yv in enumerate(m.y):
                w.writerow([_num(yv)] + [_num(v) for v in m.z[j]])
        elif fmt == "xyz":
            w.writerow([m.x_name, m.y_name, m.z_name])
            w.writerow([m.x_unit, m.y_unit, m.z_unit])
            for j, yv in enumerate(m.y):
                for i, xv in enumerate(m.x):
                    w.writerow([_num(xv), _num(yv), _num(m.z[j, i])])
        else:
            raise ValueError(f"unknown map format '{fmt}' (matrix or xyz)")
    return path


# ──────────────────────────────── figures ─────────────────────────────────────

def _figure(size):
    # Figure() directly, never pyplot: pyplot would pick an interactive backend
    # and open windows from inside a Qt app.
    from matplotlib.figure import Figure
    return Figure(figsize=size, dpi=100, layout="constrained")


def figure_curves(curves: list[Curve], norm: str = "none", offset: float = 0.0,
                  logy: bool = False, title: str = "", size=(6.4, 4.2)):
    """Publication-style (white) figure of the visible curves."""
    fig = _figure(size)
    ax = fig.add_subplot()
    shown = [c for c in curves if c.visible]
    n_lines = 0
    for c, y in zip(shown, displayed_y(curves, norm, offset)):
        # one line per run of an up-and-down X, each its own colour
        for run, idx in curve_runs(c):
            ax.plot(np.asarray(c.x)[idx], np.asarray(y)[idx], marker="o", ms=2.5,
                    lw=1.2, label=run_label(c.label, run))
            n_lines += 1
    if shown:
        xt, yt = curve_axis_titles(shown, norm)       # a unit only if all share it
        ax.set_xlabel(xt)
        ax.set_ylabel(yt)
    if logy:
        ax.set_yscale("log")
    if 1 < n_lines <= 16:
        ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def figure_map(m: Map, style: MapStyle | None = None, title: str = "", size=(6.4, 4.8)):
    style = style or MapStyle()
    fig = _figure(size)
    ax = fig.add_subplot()
    cmap = mpl_cmap(style.cmap, style.invert)
    # pcolormesh with edges from the coordinates: correct for uneven spacing too,
    # where imshow would silently stretch the pixels evenly.
    mesh = ax.pcolormesh(edges(m.x), edges(m.y), np.ma.masked_invalid(m.z),
                         cmap=cmap, vmin=m.levels[0], vmax=m.levels[1], shading="flat")
    ax.set_xlabel(axis_title(m.x_name, m.x_unit))
    ax.set_ylabel(axis_title(m.y_name, m.y_unit))
    fig.colorbar(mesh, ax=ax, label=axis_title(m.z_name, m.z_unit))
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def mpl_cmap(name: str, invert: bool = False):
    import matplotlib
    try:
        cmap = matplotlib.colormaps[CMAPS.get(name, name)]
    except KeyError:
        cmap = matplotlib.colormaps["magma"]
    return cmap.reversed() if invert else cmap


def edges(c: np.ndarray) -> np.ndarray:
    """Pixel edges from pixel centres (half a step beyond each end)."""
    c = np.asarray(c, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def save_figure(fig, path: str | Path, dpi: int = 300) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path


# ─────────────────────────────── notebooks ────────────────────────────────────

# The notebook carries its OWN copy of the reduction instead of importing
# this package: it has to run on a laptop that has never seen this repository
# (numpy + xarray + h5netcdf + matplotlib is all it needs). The price is two
# implementations of one idea, so tests/test_export.py executes the
# generated code and checks it produces exactly what view.py does.
_NB_HELPERS = '''\
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def load(path):
    """Open a AaltoFlow measurement (.nc) completely, then close the file."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return ds.load()


def detector(ds, name):
    """A detector by name. Complex ones are stored as <name>_real + <name>_imag."""
    if f"{name}_real" in ds and f"{name}_imag" in ds:
        return ds[f"{name}_real"] + 1j * ds[f"{name}_imag"]
    return ds[name]


def runs_of(ds, dim):
    """(run number per point, one dict per run) of an axis that goes up AND
    down (a hysteresis loop), or None. From the file's <dim>_segment; an older
    file of a typed loop is split where its values turn back."""
    if dim not in ds.coords:
        return None
    if f"{dim}_segment" in ds.coords:
        seg = np.asarray(ds[f"{dim}_segment"].values).astype(int)
        try:
            info = json.loads(ds[dim].attrs.get("segments_json") or "[]")
        except ValueError:
            info = []
        if len(info) != (seg.max() + 1 if seg.size else 0):
            d = np.asarray(ds[f"{dim}_direction"].values).astype(int)
            info = [{"direction": int(d[seg == k][0]), "joined": int(k > 0)}
                    for k in range(seg.max() + 1)]
        return seg, info
    try:
        v = np.asarray(ds[dim].values, dtype=float)
    except (TypeError, ValueError):
        return None
    if v.ndim != 1 or v.size < 3 or not np.all(np.isfinite(v)):
        return None
    seg, dirs, k, cur = np.zeros(v.size, dtype=int), [], 0, 0
    for i in range(1, v.size):
        step = int(np.sign(v[i] - v[i - 1]))
        if step != 0 and cur != 0 and step != cur:
            k += 1
            dirs.append(cur)
        if step != 0:
            cur = step
        seg[i] = k
    if k == 0:
        return None
    dirs.append(cur or 1)
    return seg, [{"direction": dirs[j], "joined": int(j > 0)} for j in range(k + 1)]


def run_name(info, k):
    way = "up" if info[k].get("direction", 1) > 0 else "down"
    return way if len(info) <= 2 else f"seg {k + 1} {way}"


def pick_runs(da, ds, x, y, runs):
    """Which run of an up-and-down image axis to show: a map shows ONE run
    (default the first), ascending; 'index' = every point against its number;
    a line along x shows every run ('lines', drawn one line each).
    Returns (da, {dim: file index of each point shown})."""
    runs, shown = runs or {}, {}
    for dim in [d for d in (x, y) if d]:
        r = runs_of(ds, dim)
        if r is None or dim not in da.dims or r[0].size != da.sizes[dim]:
            continue
        seg, info = r
        line = y is None and dim == x
        c = runs.get(dim)
        ok = (c == "index" or (c == "lines" and line)
              or (isinstance(c, int) and not isinstance(c, bool) and 0 <= c < len(info)))
        c = c if ok else ("lines" if line else 0)
        n = da.sizes[dim]
        if c == "index":
            da = da.assign_coords({dim: np.arange(n, dtype=float)})
            shown[dim] = np.arange(n)
        elif c == "lines":
            shown[dim] = np.arange(n)
        else:
            idx = np.flatnonzero(seg == int(c))
            v = np.asarray(da[dim].values, dtype=float)[idx]
            if idx.size > 1 and v[-1] < v[0]:
                idx = idx[::-1]
            da = da.isel({dim: idx})
            shown[dim] = idx
    return da, shown


def reduce(da, x, y=None, slices=None, part="abs", runs=None, ds=None, with_index=False):
    """Keep dims (y, x); hold ('at') or average ('mean', NaN-skipping) the rest.

    A complex detector is averaged as a complex number FIRST and turned into
    |z| / arg z afterwards (coherent averaging), as in the viewer. runs/ds:
    the run shown of an axis that goes up and down (pick_runs).
    """
    slices = slices or {}
    shown = {}
    if ds is not None:
        da, shown = pick_runs(da, ds, x, y, runs)
    others = [d for d in da.dims if d not in (x, y)]
    mean_dims = []
    for d in others:
        n = da.sizes[d]
        s = slices.get(d, {"mode": "at", "i0": 0, "i1": None})
        i0 = min(max(int(s["i0"]), 0), n - 1)
        i1 = i0 if s["mode"] == "at" else (n - 1 if s.get("i1") is None else min(max(int(s["i1"]), 0), n - 1))
        i0, i1 = min(i0, i1), max(i0, i1)
        da = da.isel({d: slice(i0, i1 + 1)})
        if i1 > i0:
            mean_dims.append(d)
    if mean_dims:
        da = da.mean(dim=mean_dims, skipna=True)
    for d in others:
        if d in da.dims:
            da = da.isel({d: 0})
    if np.iscomplexobj(da.values) and part != "complex":
        da = xr.DataArray(part_of(da.values, part), dims=da.dims, coords=da.coords)
    da = da.transpose(*[d for d in (y, x) if d])
    return (da, shown) if with_index else da


def axis_of(da, dim):
    """The coordinate of an image axis as shown (one run, or point numbers)."""
    return np.asarray(da[dim].values, dtype=float) if dim in da.coords else np.arange(da.sizes[dim])


def part_of(z, part):
    """Complex -> the part on screen; a real array stays as it is."""
    if not np.iscomplexobj(z):
        return np.asarray(z, dtype=float)
    return {"abs": np.abs, "arg": np.angle, "real": np.real, "imag": np.imag}[part](z)


def reference(z, mode, op, i, xs, ys):
    """The map against a reference, on the COMPLEX values (before |z| / arg).
    row/column: every line divided by (or minus) line i; median_rows/_columns:
    by the median line; dd_y/dd_x: derivative-divide (z[k+1] - z[k-1]) /
    ((c[k+1] - c[k-1]) z[k]) along that axis."""
    if mode == "none":
        return z
    z = np.asarray(z) if np.iscomplexobj(z) else np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode in ("dd_y", "dd_x"):
            zz, c = (z, ys) if mode == "dd_y" else (z.T, xs)
            out = np.full(zz.shape, np.nan, dtype=zz.dtype)
            if zz.shape[0] >= 3:
                dc = (np.asarray(c, dtype=float)[2:] - np.asarray(c, dtype=float)[:-2])[:, None]
                out[1:-1] = (zz[2:] - zz[:-2]) / (dc * zz[1:-1])
            out[~np.isfinite(out)] = np.nan
            return out if mode == "dd_y" else out.T
        if mode in ("row", "column"):
            n = z.shape[0 if mode == "row" else 1]
            i = int(i) + n if int(i) < 0 else int(i)
            i = min(max(i, 0), n - 1)
            ref = z[i:i + 1, :] if mode == "row" else z[:, i:i + 1]
        else:
            axis = 0 if mode == "median_rows" else 1
            ref = np.nanmedian(z.real, axis=axis, keepdims=True)
            if np.iscomplexobj(z):
                ref = ref + 1j * np.nanmedian(z.imag, axis=axis, keepdims=True)
        if op == "subtract":
            return z - ref
        out = z / ref
        out[~np.isfinite(out)] = np.nan
        return out


def ref_line(ds, c):
    """A cut of a referenced map: the map referenced as a whole, then the line."""
    r = c["ref"]
    held = r["y"] if c["x"] == r["x"] else r["x"]
    slices = {d: s for d, s in c["slices"].items() if d != held}
    # an up-and-down axis: the run chosen, else every point in measured order
    runs = dict(c.get("runs", {}))
    every = [d for d in (r["x"], r["y"]) if runs.get(d) in (None, "lines")]
    runs.update({d: "index" for d in every})
    m, shown = reduce(detector(ds, c["detector"]), r["x"], r["y"], slices, "complex",
                      runs, ds, with_index=True)
    xs, ys = [np.asarray(ds[d].values, dtype=float) if d in every and d in ds.coords
              else axis_of(m, d) for d in (r["x"], r["y"])]
    z = reference(m.values, r["mode"], r.get("op", "divide"), r.get("i", -1), xs, ys)
    n = ds.sizes[held]
    k = min(max(int(c["slices"].get(held, {"i0": 0})["i0"]), 0), n - 1)
    if held in shown:                        # the map shows one run of it
        k = int(np.flatnonzero(shown[held] == k)[0])
    return part_of(z[k] if held == r["y"] else z[:, k], c["part"]), (xs if held == r["y"] else ys)


def normalize(v, mode):
    v = np.asarray(v, dtype=float)
    fin = v[np.isfinite(v)]
    if mode == "none" or fin.size == 0:
        return v
    # the same guards as the viewer: a flat or zero curve is left as it is
    # rather than divided by zero (which would draw NaN / inf instead)
    if mode == "peak":
        m = np.max(np.abs(fin))
        return v / m if m > 0 else v
    if mode == "minmax":
        lo, hi = fin.min(), fin.max()
        return (v - lo) / (hi - lo) if hi > lo else v - lo
    if mode == "first":
        return v / fin[0] if fin[0] != 0 else v
    if mode == "zero_mean":
        return v - fin.mean()
    raise ValueError(mode)
'''


def _py(obj) -> str:
    """A Python literal for a code cell. NOT json.dumps: JSON spells False and
    None as false and null, which is a NameError the moment the cell runs."""
    return pprint.pformat(obj, indent=1, width=88, sort_dicts=False)


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(True)}


def _rel(source: str, nb_dir: Path) -> str:
    """Path to the data as seen from the notebook: relative when they are on one
    drive (the notebook and its data can then move together), else absolute."""
    try:
        return Path(os.path.relpath(source, nb_dir)).as_posix()
    except ValueError:                      # different drive letters on Windows
        return Path(source).as_posix()


def notebook_cells(nb_dir: Path, m: Map | None = None, style: MapStyle | None = None,
                   curves: list[Curve] | None = None, norm: str = "none",
                   offset: float = 0.0, logy: bool = False,
                   loading_script: str | Path | None = None) -> list[dict]:
    """The cells of the notebook, as a list (so a test can execute the code).

    loading_script: the viewer read the files through this script
    (aaltoview/loading.py) -- the notebook reads them through it too."""
    curves = [c for c in (curves or []) if c.visible]
    sources = []
    for s in ([m.source] if m else []) + [c.source for c in curves]:
        if s is None:
            raise ValueError("this view comes from an unsaved run -- save the "
                             "measurement first, the notebook reads it from a file")
        if s not in sources:
            sources.append(s)

    cells = [_md("# AaltoView\n\n"
                 "Generated by the AaltoView. Every number below is "
                 "recomputed from the measurement files, so you can change a "
                 "range or an average and run it again.\n\n"
                 + "\n".join(f"* `{Path(s).name}`" for s in sources) + "\n"),
             _code(_NB_HELPERS)]
    files = "FILES = " + _py({f"f{i}": _rel(s, nb_dir) for i, s in enumerate(sources)})
    if loading_script is None:
        cells.append(_code(files + "\ndata = {key: load(path) for key, path in FILES.items()}\n"
                                   "data['f0']"))
    else:
        # the correction the viewer applied, applied here the same way (the
        # script may use aaltoview.loading's helpers: AaltoView must be installed)
        cells.append(_md(f"The files are read through the loading script "
                         f"`{Path(loading_script).name}`, as they were in the viewer."))
        cells.append(_code(
            files + f"\nLOADING_SCRIPT = {_rel(str(loading_script), nb_dir)!r}\n\n"
            "import importlib.util\n"
            "_spec = importlib.util.spec_from_file_location('loading_script', LOADING_SCRIPT)\n"
            "_script = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_script)\n"
            "data = {key: _script.load(load(path), path) for key, path in FILES.items()}\n"
            "data['f0']"))
    key = {s: f"f{i}" for i, s in enumerate(sources)}

    if m is not None:
        style = style or MapStyle()
        sel = m.selection.to_dict()
        cmap = CMAPS.get(style.cmap, style.cmap)
        cells.append(_md("## Map\n\n" + f"`{m.z_name}` against `{m.x_name}` and `{m.y_name}`."))
        cells.append(_code(
            f"sel = {_py(sel)}\n"
            f"style = {_py(style.to_dict())}\n\n"
            f"ds = data[{key[m.source]!r}]\n"
            "m = reduce(detector(ds, sel['detector']), sel['x'], sel['y'], sel['slices'], 'complex',\n"
            "           sel.get('runs'), ds)\n"
            "xs, ys = axis_of(m, sel['x']), axis_of(m, sel['y'])   # one run of an up-and-down axis\n"
            "z = reference(m.values, style['ref'], style['ref_op'], style['ref_i'], xs, ys)\n"
            "z = part_of(z, sel['part'])\n"
            "if style['norm'] in ('rows', 'columns'):\n"
            "    axis = 1 if style['norm'] == 'rows' else 0\n"
            "    peak = np.nanmax(np.abs(z), axis=axis, keepdims=True)\n"
            "    z = np.where(peak > 0, z / peak, z)\n"
            "if style['log']:\n"
            "    with np.errstate(divide='ignore'):\n"
            "        z = np.log10(np.abs(z))\n"
            "    z[~np.isfinite(z)] = np.nan\n"
            f"vmin, vmax = {m.levels[0]!r}, {m.levels[1]!r}   # the limits used in the viewer\n\n"
            "fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')\n"
            f"cmap = plt.get_cmap({cmap!r})\n"
            "if style['invert']:\n"
            "    cmap = cmap.reversed()\n"
            "mesh = ax.pcolormesh(xs, ys, z, cmap=cmap,\n"
            "                     vmin=vmin, vmax=vmax, shading='nearest')\n"
            f"ax.set_xlabel({axis_title(m.x_name, m.x_unit)!r})\n"
            f"ax.set_ylabel({axis_title(m.y_name, m.y_unit)!r})\n"
            f"fig.colorbar(mesh, ax=ax, label={axis_title(m.z_name, m.z_unit)!r})\n"
            "plt.show()"))

    if curves:
        spec = [{"file": key[c.source], "label": c.label, **c.selection.to_dict()}
                for c in curves]
        cells.append(_md("## Curves\n\n" + f"{len(curves)} curve(s), normalisation: "
                         f"{NORMS[norm]}" + (f", stacked by {offset:g}" if offset else "") + "."))
        cells.append(_code(
            f"CURVES = {_py(spec)}\n"
            f"NORM, OFFSET, LOGY = {norm!r}, {offset!r}, {logy!r}\n\n"
            "fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')\n"
            "lines = []\n"
            "for k, c in enumerate(CURVES):\n"
            "    ds = data[c['file']]\n"
            "    if c.get('ref'):                  # a cut of a referenced map\n"
            "        y, x = ref_line(ds, c)\n"
            "    else:\n"
            "        r = reduce(detector(ds, c['detector']), c['x'], None, c['slices'], c['part'],\n"
            "                   c.get('runs'), ds)\n"
            "        y, x = r.values, axis_of(r, c['x'])\n"
            "    yv = normalize(y, NORM) + k * OFFSET\n"
            "    lines.append((x, yv))\n"
            "    # an X that goes up AND down: one line per run, the loop closed as measured\n"
            "    runs = runs_of(ds, c['x'])\n"
            "    if runs is not None and runs[0].size == x.size and c.get('runs', {}).get(c['x'], 'lines') == 'lines':\n"
            "        seg, info = runs\n"
            "        for j in range(len(info)):\n"
            "            idx = np.flatnonzero(seg == j)\n"
            "            if j > 0 and info[j].get('joined') and idx[0] > 0:\n"
            "                idx = np.r_[idx[0] - 1, idx]\n"
            "            ax.plot(x[idx], yv[idx], marker='o', ms=2.5, lw=1.2,\n"
            "                    label=c['label'] + ' · ' + run_name(info, j))\n"
            "    else:\n"
            "        ax.plot(x, yv, marker='o', ms=2.5, lw=1.2, label=c['label'])\n"
            f"ax.set_xlabel({curve_axis_titles(curves, norm)[0]!r})\n"
            f"ax.set_ylabel({curve_axis_titles(curves, norm)[1]!r})\n"
            "if LOGY:\n"
            "    ax.set_yscale('log')\n"
            "ax.legend(fontsize=7, frameon=False)\n"
            "plt.show()"))
    return cells


def write_notebook(path: str | Path, **kw) -> Path:
    """Write a .ipynb (nbformat 4, plain JSON -- no jupyter needed to create it)."""
    path = Path(path)
    nb = {"cells": notebook_cells(path.parent, **kw),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path
