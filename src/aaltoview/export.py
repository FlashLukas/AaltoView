"""export.py -- getting a view OUT of the viewer: curves, maps, files, notebooks.

The viewer (aaltoview/apps/viewer.py) is the successor of the LabVIEW AaltoView. What it
shows is always one of two things, and both are described here as DATA, with
no Qt anywhere, so every export can be tested without a screen:

* a MAP  -- `Selection(detector, x, y, slices)` reduced to (y, x), drawn with a
  `MapStyle` (colour map, limits, symmetric, log, per-line normalisation);
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

from .view import Reduced, Slice, coord_text, detector, reduce_cube

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

    def to_dict(self) -> dict:
        return {"detector": self.detector, "x": self.x, "y": self.y, "part": self.part,
                "slices": {d: {"mode": s.mode, "i0": int(s.i0),
                               "i1": None if s.i1 is None else int(s.i1)}
                           for d, s in self.slices.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        return cls(detector=d["detector"], x=d["x"], y=d.get("y"),
                   part=d.get("part", "abs"),
                   slices={k: Slice(v["mode"], v["i0"], v.get("i1"))
                           for k, v in d.get("slices", {}).items()})


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


@dataclass
class MapStyle:
    cmap: str = "magma"
    invert: bool = False            # flip the colour map
    symmetric: bool = False         # limits +-max, centred on zero (Kerr signals)
    auto: bool = True               # limits from the 1st/99th percentile
    lo: float = 0.0                 # used when auto is False
    hi: float = 1.0
    log: bool = False               # log10 |value|
    norm: str = "none"              # MAP_NORMS

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
        elif i0 == 0 and i1 == da.sizes[d] - 1:
            bits.append(f"{d} mean")
        else:
            bits.append(f"{d} mean {coord_text(ds, d, i0)} ... {coord_text(ds, d, i1)}")
    return ", ".join(bits)


def reduce(ds: xr.Dataset, sel: Selection) -> Reduced:
    return reduce_cube(detector(ds, sel.detector), sel.x, sel.y, sel.slices, sel.part)


def make_curve(ds: xr.Dataset, sel: Selection, source: str | Path | None = None,
               label: str | None = None) -> Curve:
    """Freeze the current 1-D selection into a Curve (the numbers are copied)."""
    sel = Selection.from_dict(sel.to_dict())        # detach from the caller's dict
    sel.y = None
    da = detector(ds, sel.detector)
    red = reduce_cube(da, sel.x, None, sel.slices, sel.part)
    y = np.asarray(red.data.values, dtype=float).copy()
    x = coords_of(ds, sel.x, y.size).copy()
    if label is None:
        label = describe_slices(ds, sel) or sel.detector
    return Curve(x=x, y=y, label=label, x_name=sel.x, x_unit=units_of(ds, sel.x),
                 y_name=quantity_name(sel, np.iscomplexobj(da.values)),
                 y_unit="rad" if (sel.part == "arg" and np.iscomplexobj(da.values))
                 else units_of(ds, sel.detector),
                 selection=sel, source=str(source) if source else None)


def curves_along(ds: xr.Dataset, sel: Selection, dim: str, indices,
                 source: str | Path | None = None) -> list[Curve]:
    """One curve per chosen value of `dim` -- AaltoView's multi-select parameter list.

    Every other dim keeps what `sel` says about it; `dim` is held at each index
    in turn.
    """
    out = []
    for i in indices:
        s = Selection.from_dict(sel.to_dict())
        s.slices[dim] = Slice("at", int(i))
        out.append(make_curve(ds, s, source))
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
    red = reduce_cube(da, sel.x, sel.y, sel.slices, sel.part)
    z = style_values(np.asarray(red.data.values, dtype=float), style)
    ny, nx = z.shape
    name = quantity_name(sel, np.iscomplexobj(da.values))
    unit = "rad" if (sel.part == "arg" and np.iscomplexobj(da.values)) else units_of(ds, sel.detector)
    if style.norm != "none":
        name, unit = f"{name} (normalised per {style.norm[:-1]})", ""
    if style.log:
        name, unit = f"log10 {name}", ""
    m = Map(z=z, x=coords_of(ds, sel.x, nx), y=coords_of(ds, sel.y, ny),
            x_name=sel.x, x_unit=units_of(ds, sel.x),
            y_name=sel.y, y_unit=units_of(ds, sel.y),
            z_name=name, z_unit=unit, levels=map_levels(z, style),
            selection=sel, source=str(source) if source else None)
    return m, red


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
        lo, hi = float(np.percentile(fin, 1)), float(np.percentile(fin, 99))
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


# ─────────────────────────────── text files ───────────────────────────────────

def _delimiter(path: Path) -> str:
    return "," if path.suffix.lower() == ".csv" else "\t"


def _num(v: float) -> str:
    return "" if not np.isfinite(v) else repr(float(v))


def write_curves(path: str | Path, curves: list[Curve], norm: str = "none",
                 offset: float = 0.0) -> Path:
    """Visible curves as columns. Row 1 = long names, row 2 = units, row 3 = the
    curve label -- the three header rows Origin's import recognises as
    Long Name / Units / Comments, and easy to skip anywhere else
    (`pandas.read_csv(path, skiprows=[1, 2])`).

    Curves that share their X values exactly get ONE X column (the usual case:
    one field sweep, several frequencies). Otherwise each curve brings its own
    X column, and shorter columns end in empty cells rather than invented zeros.
    """
    path = Path(path)
    shown = [c for c in curves if c.visible]
    if not shown:
        raise ValueError("no visible curves to write")
    ys = displayed_y(curves, norm, offset)
    shared = all(c.x.shape == shown[0].x.shape and np.array_equal(c.x, shown[0].x)
                 for c in shown)
    cols, names, units, notes = [], [], [], []
    if shared:
        cols.append(shown[0].x); names.append(shown[0].x_name)
        units.append(shown[0].x_unit); notes.append("")
    for c, y in zip(shown, ys):
        if not shared:
            cols.append(c.x); names.append(c.x_name); units.append(c.x_unit); notes.append(c.label)
        cols.append(y); names.append(c.y_name)
        units.append(c.y_unit if norm == "none" else "")
        notes.append(c.label + ("" if norm == "none" else f" [{NORMS[norm]}]"))
    n = max(len(c) for c in cols)
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    for c, y in zip(shown, displayed_y(curves, norm, offset)):
        ax.plot(c.x, y, marker="o", ms=2.5, lw=1.2, label=c.label)
    if shown:
        ax.set_xlabel(axis_title(shown[0].x_name, shown[0].x_unit))
        yl = shown[0].y_name if len({c.y_name for c in shown}) == 1 else "signal"
        ax.set_ylabel(axis_title(yl, shown[0].y_unit if norm == "none" else
                                 NORMS[norm]))
    if logy:
        ax.set_yscale("log")
    if 1 < len(shown) <= 16:
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


def reduce(da, x, y=None, slices=None, part="abs"):
    """Keep dims (y, x); hold ('at') or average ('mean', NaN-skipping) the rest.

    A complex detector is averaged as a complex number FIRST and turned into
    |z| / arg z afterwards (coherent averaging), as in the viewer.
    """
    slices = slices or {}
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
    if np.iscomplexobj(da.values):
        fn = {"abs": np.abs, "arg": np.angle, "real": np.real, "imag": np.imag}[part]
        da = xr.DataArray(fn(da.values), dims=da.dims, coords=da.coords)
    return da.transpose(*[d for d in (y, x) if d])


def normalize(v, mode):
    v = np.asarray(v, dtype=float)
    fin = v[np.isfinite(v)]
    if mode == "none" or fin.size == 0:
        return v
    if mode == "peak":
        return v / np.max(np.abs(fin))
    if mode == "minmax":
        return (v - fin.min()) / (fin.max() - fin.min())
    if mode == "first":
        return v / fin[0]
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
                   offset: float = 0.0, logy: bool = False) -> list[dict]:
    """The cells of the notebook, as a list (so a test can execute the code)."""
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
             _code(_NB_HELPERS),
             _code("FILES = " + _py({f"f{i}": _rel(s, nb_dir)
                                    for i, s in enumerate(sources)})
                   + "\ndata = {key: load(path) for key, path in FILES.items()}\n"
                   "data['f0']")]
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
            "m = reduce(detector(ds, sel['detector']), sel['x'], sel['y'], sel['slices'], sel['part'])\n"
            "z = m.values.astype(float)\n"
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
            "xs = ds[sel['x']].values if sel['x'] in ds.coords else np.arange(z.shape[1])\n"
            "ys = ds[sel['y']].values if sel['y'] in ds.coords else np.arange(z.shape[0])\n"
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
            "    y = reduce(detector(ds, c['detector']), c['x'], None, c['slices'], c['part'])\n"
            "    x = ds[c['x']].values if c['x'] in ds.coords else np.arange(y.size)\n"
            "    yv = normalize(y.values, NORM) + k * OFFSET\n"
            "    lines.append((x, yv))\n"
            "    ax.plot(x, yv, marker='o', ms=2.5, lw=1.2, label=c['label'])\n"
            f"ax.set_xlabel({axis_title(curves[0].x_name, curves[0].x_unit)!r})\n"
            f"ax.set_ylabel({curves[0].y_name!r})\n"
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
