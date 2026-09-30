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

    def to_dict(self) -> dict:
        d = {"detector": self.detector, "x": self.x, "y": self.y, "part": self.part,
             "slices": {d: {"mode": s.mode, "i0": int(s.i0),
                            "i1": None if s.i1 is None else int(s.i1)}
                        for d, s in self.slices.items()}}
        if self.ref:
            d["ref"] = dict(self.ref)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        return cls(detector=d["detector"], x=d["x"], y=d.get("y"),
                   part=d.get("part", "abs"),
                   slices={k: Slice(v["mode"], v["i0"], v.get("i1"))
                           for k, v in d.get("slices", {}).items()},
                   ref=dict(d["ref"]) if d.get("ref") else None)


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
    cplx = np.iscomplexobj(da.values)
    y_name = quantity_name(sel, cplx)
    y_unit = "rad" if (sel.part == "arg" and cplx) else units_of(ds, sel.detector)
    if sel.ref:
        zl = ref_line(ds, sel)
        y = np.asarray(apply_part(xr.DataArray(zl), sel.part).values, dtype=float).copy()
        style, msel = _ref_style(sel)
        y_name, y_unit = _ref_name_unit(ds, msel, style, y_name, y_unit)
    else:
        red = reduce_cube(da, sel.x, None, sel.slices, sel.part)
        y = np.asarray(red.data.values, dtype=float).copy()
    x = coords_of(ds, sel.x, y.size).copy()
    if label is None:
        label = describe_slices(ds, sel) or sel.detector
        if sel.ref:
            label += f" ({reference_text(ds, msel, style)})"
    z = None
    if cplx:
        z = (zl.astype(complex).copy() if sel.ref else
             np.asarray(reduce_cube(da, sel.x, None, sel.slices, "complex").data.values,
                        dtype=complex).copy())
    return Curve(x=x, y=y, label=label, x_name=sel.x, x_unit=units_of(ds, sel.x),
                 y_name=y_name, y_unit=y_unit,
                 selection=sel, source=str(source) if source else None,
                 z=z, held=held_values(ds, sel))


def _ref_style(sel: Selection) -> tuple["MapStyle", Selection]:
    """The map a referenced cut came from: its style (reference only) and axes."""
    r = sel.ref
    return (MapStyle(ref=r["mode"], ref_op=r.get("op", "divide"), ref_i=int(r.get("i", -1))),
            Selection(sel.detector, x=r["x"], y=r["y"], slices=sel.slices, part=sel.part))


def ref_line(ds: xr.Dataset, sel: Selection) -> np.ndarray:
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
    red = reduce_cube(da, msel.x, msel.y, slices, "complex")
    ny, nx = red.data.shape
    zc = reference_values(red.data.values, style, coords_of(ds, msel.x, nx),
                          coords_of(ds, msel.y, ny))
    k = sel.slices.get(held, Slice()).span(da.sizes[held])[0]
    return np.asarray(zc[k] if held == msel.y else zc[:, k])


def _ref_name_unit(ds, sel: Selection, style: "MapStyle", name: str, unit: str):
    """'|s21| (÷ field = 800 mT)', and the unit a reference leaves."""
    name = f"{name} ({reference_text(ds, sel, style)})"
    if style.ref in ("dd_y", "dd_x") and sel.part != "arg":
        dim = sel.y if style.ref == "dd_y" else sel.x
        unit = f"1/{units_of(ds, dim)}" if units_of(ds, dim) else ""
    elif style.ref_op == "divide" and sel.part != "arg":
        unit = ""
    return name, unit


def held_values(ds: xr.Dataset, sel: Selection) -> dict[str, tuple[float, str]]:
    """{dim: (value, unit)} for every dim held at ONE index -- a dim of length 1
    too: "this sweep was at 10 GHz" is what an analysis needs, even when the
    label leaves it out as obvious."""
    da = detector(ds, sel.detector)
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
    cplx = np.iscomplexobj(da.values)
    name = quantity_name(sel, cplx)
    unit = "rad" if (sel.part == "arg" and cplx) else units_of(ds, sel.detector)
    if style.ref == "none":
        red = reduce_cube(da, sel.x, sel.y, sel.slices, sel.part)
        z = np.asarray(red.data.values, dtype=float)
    else:
        # reference BEFORE |z| / arg: the ratio of two complex numbers
        red = reduce_cube(da, sel.x, sel.y, sel.slices, "complex")
        ny, nx = red.data.shape
        zc = reference_values(red.data.values, style, coords_of(ds, sel.x, nx),
                              coords_of(ds, sel.y, ny))
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
    m = Map(z=z, x=coords_of(ds, sel.x, nx), y=coords_of(ds, sel.y, ny),
            x_name=sel.x, x_unit=units_of(ds, sel.x),
            y_name=sel.y, y_unit=units_of(ds, sel.y),
            z_name=name, z_unit=unit, levels=map_levels(z, style),
            selection=sel, source=str(source) if source else None)
    return m, red


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
        n = detector(ds, sel.detector).sizes[dim]
        return f"{op} {dim} = {coord_text(ds, dim, _ref_index(style.ref_i, n))}"
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
    return "" if not np.isfinite(v) else repr(float(v))


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
    for c, y in zip(shown, ys):
        if not shared:
            cols.append(c.x); names.append(c.x_name); units.append(c.x_unit); notes.append(c.label)
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
    for c, y in zip(shown, displayed_y(curves, norm, offset)):
        ax.plot(c.x, y, marker="o", ms=2.5, lw=1.2, label=c.label)
    if shown:
        xt, yt = curve_axis_titles(shown, norm)       # a unit only if all share it
        ax.set_xlabel(xt)
        ax.set_ylabel(yt)
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
    if np.iscomplexobj(da.values) and part != "complex":
        da = xr.DataArray(part_of(da.values, part), dims=da.dims, coords=da.coords)
    return da.transpose(*[d for d in (y, x) if d])


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
    m = reduce(detector(ds, c["detector"]), r["x"], r["y"], slices, "complex")
    xs = ds[r["x"]].values if r["x"] in ds.coords else np.arange(m.shape[1])
    ys = ds[r["y"]].values if r["y"] in ds.coords else np.arange(m.shape[0])
    z = reference(m.values, r["mode"], r.get("op", "divide"), r.get("i", -1), xs, ys)
    n = ds.sizes[held]
    k = min(max(int(c["slices"].get(held, {"i0": 0})["i0"]), 0), n - 1)
    return part_of(z[k] if held == r["y"] else z[:, k], c["part"])


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
            "m = reduce(detector(ds, sel['detector']), sel['x'], sel['y'], sel['slices'], 'complex')\n"
            "xs = ds[sel['x']].values if sel['x'] in ds.coords else np.arange(m.shape[1])\n"
            "ys = ds[sel['y']].values if sel['y'] in ds.coords else np.arange(m.shape[0])\n"
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
            "        y = ref_line(ds, c)\n"
            "    else:\n"
            "        y = reduce(detector(ds, c['detector']), c['x'], None, c['slices'], c['part']).values\n"
            "    x = ds[c['x']].values if c['x'] in ds.coords else np.arange(y.size)\n"
            "    yv = normalize(y, NORM) + k * OFFSET\n"
            "    lines.append((x, yv))\n"
            "    ax.plot(x, yv, marker='o', ms=2.5, lw=1.2, label=c['label'])\n"
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
