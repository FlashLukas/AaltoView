"""view.py -- turning an N-dimensional scan into the 2-D picture you asked for.

A scan is a cube: frequency x X x Y, or field x freq x X x Y, or more. A screen
shows two dimensions. Everything between those two facts is this module: pick
which dims are the image axes, and say what should happen to each of the others
-- hold it at one value, or average over it.

No Qt here on purpose. The choices are DATA (`Slice` objects), the reduction is
a function, and both are testable without a screen. `apps/viewer.py` is only
the set of widgets that produces the choices and draws the answer.

    sel = {"freq": Slice("mean")}                 # average over every frequency
    da  = reduce_cube(ds["kerr"], x="x", y="y", slices=sel)
    # -> a (y, x) DataArray ready to hand to an image

Two decisions that matter physically
------------------------------------
* **NaN-aware.** A scan that was aborted, or one still running, is a cube with
  holes. `numpy.mean` over a hole gives NaN for the whole average, so one
  missing frequency would blank an entire map. Every reduction here skips NaN
  and reports how much real data went in (`Reduced.coverage`).
* **Complex is reduced BEFORE the magnitude is taken** (coherent averaging).
  mean(|z|) and |mean(z)| are different numbers: the first keeps noise, the
  second cancels it -- and also cancels the signal if the phase wanders. With a
  stable phase (a lock-in on a locked reference) coherent is the one you want,
  and it is what you get by reducing first. `Reduced.note` says so on screen, so
  the choice is never silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .data import as_complex, complex_names

#: How a dimension that is neither of the two image axes is disposed of.
#: "at"   : hold it at index i0 (one slice of the cube)
#: "mean" : average over indices i0..i1 inclusive (the whole axis if i1 is None)
MODES = ("at", "mean")

#: Ways to turn a complex detector into something with a colour.
PARTS = ("abs", "arg", "real", "imag")


@dataclass
class Slice:
    """What to do with one non-image dimension."""
    mode: str = "at"
    i0: int = 0
    i1: int | None = None          # inclusive; None = to the end of the axis

    def span(self, n: int) -> tuple[int, int]:
        """The index range this selects in an axis of length n, clamped."""
        i0 = max(0, min(int(self.i0), n - 1))
        if self.mode == "at":
            return i0, i0
        i1 = n - 1 if self.i1 is None else int(self.i1)
        i1 = max(0, min(i1, n - 1))
        return (i0, i1) if i0 <= i1 else (i1, i0)


@dataclass
class Reduced:
    """The 1-D or 2-D result, plus what it took to get there."""
    data: xr.DataArray
    x: str
    y: str | None
    label: str
    coverage: float                # fraction of contributing cells that were real
    averaged: int                  # how many cube cells went into each pixel
    note: str = ""


# ─────────────────────────── what can be displayed ────────────────────────────

def detector_names(ds: xr.Dataset) -> list[str]:
    """Every quantity the viewer can show, complex ones first.

    A complex detector is stored as a `_real`/`_imag` PAIR (see data.py). The
    pair is hidden and the four derived views are offered instead: an FMR map is
    read as |S21|, and nobody goes looking for a resonance in the real part.
    """
    pairs = complex_names(ds)
    hidden = {f"{p}_{half}" for p in pairs for half in ("real", "imag")}
    return list(pairs) + [n for n in ds.data_vars if n not in hidden]


def is_complex(ds: xr.Dataset, name: str) -> bool:
    return name in complex_names(ds) or (
        name in ds.data_vars and np.iscomplexobj(ds[name].values))


def detector(ds: xr.Dataset, name: str) -> xr.DataArray:
    """The DataArray behind a name, complex recombined if it was split."""
    if name in complex_names(ds):
        return as_complex(ds, name)
    return ds[name]


def apply_part(da: xr.DataArray, part: str) -> xr.DataArray:
    """Complex -> real, the way the operator asked. A real array is returned as is."""
    if not np.iscomplexobj(da.values):
        return da
    if part == "arg":
        out = xr.DataArray(np.angle(da.values), dims=da.dims, coords=da.coords)
        out.attrs = {**da.attrs, "units": "rad"}
        return out
    fn = {"abs": np.abs, "real": np.real, "imag": np.imag}[part]
    out = xr.DataArray(fn(da.values), dims=da.dims, coords=da.coords)
    out.attrs = dict(da.attrs)
    return out


# ───────────────────────────── the reduction ──────────────────────────────────

def reduce_cube(da: xr.DataArray, x: str, y: str | None,
                slices: dict[str, Slice] | None = None,
                part: str = "abs") -> Reduced:
    """Collapse `da` onto the axes (y, x), disposing of every other dim.

    x, y   : dimension names for the image; y=None gives a 1-D line along x.
    slices : what to do with each remaining dim (default: hold at index 0).
    part   : which piece of a complex detector to end up with. Applied LAST, so
             an average over a complex axis is coherent -- see the module note.
    """
    slices = dict(slices or {})
    keep = [d for d in (y, x) if d]
    if x not in da.dims:
        raise KeyError(f"'{x}' is not a dimension of this data ({list(da.dims)})")
    if y is not None and y not in da.dims:
        raise KeyError(f"'{y}' is not a dimension of this data ({list(da.dims)})")
    if y is not None and y == x:
        raise ValueError("the two image axes must be different dimensions")

    others = [d for d in da.dims if d not in keep]
    # Count the real (non-NaN) cells BEFORE reducing, so "62 % of the average is
    # actually data" can be shown. On a finished scan this is 1.0 and the label
    # stays quiet; on a half-finished one it is the difference between a map and
    # a rumour.
    sub = da
    averaged = 1
    reduced_dims = []
    for d in others:
        n = sub.sizes[d]
        i0, i1 = slices.get(d, Slice()).span(n)
        sub = sub.isel({d: slice(i0, i1 + 1)})
        if i1 > i0:
            reduced_dims.append(d)
            averaged *= (i1 - i0 + 1)

    finite = np.isfinite(sub.values if not np.iscomplexobj(sub.values)
                         else np.abs(sub.values))
    coverage = float(finite.mean()) if finite.size else 0.0

    if reduced_dims:
        # skipna keeps a hole in the cube from swallowing a whole pixel.
        sub = sub.mean(dim=reduced_dims, skipna=True)
    # drop the length-1 dims left by the "at" selections. Note: NOT squeeze() --
    # that would also flatten an image axis that happens to have one point, and
    # a 1-point y axis is a legal (if thin) scan.
    for d in others:
        if d in sub.dims:
            sub = sub.isel({d: 0})

    out = apply_part(sub, part)
    order = [d for d in (y, x) if d]
    out = out.transpose(*order)

    note = ""
    if averaged > 1 and np.iscomplexobj(sub.values) and part in ("abs", "arg"):
        note = "coherent average (complex averaged, then " + part + ")"
    return Reduced(data=out, x=x, y=y, label=str(da.name or ""),
                   coverage=coverage, averaged=averaged, note=note)


def default_axes(da: xr.DataArray) -> tuple[str, str | None]:
    """A sensible first picture: the two INNERMOST dims, x innermost.

    That is the pair the engine varies fastest, so on a partly finished scan it
    is the part that already has data in it -- which is what you want to see
    while the run is going.
    """
    dims = list(da.dims)
    if not dims:
        return "", None
    if len(dims) == 1:
        return dims[0], None
    return dims[-1], dims[-2]


def coord_text(ds, dim: str, i: int) -> str:
    """'3.712 GHz' rather than 'index 7' -- the operator thinks in the quantity."""
    try:
        val = float(np.asarray(ds[dim].values)[i])
    except Exception:
        return f"#{i}"
    unit = ds[dim].attrs.get("units", "") if dim in ds.coords else ""
    return f"{val:g} {unit}".strip()
