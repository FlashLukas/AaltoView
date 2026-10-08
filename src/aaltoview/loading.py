"""loading.py -- "load with a script": a correction applied as a file is read.

Some measurements need fixing before anything looks at them -- a TR-MOKE file
whose complex lock-in signal is conjugated on half of its frequencies by the
laser's aliasing, a detector with a known offset, an old file with its axes in
the wrong order. Doing it ONCE, at loading, means the map, the 1D plots, every
export, the notebook and the analysis modules all see the corrected data.

A loading script is a .py file in LoadingScripts/ (next to AnalysisModules/;
more folders through AALTOVIEW_LOADING_SCRIPTS, os.pathsep-separated). Drop
one in, and the viewer's "load with" list offers it. The whole contract:

    \"\"\"One line: what it does (the list's tooltip).\"\"\"
    NAME = "TR-MOKE unfold (80 MHz laser)"       # optional; else the file name

    def load(ds, path):                           # xarray.Dataset, pathlib.Path
        ...
        return ds                                 # the corrected dataset

It gets the file as AaltoFlow wrote it (complex detectors still as their
_real / _imag pair: see data.py) and returns a Dataset of the same layout.
Files starting with "_" are not listed (helpers a script may share).

No Qt here.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from . import home
from .data import complex_names

SCRIPTS_ENV = "AALTOVIEW_LOADING_SCRIPTS"
SCRIPTS_DIR = "LoadingScripts"
#: the attribute a loaded dataset carries: which script made it
ATTR = "aaltoview_loading_script"


@dataclass
class Script:
    name: str
    description: str
    path: Path


def script_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get(SCRIPTS_ENV, "").split(os.pathsep) if p]
    dirs.append(home.repo() / SCRIPTS_DIR)   # the checkout's, also from scan-core
    return [d for d in dirs if d.is_dir()]


def _read(path: Path):
    spec = importlib.util.spec_from_file_location(f"aaltoview_loading_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "load", None)):
        raise ValueError(f"{path.name} has no load(ds, path) function")
    return mod


def available() -> list[Script]:
    """Every script that can be read, by name. A broken one is left out (it
    must not take the list down) -- run() on it says what is wrong."""
    out, seen = [], set()
    for d in script_dirs():
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_") or p.stem in seen:
                continue
            try:
                mod = _read(p)
            except Exception:
                continue
            seen.add(p.stem)
            doc = (mod.__doc__ or "").strip().splitlines()
            out.append(Script(name=str(getattr(mod, "NAME", p.stem)),
                              description=doc[0] if doc else "", path=p))
    return sorted(out, key=lambda s: s.name.lower())


def find(name: str) -> Script | None:
    return next((s for s in available() if s.name == name), None)


def run(script: Script | str | Path, ds: xr.Dataset, path=None) -> xr.Dataset:
    """The dataset as the script corrects it, marked with the script's name."""
    sp = script.path if isinstance(script, Script) else Path(script)
    mod = _read(sp)
    out = mod.load(ds.copy(deep=True), Path(path) if path else None)
    if not isinstance(out, xr.Dataset):
        raise TypeError(f"{sp.name}: load() returned {type(out).__name__}, not a Dataset")
    name = script.name if isinstance(script, Script) else str(getattr(mod, "NAME", sp.stem))
    out.attrs[ATTR] = name
    return out


# ──────────────────────── helpers for the scripts ─────────────────────────────

#: frequency units -> GHz
_FREQ = {"Hz": 1e-9, "kHz": 1e-6, "MHz": 1e-3, "GHz": 1.0}


def frequency_dim(ds: xr.Dataset) -> str | None:
    """The dim whose coordinate is a frequency (by its unit): rf_freq, ..."""
    for d in ds.dims:
        if d in ds.coords and str(ds[d].attrs.get("units", "")).strip() in _FREQ:
            return d
    return None


def alias(f_ghz, f_rep_mhz: float) -> np.ndarray:
    """What a laser pulsing at f_rep samples a precession at f down to (MHz):
    f - n f_rep, n the nearest harmonic, in (-f_rep/2, +f_rep/2], to 1 Hz."""
    f = np.asarray(f_ghz, dtype=float) * 1e3
    d = np.round(np.mod(f + f_rep_mhz / 2, f_rep_mhz) - f_rep_mhz / 2, 6)
    return np.where(d == -f_rep_mhz / 2, f_rep_mhz / 2, d)


def tr_moke_unfold(ds: xr.Dataset, f_rep_mhz: float, invert: bool = False,
                   freq_dim: str | None = None) -> xr.Dataset:
    """Undo the conjugation of stroboscopic detection on every complex detector.

    The laser samples the precession at f_rep, so the lock-in sees it at the
    alias f - n f_rep -- and cannot tell a negative alias from a positive one:
    there it records the COMPLEX CONJUGATE (a wave running the other way; in a
    spatial FFT the branch jumps between +k and -k every f_rep / 2). The
    imaginary part is negated on those frequencies. `invert`: the other half
    (the lock-in's sign convention). Frequencies on a harmonic or half-way
    (alias 0 or f_rep / 2) carry no direction and are left alone. Real
    detectors (a power meter, a DC signal) are not touched."""
    dim = freq_dim or frequency_dim(ds)
    if dim is None:
        raise ValueError("no frequency axis in this file (a coordinate in Hz / kHz / MHz / "
                         "GHz): nothing to unfold")
    f = np.asarray(ds[dim].values, dtype=float) * _FREQ[str(ds[dim].attrs["units"]).strip()]
    d = alias(f, f_rep_mhz)
    flip = (d > 0) if invert else (d < 0)
    flip &= np.abs(d) < f_rep_mhz / 2
    sign = xr.DataArray(np.where(flip, -1.0, 1.0), dims=[dim], coords={dim: ds[dim]})
    out = ds.copy()
    names = complex_names(ds)
    for name in names:
        im = f"{name}_imag"
        if dim in ds[im].dims:
            out[im] = (ds[im] * sign).assign_attrs(ds[im].attrs)
    for name, var in ds.data_vars.items():          # natively complex, if any
        if np.iscomplexobj(var.values) and dim in var.dims:
            out[name] = xr.where(sign < 0, np.conj(var), var).assign_attrs(var.attrs)
    out.attrs["tr_moke_unfold"] = (f"{f_rep_mhz:g} MHz{', other half' if invert else ''}: "
                                   f"{int(flip.sum())} of {f.size} {dim} values conjugated")
    return out
