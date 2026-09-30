"""results.py -- what leaves the module: the FFT map, the peak table, the
dispersion results, and their figures. No Qt.

The same conventions as AaltoView's own exports: the FFT map is an
aaltoview.export.Map (write_map: matrix or XYZ; Origin: a matrix + colour
map), tables have the three header rows Long Name / Units / Comments, which
Origin's import recognises.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aaltoview.export import MapStyle, Map, Selection, axis_title, map_levels
from aaltoview.export import _delimiter, _encoding

from . import fft as F
from . import inputs as I
from . import waveguide as W

SHOW = ("abs", "power", "log")
SHOW_TEXT = {"abs": "|FFT|", "power": "|FFT|²", "log": "log10 |FFT|"}


@dataclass
class Column:
    name: str
    unit: str
    comment: str
    values: list = field(default_factory=list)
    kind: str = "Y"                   # Origin: X, Y, E (error of the Y before), L


def shown(sp: F.Spectrum, show: str) -> np.ndarray:
    m = sp.magnitude
    if show == "power":
        return m ** 2
    if show == "log":
        with np.errstate(divide="ignore"):
            out = np.log10(m)
        out[~np.isfinite(out)] = np.nan
        return out
    return m


def k_title(k_unit: str) -> tuple[str, str]:
    return ("k", "rad/µm") if k_unit == "rad/um" else ("1/λ", "1/µm")


def fft_map(inp: I.Input, sp: F.Spectrum, show: str = "abs") -> Map:
    """The spectra as an AaltoView Map: k along x, the input's y along y."""
    z = shown(sp, show)
    kn, ku = k_title(sp.k_unit)
    part = sp.settings.part
    det = inp.detector or inp.z_name
    what = {"complex": det, "real": f"Re {det}", "imag": f"Im {det}",
            "abs": f"|{det}|"}[part]
    name = SHOW_TEXT[show].replace("FFT", f"FFT {what}")
    unit = "" if show == "log" else (inp.z_unit if show == "abs" else
                                    (f"{inp.z_unit}²" if inp.z_unit else ""))
    order = np.argsort(inp.y)
    return Map(z=z[order], x=sp.k, y=np.asarray(inp.y)[order], x_name=kn, x_unit=ku,
               y_name=inp.y_name, y_unit=inp.y_unit, z_name=name, z_unit=unit,
               levels=map_levels(z, MapStyle()),
               selection=Selection(detector=inp.z_name, x=inp.x_name, y=inp.y_name),
               source=inp.source)


def peak_columns(inp: I.Input, sp: F.Spectrum, peaks: list[F.Peak],
                 points: list[W.Point] | None = None) -> list[Column]:
    """One row per peak: the line's y, k (signed), |k|, lambda, amplitude, width,
    and -- when the points are given -- the f and B the dispersion uses."""
    if not peaks:
        raise ValueError("no peaks -- press Find peaks")
    kn, ku = k_title(sp.k_unit)
    two_pi = 2 * np.pi if sp.k_unit == "rad/um" else 1.0
    cols = [Column(inp.y_name, inp.y_unit, "the line", [float(inp.y[p.line]) for p in peaks],
                   kind="X"),
            Column(kn, ku, "peak position (signed: + = towards +x on complex data)",
                   [p.k for p in peaks]),
            Column(f"|{kn}|", ku, "peak position", [abs(p.k) for p in peaks]),
            Column("wavelength", "µm", "2 pi / |k|",
                   [two_pi / abs(p.k) if p.k else np.nan for p in peaks]),
            Column("amplitude", inp.z_unit, "|FFT| at the peak (a pure wave: its amplitude)",
                   [p.amplitude for p in peaks]),
            Column("fwhm", ku, "peak width in k (the window's, for a long wave)",
                   [p.fwhm for p in peaks]),
            Column("snr", "", "amplitude / the line's noise (median |FFT| over the allowed k)",
                   [p.snr for p in peaks])]
    if points is not None:
        cols += [Column("f", "GHz", "frequency used in the dispersion", [p.f for p in points]),
                 Column("B", "mT", "field used in the dispersion", [p.B for p in points])]
    return cols


def result_columns(res: W.Result) -> list[Column]:
    rows = res.rows()
    return [Column("parameter", "", "", [r[0] for r in rows], kind="L"),
            Column("value", "", "", [float(r[1]) for r in rows]),
            Column("error", "", "1 sigma", [np.nan if r[2] is None else float(r[2])
                                             for r in rows], kind="E"),
            Column("unit", "", "", [r[3] for r in rows], kind="L"),
            Column("meaning", "", "", [r[4] for r in rows], kind="L")]


def write_columns(path, columns: list[Column]) -> Path:
    path = Path(path)
    n = max((len(c.values) for c in columns), default=0)

    def cell(v):
        if isinstance(v, str):
            return v
        return "" if not np.isfinite(v) else repr(float(v))

    with open(path, "w", newline="", encoding=_encoding(path)) as f:
        w = csv.writer(f, delimiter=_delimiter(path))
        w.writerow([c.name for c in columns])
        w.writerow([c.unit for c in columns])
        w.writerow([c.comment for c in columns])
        for i in range(n):
            w.writerow([cell(c.values[i]) if i < len(c.values) else "" for c in columns])
    return path


def model_curve(res: W.Result, k_max: float, n: int = 400, B=None):
    """(k, f) of the fitted dispersion, both signs of k, rad/um and GHz."""
    k = np.linspace(-k_max, k_max, n)
    return k, W.frequency(k, res.values, res.pinning, None if B is None else np.full(n, B))


def figure_fft(m: Map, peaks_xy=None, model_xy=None, title: str = "", size=(6.4, 4.8)):
    from aaltoview.export import figure_map
    fig = figure_map(m, title=title, size=size)
    ax = fig.axes[0]
    if peaks_xy is not None and len(peaks_xy[0]):
        ax.plot(peaks_xy[0], peaks_xy[1], "o", ms=3, mfc="none", mec="#1f77b4", mew=1.0,
                label="peaks")
    if model_xy is not None:
        ax.plot(model_xy[0], model_xy[1], "-", color="#2ca02c", lw=1.2, label="dispersion fit")
    if peaks_xy is not None or model_xy is not None:
        ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(m.x[0], m.x[-1])
    return fig


def figure_dispersion(points: list[W.Point], res: W.Result | None, k_unit="rad/um",
                      size=(6.4, 4.8)):
    from aaltoview.export import _figure
    fig = _figure(size)
    ax = fig.add_subplot()
    used = [p for p in points if p.use]
    off = [p for p in points if not p.use]
    ax.plot([p.k for p in used], [p.f for p in used], "o", ms=4, color="#1f77b4",
            label="peaks")
    if off:
        ax.plot([p.k for p in off], [p.f for p in off], "x", ms=4, color="#7f7f7f",
                label="left out")
    if res is not None and points:
        kmax = max(abs(p.k) for p in points) * 1.15
        Bs = [p.B for p in used if np.isfinite(p.B)]
        k, f = model_curve(res, kmax, B=Bs[0] if Bs else None)
        ax.plot(k, f, "-", color="#2ca02c", lw=1.4, label=f"Kalinikos-Slavin ({res.pinning})")
    ax.set_xlabel(axis_title(*k_title(k_unit)))
    ax.set_ylabel(axis_title("f", "GHz"))
    ax.legend(fontsize=8)
    return fig
