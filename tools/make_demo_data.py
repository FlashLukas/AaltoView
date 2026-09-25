"""make_demo_data.py -- FAKE but physically sensible TR-MOKE / FMR measurements.

For trying the viewer without lab data, and for the README screenshots (which
must not depend on anyone's data folder). The files have exactly the layout the
AaltoFlow scan engine writes -- a complex lock-in signal as a `_real`/`_imag` pair,
named coordinates with units, `dims` outer -> inner, autosave names
`<YYYY-MM-DD>/<HHMMSS>_<name>.nc` -- so the viewer cannot tell them apart.

    python tools/make_demo_data.py [folder]        # default: demo_data/

The physics is a thin in-plane magnetised film (permalloy-like):

    Kittel         f0(B)   = g * sqrt(B (B + Ms))            g = 28 GHz/T, Ms = 1 T
    PSSW (n = 1)   f1(B)   = g * sqrt((B + Bex)(B + Bex + Ms))    Bex = 150 mT, weaker
    linewidth      df      = df0 + 2 alpha f0                 alpha = 0.008
    response       chi(f)  = (df/2) / (f0 - f - i df/2)       |chi| = 1 on resonance,
                                                              the phase turns by 180 deg
    propagation    exp(-d / L) * exp(i k d)                   L = 5 um, k from lambda

plus a detection phase offset and complex Gaussian noise. Numbers are chosen to
look like a lab measurement, not to fit any real sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

G = 28.0          # GHz per T
MS = 1000.0       # mT
BEX = 150.0       # mT, first perpendicular standing spin wave
ALPHA = 0.008
DF0 = 0.18        # GHz, inhomogeneous linewidth
PHASE0 = 0.6      # rad, detection phase
RNG = np.random.default_rng(20260916)


def kittel(b_mT, extra=0.0):
    b = np.asarray(b_mT, dtype=float) + extra
    return G * np.sqrt(np.clip(b * (b + MS), 0, None)) / 1000.0


def chi(f, f0, amp=1.0):
    df = DF0 + 2 * ALPHA * f0
    return amp * (df / 2) / (f0 - f - 1j * df / 2)


def fmr(field, freq, amp_uV=18.0):
    """Lock-in signal (uV) on a field x frequency grid: Kittel + a weak PSSW."""
    B, F = np.meshgrid(field, freq, indexing="ij")
    f0, f1 = kittel(B), kittel(B, BEX)
    # the drive efficiency of a stripline falls with frequency
    z = chi(F, f0, amp_uV * (4.0 / np.maximum(F, 4.0)) ** 0.5)
    z += chi(F, f1, 0.22 * amp_uV)
    return z * np.exp(1j * PHASE0)


def noise(shape, sigma):
    return sigma * (RNG.normal(size=shape) + 1j * RNG.normal(size=shape)) / np.sqrt(2)


def complex_vars(name, dims, z, units):
    attrs = {"units": units}
    return {f"{name}_real": (dims, z.real, {**attrs, "complex_pair": name, "complex_part": "real"}),
            f"{name}_imag": (dims, z.imag, {**attrs, "complex_pair": name, "complex_part": "imag"})}


def coord(name, values, units):
    return (name, np.asarray(values, dtype=float), {"units": units})


def recipe(name, comment, axes, detectors):
    return json.dumps({"name": name, "comment": comment, "fixed": {}, "axes": axes,
                       "detectors": detectors})


def lin(param, a, b, n):
    return {"type": "linear", "param": param, "start": a, "stop": b, "num": n}


def save(ds: xr.Dataset, path: Path, seconds: float):
    dims = ds.attrs.pop("_dims")
    ds.attrs.update(dims=",".join(dims), n_points=int(np.prod([ds.sizes[d] for d in dims])),
                    seconds=seconds, created="demo")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf")
    print(f"  {path}  {dict(ds.sizes)}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else Path("demo_data")

    # 1. the classic map: field x frequency at the antenna edge
    field = np.linspace(0, 200, 101)
    freq = np.linspace(2, 16, 141)
    z = fmr(field, freq) + noise((101, 141), 0.35)
    refl = 1.212 + 0.004 * np.sin(np.linspace(0, 3, 101))[:, None] + RNG.normal(0, 6e-4, (101, 141))
    save(xr.Dataset(
        {**complex_vars("lockin", ("field", "rf_freq"), z, "uV"),
         "reflectivity": (("field", "rf_freq"), refl, {"units": "V"})},
        coords={"field": coord("field", field, "mT"), "rf_freq": coord("rf_freq", freq, "GHz")},
        attrs={"name": "fmr_field_freq", "_dims": ["field", "rf_freq"],
               "comment": "Py 20 nm, CPW 10 um, spot at the signal line edge",
               "recipe_json": recipe("fmr_field_freq", "", [lin("field", 0, 200, 101),
                                     lin("rf_freq", 2, 16, 141)], ["lockin", "reflectivity"])}),
        out / "2026-09-14" / "153012_fmr_field_freq.nc", 5210.0)

    # 2. the same map at several distances from the antenna: a 3-D cube
    dist = np.array([0.0, 2.0, 5.0, 10.0])
    field3 = np.linspace(0, 200, 81)
    freq3 = np.linspace(2, 16, 113)
    base = fmr(field3, freq3)
    lam = 3.0 + 0.9 * freq3 / 10.0                  # um; longer waves at higher f here
    cube = np.stack([base * np.exp(-d / 5.0) * np.exp(1j * 2 * np.pi * d / lam)[None, :]
                     for d in dist])
    cube = cube + noise(cube.shape, 0.35)
    save(xr.Dataset(
        complex_vars("lockin", ("distance", "field", "rf_freq"), cube, "uV"),
        coords={"distance": coord("distance", dist, "um"), "field": coord("field", field3, "mT"),
                "rf_freq": coord("rf_freq", freq3, "GHz")},
        attrs={"name": "fmr_distance_cube", "_dims": ["distance", "field", "rf_freq"],
               "comment": "spot moved away from the antenna: 0, 2, 5, 10 um",
               "recipe_json": recipe("fmr_distance_cube", "", [
                   {"type": "array", "param": "distance", "values": dist.tolist()},
                   lin("field", 0, 200, 81), lin("rf_freq", 2, 16, 113)], ["lockin"])}),
        out / "2026-09-15" / "101530_fmr_distance_cube.nc", 14870.0)

    # 3. spin waves leaving the antenna, imaged at 8 GHz for three fields
    fields = np.array([60.0, 75.0, 90.0])
    x = np.linspace(0, 24, 97)
    y = np.linspace(-12, 12, 81)
    X, Y = np.meshgrid(x, y)
    img = []
    for b in fields:
        f0 = kittel(b)
        drive = chi(8.0, f0, 20.0)                  # how resonant 8 GHz is at this field
        lam = 2.2 + 0.06 * (b - 60.0)               # um; the wavelength moves with field
        wave = np.exp(-X / 7.0) * np.exp(1j * 2 * np.pi * X / lam) * np.exp(-(Y / 7.5) ** 2)
        img.append(drive * wave * np.exp(1j * PHASE0))
    img = np.stack(img) + noise((3, 81, 97), 0.25)
    save(xr.Dataset(
        complex_vars("lockin", ("field", "pos_y", "pos_x"), img, "uV"),
        coords={"field": coord("field", fields, "mT"), "pos_y": coord("pos_y", y, "um"),
                "pos_x": coord("pos_x", x, "um")},
        attrs={"name": "spinwave_image_8GHz", "_dims": ["field", "pos_y", "pos_x"],
               "comment": "8 GHz, XY raster next to the antenna (x = 0)",
               "recipe_json": recipe("spinwave_image_8GHz", "", [
                   {"type": "array", "param": "field", "values": fields.tolist()},
                   {"type": "raster", "x": lin("pos_x", 0, 24, 97), "y": lin("pos_y", -12, 12, 81)}],
                   ["lockin"])}),
        out / "2026-09-16" / "094500_spinwave_image_8GHz.nc", 9320.0)

    # 4. field sweeps at a few frequencies -- the "1D" measurement
    fr = np.array([6.0, 8.0, 10.0, 12.0])
    fb = np.linspace(0, 200, 401)
    zz = np.stack([fmr(fb, [f])[:, 0] for f in fr]) + noise((4, 401), 0.3)
    save(xr.Dataset(
        complex_vars("lockin", ("rf_freq", "field"), zz, "uV"),
        coords={"rf_freq": coord("rf_freq", fr, "GHz"), "field": coord("field", fb, "mT")},
        attrs={"name": "field_sweeps", "_dims": ["rf_freq", "field"],
               "comment": "fine field sweeps at 6, 8, 10, 12 GHz",
               "recipe_json": recipe("field_sweeps", "", [
                   {"type": "array", "param": "rf_freq", "values": fr.tolist()},
                   lin("field", 0, 200, 401)], ["lockin"])}),
        out / "2026-09-16" / "131205_field_sweeps.nc", 1604.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
