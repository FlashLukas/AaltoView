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
    linewidth      df      = alpha g (2 (B + Bex) + Ms) + dB0 * df0/dB   (FWHM in f)
                             alpha = 0.008, dB0 = 1.5 mT: a field sweep then shows
                             FWHM = dB0 + 2 alpha f / g, the textbook form
    response       chi(f)  = (df/2) / (f0 - f - i df/2)       |chi| = 1 on resonance,
                                                              the phase turns by 180 deg
    propagation    exp(-d / L) * exp(i k d)                   L = 5 um, k from lambda

plus a detection phase offset and complex Gaussian noise. Numbers are chosen to
look like a lab measurement, not to fit any real sample.

File 7 is a VNA as it really is: S21 frequency sweeps of the isotropic film
at several fields, with the exact damped-oscillator lineshape
chi = f0 df / (f0^2 - f^2 - i f df), a 3.2 ns cable delay, a standing-wave
ripple and a loss sloping with frequency -- VNA_DELAY / VNA_RIPPLE below --
plus a sweep at 500 mT, where nothing resonates in the band: the reference.

A second, ANISOTROPIC film for the angle-dependent files (5, 6): Meff = 1.4 T,
in-plane uniaxial Bu = 12 mT along 90 deg, 4-fold B4 = 18 mT along 0 deg, 6-fold
B6 = 3 mT along 15 deg, alpha = 0.004, dB0 = 1 mT -- field in the plane at angle
phi_H. The resonance comes from the textbook in-plane formula with the
equilibrium angle found by brute force (inplane() below), deliberately NOT from
the FMR fit module's code, so that module can be tested against it.
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
DB0 = 1.5         # mT, inhomogeneous linewidth (FWHM in field)
PHASE0 = 0.6      # rad, detection phase
#: the VNA of file 7
VNA_DELAY = 3.2         # ns
VNA_RIPPLE = (0.02, 3.3)  # relative amplitude, period in GHz
VNA_DIP = 0.03          # the resonance, relative to the transmitted signal


def vna_background(f):
    """What the cables and the stripline do to S21, without the sample."""
    amp, period = VNA_RIPPLE
    return (0.8 * (1 - 0.012 * f) * (1 + amp * np.sin(2 * np.pi * f / period + 0.4))
            * np.exp(-2j * np.pi * VNA_DELAY * f))


def oscillator(f, f0, df):
    """The damped-oscillator susceptibility, exact in f; = +i on resonance."""
    return f0 * df / (f0 * f0 - f * f - 1j * f * df)


#: the anisotropic film of the angle-dependent files
ANISO = dict(Meff=1400.0, Bu=12.0, phi_u=90.0, B4=18.0, phi_4=0.0, B6=3.0, phi_6=15.0,
             alpha=0.004, dB0=1.0)
SEED = 20260916
RNG = np.random.default_rng(SEED)


def kittel(b_mT, extra=0.0):
    b = np.asarray(b_mT, dtype=float) + extra
    return G * np.sqrt(np.clip(b * (b + MS), 0, None)) / 1000.0


def linewidth(b_mT, extra=0.0):
    """FWHM in frequency (GHz) of the isotropic film: Gilbert + inhomogeneous."""
    b = np.maximum(np.asarray(b_mT, dtype=float) + extra, 1e-3)
    # df/dB diverges at zero field (the Kittel curve is vertical there), which
    # made a ghost line 21 GHz wide at 0 mT; below 10 mT it is held
    bs = np.maximum(b, 10.0)
    dfdb = G / 1000 * (2 * bs + MS) / (2 * np.sqrt(bs * (bs + MS)))
    return ALPHA * G / 1000 * (2 * b + MS) + DB0 * dfdb


def chi(f, f0, amp, df):
    return amp * (df / 2) / (f0 - f - 1j * df / 2)


def fmr(field, freq, amp_uV=18.0):
    """Lock-in signal (uV) on a field x frequency grid: Kittel + a weak PSSW."""
    B, F = np.meshgrid(field, freq, indexing="ij")
    f0, f1 = kittel(B), kittel(B, BEX)
    # the drive efficiency of a stripline falls with frequency
    z = chi(F, f0, amp_uV * (4.0 / np.maximum(F, 4.0)) ** 0.5, linewidth(B))
    z += chi(F, f1, 0.22 * amp_uV, linewidth(B, BEX))
    return z * np.exp(1j * PHASE0)


def inplane(b_mT, phi_h_deg, a=ANISO):
    """The anisotropic film, field in the plane: f0 (GHz) and the frequency FWHM
    (GHz) for every field in b_mT. Textbook in-plane Kittel formula; the angle
    of M by brute force on a 0.1 deg grid (the lowest minimum), refined by a
    parabola through the three lowest grid points."""
    b = np.asarray(b_mT, dtype=float)[:, None]
    step = 0.1
    grid = np.deg2rad(np.arange(-180.0, 180.0, step))

    def energy(phi):
        dh = phi - np.deg2rad(phi_h_deg)
        du, d4, d6 = (phi - np.deg2rad(a[k]) for k in ("phi_u", "phi_4", "phi_6"))
        return (-b * np.cos(dh) - a["Bu"] / 2 * np.cos(du) ** 2
                - a["B4"] / 16 * (3 + np.cos(4 * d4)) - a["B6"] / 36 * np.cos(6 * d6))

    E = energy(grid[None, :])
    n = grid.size
    i = np.argmin(E, axis=1)
    r = np.arange(b.shape[0])
    em, e0, ep = E[r, (i - 1) % n], E[r, i], E[r, (i + 1) % n]
    curv = em - 2 * e0 + ep
    shift = np.where(curv > 0, 0.5 * (em - ep) / np.where(curv > 0, curv, 1.0), 0.0)
    phi = grid[i] + shift * np.deg2rad(step)
    dh = phi - np.deg2rad(phi_h_deg)
    du, d4, d6 = (phi - np.deg2rad(a[k]) for k in ("phi_u", "phi_4", "phi_6"))
    b = b[:, 0]
    e_pp = (b * np.cos(dh) + a["Bu"] * np.cos(2 * du) + a["B4"] * np.cos(4 * d4)
            + a["B6"] * np.cos(6 * d6))
    e_tt = (b * np.cos(dh) + a["Meff"] + a["Bu"] * np.cos(du) ** 2
            + a["B4"] / 4 * (3 + np.cos(4 * d4)) + a["B6"] / 6 * np.cos(6 * d6))

    f0 = G / 1000 * np.sqrt(np.clip(e_pp * e_tt, 0, None))
    return f0, a["alpha"] * G / 1000 * (e_pp + e_tt)


def inplane_width(b_mT, phi_h_deg, a=ANISO):
    """f0 and the full frequency FWHM, with dB0 * df0/dB from a small step in B."""
    f0, gilbert = inplane(b_mT, phi_h_deg, a)
    fp, _ = inplane(np.asarray(b_mT) + 0.05, phi_h_deg, a)
    fm, _ = inplane(np.asarray(b_mT) - 0.05, phi_h_deg, a)
    return f0, gilbert + a["dB0"] * np.abs(fp - fm) / 0.1


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
    # reseeded on EVERY run: the files must not depend on how often this was
    # called before in the same process (the tests call it several times, and
    # the noise differed with their order -- found 2026-09-29)
    global RNG
    RNG = np.random.default_rng(SEED)
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
        drive = chi(8.0, f0, 20.0, linewidth(b))    # how resonant 8 GHz is at this field
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

    # 5. the anisotropic film: field sweeps at 10 GHz, field rotated in the plane.
    #    rf_freq is a dimension of length 1: the file says at which frequency.
    phis = np.arange(0.0, 360.0, 10.0)
    fa = np.linspace(0, 200, 401)
    za = np.empty((1, phis.size, fa.size), dtype=complex)
    for j, ph in enumerate(phis):
        f0, df = inplane_width(fa, ph)
        za[0, j] = chi(10.0, f0, 15.0, df) * np.exp(1j * PHASE0)
    za += noise(za.shape, 0.3)
    save(xr.Dataset(
        complex_vars("s21", ("rf_freq", "phi_H", "field"), za, "uV"),
        coords={"rf_freq": coord("rf_freq", [10.0], "GHz"),
                "phi_H": coord("phi_H", phis, "deg"), "field": coord("field", fa, "mT")},
        attrs={"name": "angle_field_sweeps", "_dims": ["rf_freq", "phi_H", "field"],
               "comment": "anisotropic film: field sweeps at 10 GHz, in-plane angle 0-350 deg",
               "recipe_json": recipe("angle_field_sweeps", "", [
                   {"type": "array", "param": "phi_H", "values": phis.tolist()},
                   lin("field", 0, 200, 401)], ["s21"])}),
        out / "2026-09-17" / "101500_angle_field_sweeps.nc", 7400.0)

    # 6. the same film: frequency sweeps at 40 mT, field rotated in the plane
    ff = np.linspace(2, 16, 281)
    zf = np.empty((1, phis.size, ff.size), dtype=complex)
    for j, ph in enumerate(phis):
        f0, df = inplane_width(np.array([40.0]), ph)
        zf[0, j] = chi(ff, f0[0], 15.0, df[0]) * np.exp(1j * PHASE0)
    zf += noise(zf.shape, 0.3)
    save(xr.Dataset(
        complex_vars("s21", ("field", "phi_H", "rf_freq"), zf, "uV"),
        coords={"field": coord("field", [40.0], "mT"),
                "phi_H": coord("phi_H", phis, "deg"), "rf_freq": coord("rf_freq", ff, "GHz")},
        attrs={"name": "angle_freq_sweeps", "_dims": ["field", "phi_H", "rf_freq"],
               "comment": "anisotropic film: VNA frequency sweeps at 40 mT, in-plane angle",
               "recipe_json": recipe("angle_freq_sweeps", "", [
                   {"type": "array", "param": "phi_H", "values": phis.tolist()},
                   lin("rf_freq", 2, 16, 281)], ["s21"])}),
        out / "2026-09-17" / "143000_angle_freq_sweeps.nc", 3900.0)

    # 7. VNA frequency sweeps with everything a real VNA adds; 500 mT = reference
    fields7 = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 500.0])
    f7 = np.linspace(1.0, 20.0, 1901)
    z7 = np.empty((fields7.size, f7.size), dtype=complex)
    for j, b in enumerate(fields7):
        f0 = kittel(b)
        z7[j] = vna_background(f7) * (1 + VNA_DIP * np.exp(1j * PHASE0)
                                      * oscillator(f7, f0, linewidth(b)))
    z7 += noise(z7.shape, 0.0015)
    save(xr.Dataset(
        complex_vars("s21", ("field", "rf_freq"), z7, ""),
        coords={"field": coord("field", fields7, "mT"), "rf_freq": coord("rf_freq", f7, "GHz")},
        attrs={"name": "vna_freq_sweeps", "_dims": ["field", "rf_freq"],
               "comment": "VNA S21: cable delay, ripple; 500 mT = reference (nothing in band)",
               "recipe_json": recipe("vna_freq_sweeps", "", [
                   {"type": "array", "param": "field", "values": fields7.tolist()},
                   lin("rf_freq", 1, 20, 1901)], ["s21"])}),
        out / "2026-09-18" / "110000_vna_freq_sweeps.nc", 2100.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
