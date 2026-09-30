"""waveguide.py -- the spin-wave dispersion of a magnetic stripe, and its fit.

The model: Kalinikos & Slavin, J. Phys. C 19, 7013 (1986), lowest thickness
mode (uniform across the film, unpinned surfaces), film magnetised IN-PLANE,
in the form used throughout the magnonics literature:

    f = gamma/2pi * sqrt( (B + Ms lam k^2) (B + Ms lam k^2 + Ms F) )
    F = 1 - P cos^2(phi) + Ms P (1 - P) sin^2(phi) / (B + Ms lam k^2)
    P = 1 - (1 - exp(-k d)) / (k d)
    lam = 2 A / (mu0 Ms^2)                 exchange length squared

  B = mu0 H, Ms = mu0 Ms (tesla), d the film thickness, phi the angle between
  the wavevector and the magnetisation. k -> 0 gives Kittel, sqrt(B (B + Ms));
  phi = 90 deg the Damon-Eshbach (surface) branch, phi = 0 the backward-volume
  branch.

The stripe: a waveguide of width w along x (the scan direction). Across it the
mode is a standing wave with k_y = n pi / w_eff (n = 1 the fundamental width
mode, n = 0 an infinite film). The edges are not free: the dynamic dipolar
field pins the magnetisation partly ("effective dipolar pinning", Guslienko,
Demokritov, Hillebrands & Slavin, PRB 66, 132402 (2002)), which acts as a wider
stripe:

    w_eff = w * D / (D - 2),   D(p) = 2 pi / (p (1 + 2 ln(1/p))),   p = d / w

(w_eff -> w for a wide thin stripe; 5 % wider at d = 30 nm, w = 2 um). With
"unpinned" the geometric width is used instead; with "none" there is no width
mode at all (k_y = 0: an infinite film, or a stripe so wide it does not
matter) -- w and n are then not used, and not fitted.

The total wavevector is k^2 = kx^2 + ky^2, kx the measured one (its sign --
which way the wave runs -- does not change f here). With the field at an angle
theta to the stripe's axis, M = (cos theta, sin theta) in-plane (saturated), and

    cos^2(phi) = (kx^2 cos^2 theta + ky^2 sin^2 theta) / k^2

-- the standing wave across the stripe is +ky and -ky together, whose cross
terms cancel. theta = 90 deg (field across the stripe) is the Damon-Eshbach
geometry of most waveguide experiments, theta = 0 backward volume.

Units at the interface: k in rad/um, f in GHz, B and Ms in mT, A in pJ/m, d in
nm, w in um, gamma/2pi in GHz/T, theta in deg. No Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MU0 = 4e-7 * np.pi

#: name -> (unit, meaning, default, fitted by default, lower bound, upper bound)
PARAMS = {
    "gamma": ("GHz/T", "gamma / 2 pi", 28.0, False, 10.0, 60.0),
    "Ms":    ("mT", "mu0 Ms (mu0 M_eff with anisotropy)", 1000.0, True, 1.0, 3000.0),
    "A":     ("pJ/m", "exchange stiffness", 13.0, False, 0.01, 100.0),
    "d":     ("nm", "film thickness", 30.0, False, 0.1, 1e5),
    "w":     ("um", "stripe width (0 = infinite film)", 2.0, False, 0.0, 1e4),
    "n":     ("", "width mode (0 = none, 1 = fundamental)", 1.0, False, 0.0, 20.0),
    "B":     ("mT", "mu0 H, when the points do not carry it", 50.0, False, 0.0, 1e4),
    "theta": ("deg", "field angle to the stripe axis (90 = Damon-Eshbach)", 90.0, False,
              -180.0, 180.0),
}
PINNING = ("guslienko", "unpinned", "none")
#: the parameters that only exist through the width mode
WIDTH_PARAMS = ("w", "n")


def guslienko_width(w_um: float, d_nm: float) -> float:
    """The effective width of a stripe with dipolar-pinned edges (um)."""
    if w_um <= 0:
        return 0.0
    p = (d_nm * 1e-3) / w_um
    if p <= 0:
        return float(w_um)
    if p >= 1:
        raise ValueError("thickness >= width: not a thin stripe")
    D = 2 * np.pi / (p * (1 + 2 * np.log(1 / p)))
    if D <= 2:
        raise ValueError("the stripe is too thick for the Guslienko pinning (D <= 2)")
    return float(w_um * D / (D - 2))


def width_wavevector(v: dict, pinning: str = "guslienko") -> float:
    """k_y (rad/um) of width mode n."""
    n, w = float(v["n"]), float(v["w"])
    if pinning == "none" or n <= 0 or w <= 0:
        return 0.0
    weff = guslienko_width(w, v["d"]) if pinning == "guslienko" else w
    return n * np.pi / weff


def frequency(kx, v: dict, pinning: str = "guslienko", B=None) -> np.ndarray:
    """f (GHz) at measured wavevectors kx (rad/um), parameters v (PARAMS units).
    B: per-point field in mT (an array like kx), else v["B"]."""
    kx = np.abs(np.asarray(kx, dtype=float)) * 1e6                 # rad/m
    ky = width_wavevector(v, pinning) * 1e6
    Bt = (np.full_like(kx, v["B"]) if B is None else np.asarray(B, dtype=float)) * 1e-3
    Bt = np.abs(Bt)
    Ms = v["Ms"] * 1e-3
    d = v["d"] * 1e-9
    lam = 2 * v["A"] * 1e-12 * MU0 / Ms ** 2
    k2 = kx ** 2 + ky ** 2
    k = np.sqrt(k2)
    kd = k * d
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(kd > 1e-9, 1 - (1 - np.exp(-kd)) / kd, kd / 2)
        th = np.radians(v["theta"])
        cos2 = np.where(k2 > 0, (kx ** 2 * np.cos(th) ** 2 + ky ** 2 * np.sin(th) ** 2) / k2,
                        np.cos(th) ** 2)
    Bk = Bt + Ms * lam * k2
    with np.errstate(divide="ignore", invalid="ignore"):
        F = 1 - P * cos2 + Ms * P * (1 - P) * (1 - cos2) / Bk
        f2 = Bk * (Bk + Ms * F)
    return v["gamma"] * np.sqrt(np.clip(f2, 0, None))


def group_velocity(kx, v: dict, pinning: str = "guslienko", B=None) -> np.ndarray:
    """df/dk * 2 pi, in km/s (GHz um / (rad/um) = km/s), numerically."""
    kx = np.asarray(kx, dtype=float)
    h = 1e-4 * max(float(np.nanmax(np.abs(kx))) if kx.size else 1.0, 1.0)
    return 2 * np.pi * (frequency(kx + h, v, pinning, B) - frequency(kx - h, v, pinning, B)) / (2 * h)


# ─────────────────────────────────── the fit ──────────────────────────────────

@dataclass
class Point:
    """One peak of the spectra: where, and at what frequency and field."""
    k: float                          # rad/um, signed as found
    f: float                          # GHz
    B: float = float("nan")           # mT; NaN = use the parameter B
    use: bool = True
    line: int = -1
    label: str = ""
    amplitude: float = float("nan")


@dataclass
class Spec:
    value: float
    vary: bool = False
    min: float = -np.inf
    max: float = np.inf


def default_specs() -> dict[str, Spec]:
    return {n: Spec(p[2], p[3], p[4], p[5]) for n, p in PARAMS.items()}


@dataclass
class Result:
    values: dict[str, float]
    errors: dict[str, float | None]
    vary: dict[str, bool]
    pinning: str
    rms: float                        # GHz, over the points used
    n_points: int
    success: bool
    message: str
    derived: dict[str, float] = field(default_factory=dict)

    def rows(self) -> list[tuple]:
        """(name, value, error, unit, meaning) for the results table."""
        out = []
        for n, p in PARAMS.items():
            if self.pinning == "none" and n in WIDTH_PARAMS:
                continue                  # no width mode: w and n play no part
            out.append((n, self.values[n], self.errors.get(n), p[0],
                        p[1] + ("" if self.vary.get(n) else "  (fixed)")))
        for n, (v, unit, meaning) in self.derived.items():
            out.append((n, v, None, unit, meaning))
        out.append(("rms", self.rms, None, "GHz", "rms of f(model) - f(peak)"))
        return out


def _derived(v: dict, pinning: str) -> dict:
    out = {}
    try:
        lam = 2 * v["A"] * 1e-12 * MU0 / (v["Ms"] * 1e-3) ** 2
        out["l_ex"] = (np.sqrt(lam) * 1e9, "nm", "exchange length sqrt(2A / mu0 Ms^2)")
        if pinning != "none" and v["w"] > 0 and v["n"] > 0:
            weff = guslienko_width(v["w"], v["d"]) if pinning == "guslienko" else v["w"]
            out["w_eff"] = (weff, "um", f"effective width ({pinning})")
            out["k_y"] = (width_wavevector(v, pinning), "rad/um", "n pi / w_eff")
        out["f(k=0)"] = (float(frequency(np.array([0.0]), v, pinning)[0]), "GHz",
                         "the bottom of the band, at the parameter B")
    except Exception:
        pass
    return out


def fit(points: list[Point], specs: dict[str, Spec], pinning: str = "guslienko",
        robust: bool = False) -> Result:
    """Least squares in f: model f(|k|, B) - measured f, over the points used."""
    from scipy.optimize import least_squares
    pts = [p for p in points if p.use and np.isfinite(p.k) and np.isfinite(p.f)]
    names = [n for n in PARAMS if specs[n].vary
             and not (pinning == "none" and n in WIDTH_PARAMS)]
    if not pts:
        raise ValueError("no points to fit -- find the peaks first")
    if len(pts) <= len(names):
        raise ValueError(f"{len(pts)} points for {len(names)} free parameters: fix some")
    k = np.array([p.k for p in pts])
    f = np.array([p.f for p in pts])
    Bp = np.array([p.B for p in pts])
    if "B" in names and np.isfinite(Bp).all():
        raise ValueError("every point carries its own field: B cannot be fitted")
    base = {n: float(specs[n].value) for n in PARAMS}

    def values(p):
        v = dict(base)
        v.update(zip(names, p))
        return v

    def field_of(v):
        return np.where(np.isfinite(Bp), Bp, v["B"])

    def resid(p):
        v = values(p)
        return frequency(k, v, pinning, field_of(v)) - f

    if names:
        p0 = np.array([np.clip(base[n], specs[n].min, specs[n].max) for n in names])
        lo = np.array([specs[n].min for n in names])
        hi = np.array([specs[n].max for n in names])
        out = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac",
                            loss="soft_l1" if robust else "linear", f_scale=0.05)
        v = values(out.x)
        r = resid(out.x)
        dof = max(len(pts) - len(names), 1)
        s2 = float(np.sum(r ** 2) / dof)
        try:
            cov = np.linalg.pinv(out.jac.T @ out.jac) * s2
            err = {n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)}
        except np.linalg.LinAlgError:
            err = {n: None for n in names}
        success, message = bool(out.success), str(out.message)
    else:
        v, r = base, resid(np.array([]))
        err, success, message = {}, True, "nothing free: the model at the typed values"
    return Result(values=v, errors={n: err.get(n) for n in PARAMS},
                  vary={n: n in names for n in PARAMS}, pinning=pinning,
                  rms=float(np.sqrt(np.mean(r ** 2))), n_points=len(pts),
                  success=success, message=message, derived=_derived(v, pinning))
