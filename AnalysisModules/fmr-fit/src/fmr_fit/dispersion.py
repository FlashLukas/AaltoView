"""dispersion.py -- from fitted resonances to material parameters. No Qt.

The Resonances tab gives, per curve and peak, WHERE the resonance is (x0) and
how wide (FWHM), plus the coordinates the curve was held at. Here those points
are fitted with one magnetic model:

    energy density / Ms, in field units (mT), m = unit magnetisation,
    theta from the film NORMAL (z), phi in the plane from x:

    E = - mu0H (m . h)                                   Zeeman
        + (Meff / 2) m_z^2                               shape + perpendicular
        - (Bu / 4) [ s^2 + s^2 cos 2(phi - phi_u) ]      in-plane uniaxial
        - (B4 / 16)[ 3 s^4 + s^4 cos 4(phi - phi_4) ]    4-fold
        - (B6 / 36)  s^6 cos 6(phi - phi_6)              6-fold        (s = sin theta)

    mu0 Meff = mu0 Ms - 2 K_perp / Ms       (positive: in-plane easy)
    Bn = the amplitude of the n-fold term in the in-plane STIFFNESS: with the
    field in-plane the Kittel formula reads
        f = (gamma/2pi) sqrt( (H cos(phi-phi_H) + Bu cos 2dphi_u + B4 cos 4dphi_4 + B6 cos 6dphi_6)
                             x (H cos(phi-phi_H) + Meff + Bu cos^2 dphi_u + B4 (3 + cos 4dphi_4)/4
                                + B6 cos 6dphi_6 / 6) )
    i.e. Bu = 2 Ku / Ms, B4 = 2 K4 / Ms, B6 = 2 K6 / Ms in the usual thin-film
    notation (Farle, Rep. Prog. Phys. 61, 755 (1998)). phi_n = an easy axis
    when Bn > 0.

For every point: the EQUILIBRIUM direction of m (numerically, the lowest of
several local minima -- no hysteresis), then the resonance from the curvature
of E around it (Smit-Beljers, in a local frame so the poles are not special):

    f = (gamma/2pi) sqrt( (E_aa + Hex)(E_bb + Hex) - E_ab^2 )
    FWHM in frequency = alpha (gamma/2pi) (E_aa + E_bb + 2 Hex) + dH0 * df/dH

Hex = 0 for the uniform (Kittel) mode; for a perpendicular standing spin wave
of order n, Hex = 2 A (n pi / d)^2 / Ms (unpinned surfaces), fitted per mode or
through A. Field-swept linewidth = FWHM_f / (df/dH): for an in-plane film that
is the familiar dH = dH0 + 2 alpha f / (gamma/2pi) -- the general formula also
covers out-of-plane angles, where the magnetisation lags behind the field.

Why one general model instead of the textbook formulas: field sweeps at an
angle, angle sweeps at a field and frequency sweeps are then all the same
fit, and the in-plane / perpendicular textbook cases are tests of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Spec

MU0 = 4e-7 * np.pi
GAMMA_E = 13.996245            # GHz/T per unit g: mu_B / h
ROLE_UNIFORM, ROLE_IGNORE = 0, -1          # PSSW of order n: role = n


# ─────────────────────────────────── units ────────────────────────────────────

FIELD_UNITS = {"mt": 1.0, "t": 1e3, "ut": 1e-3, "g": 0.1, "gs": 0.1, "gauss": 0.1,
               "oe": 0.1, "koe": 100.0, "a/m": MU0 * 1e3, "ka/m": MU0 * 1e6}
FREQ_UNITS = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}
ANGLE_UNITS = {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "°": 1.0, "rad": 180 / np.pi,
               "mrad": 0.18 / np.pi}
#: what each coordinate is measured in here
INTERNAL = {"H": "mT", "f": "GHz", "phi": "deg", "theta": "deg"}


def unit_kind(unit: str) -> tuple[str | None, float]:
    """'mT' -> ('field', 1.0); 'MHz' -> ('freq', 1e-3); 'rad' -> ('angle', 57.3).
    The factor converts to mT / GHz / deg. Unknown -> (None, 1.0)."""
    u = (unit or "").strip().lower().replace("µ", "u").replace("μ", "u")
    for kind, table in (("field", FIELD_UNITS), ("freq", FREQ_UNITS), ("angle", ANGLE_UNITS)):
        if u in table:
            return kind, table[u]
    return None, 1.0


# ─────────────────────────────────── points ───────────────────────────────────

@dataclass
class Point:
    """One resonance: where it was found, and in which sweep."""
    H: float                     # mT
    f: float                     # GHz
    theta: float                 # deg, field from the film normal (90 = in-plane)
    phi: float                   # deg, field in the plane
    swept: str                   # which of H / f / phi / theta the resonance x0 is
    err: float = np.nan          # 1 sigma of x0, in the swept coordinate's unit
    fwhm: float = np.nan         # in the swept coordinate's unit
    fwhm_err: float = np.nan
    role: int = ROLE_UNIFORM     # 0 uniform, n >= 1 PSSW n, -1 ignore
    label: str = ""
    use: bool = True
    key: tuple = ()              # (row, peak) it came from, for the operator's choices


#: what a dimension of the measurement can be, for the model
COORDS = {"H": "field  μ0H", "f": "frequency  f", "phi": "in-plane angle  φ_H",
          "theta": "polar angle  θ_H (from the normal)",
          "elev": "elevation (from the plane) = 90° − θ_H"}
#: where a coordinate comes from: ("const", value in mT/GHz/deg), or
#: ("dim", name, fallback): that dimension of each curve's file -- and, for a
#: file WITHOUT that dimension (a field sweep at one fixed angle next to an
#: angle series), the fallback value; None = such a curve cannot be placed
Source = tuple
#: the fallback when a file lacks the dimension: the field in the plane, along x
FALLBACK = {"phi": 0.0, "theta": 90.0, "elev": 0.0, "H": None, "f": None}


def guess_sources(rows) -> dict[str, Source]:
    """Which dimension is which, from the units (T/Oe/A/m = field, Hz = frequency,
    deg/rad = angle; an angle named theta/polar/oop is polar, elev = elevation,
    anything else in-plane). What no dimension supplies is a constant: the
    field in-plane (theta 90 deg, phi 0) unless a dimension says otherwise."""
    dims: dict[str, str] = {}
    for c, _ in rows:
        dims.setdefault(c.x_name, c.x_unit)
        for d, (_, u) in c.held.items():
            dims.setdefault(d, u)
    src: dict[str, Source] = {}
    for d, u in dims.items():
        kind, _ = unit_kind(u)
        name = d.lower()
        role = {"field": "H", "freq": "f"}.get(kind)
        if kind == "angle":
            role = ("theta" if any(k in name for k in ("theta", "polar", "oop"))
                    else "elev" if "elev" in name else "phi")
        if role and role not in src:
            src[role] = ("dim", d, FALLBACK[role])
    if "theta" in src or "elev" in src:
        src.setdefault("phi", ("const", 0.0))
    else:
        src.setdefault("theta", ("const", 90.0))
        src.setdefault("phi", ("const", 0.0))
    src.setdefault("H", ("const", 0.0))
    src.setdefault("f", ("const", 10.0))
    return src


def default_roles(res) -> dict[int, int]:
    """Peak k -> role: the strongest peak is the uniform mode, the rest are
    ignored until the operator flags them (PSSW n, ...)."""
    amps = {k: abs(res.values[f"p{k}_amp"]) for k in range(1, res.setup.n_peaks + 1)}
    top = max(amps, key=amps.get)
    return {k: (ROLE_UNIFORM if k == top else ROLE_IGNORE) for k in amps}


def points_from_fits(rows, sources: dict[str, Source],
                     roles: dict[tuple[int, int], int] | None = None,
                     notes: list[str] | None = None) -> tuple[list[Point], list[str]]:
    """Fitted curves -> dispersion points, one per (curve, peak).

    rows: (Curve, model.Result) pairs. The curve's x axis is the SWEPT
    coordinate: x0 is its resonance, the FWHM its width. Everything else comes
    from the held coordinates, the fallbacks (a file without that dimension)
    or the constants. roles: {(row index, peak): role}.
    Returns the points and, for anything that could not be placed, why;
    `notes` (if given) collects where a fallback value was used.
    """
    points, problems = [], []
    fell_back: dict[str, int] = {}
    by_dim = {s[1]: role for role, s in sources.items() if s[0] == "dim"}
    for i, (c, res) in enumerate(rows):
        swept_role = by_dim.get(c.x_name)
        kind, fac = unit_kind(c.x_unit)
        if swept_role is None or kind is None:
            problems.append(f"{c.label}: its x axis '{c.x_name}' ({c.x_unit or 'no unit'}) "
                            "is not set as field, frequency or angle")
            continue
        base = {}
        missing = []
        for role, src in sources.items():
            how, what = src[0], src[1]
            fallback = src[2] if len(src) > 2 else None
            if role == swept_role:
                continue
            if how == "const":
                base[role] = float(what)
            elif what in c.held:
                v, u = c.held[what]
                base[role] = v * unit_kind(u)[1]
            elif fallback is not None:
                base[role] = float(fallback)
                key = f"{what} = {fallback:g} {INTERNAL.get(role, 'deg')}"
                fell_back[key] = fell_back.get(key, 0) + 1
            else:
                missing.append(what)
        if missing:
            problems.append(f"{c.label}: no value for {', '.join(missing)} (its file has no "
                            "such dimension; give a value in Coordinates)")
            continue
        rl = default_roles(res)
        for k in range(1, res.setup.n_peaks + 1):
            x0 = res.values[f"p{k}_center"] * fac
            e = res.errors.get(f"p{k}_center")
            w, we = res.fwhm(k)
            coord = dict(base)
            coord[swept_role] = x0
            swept = swept_role
            if "elev" in coord:
                coord["theta"] = 90.0 - coord.pop("elev")
                swept = "theta" if swept == "elev" else swept
            p = Point(H=coord.get("H", 0.0), f=coord.get("f", 0.0),
                      theta=coord.get("theta", 90.0), phi=coord.get("phi", 0.0),
                      swept=swept, err=np.nan if e is None else e * fac,
                      fwhm=w * fac, fwhm_err=np.nan if we is None else we * fac,
                      role=(roles or {}).get((i, k), rl[k]),
                      label=c.label + (f" · peak {k}" if res.setup.n_peaks > 1 else ""),
                      key=(i, k))
            points.append(p)
    if notes is not None:
        notes += [f"{what} for {n} curve{'s' if n != 1 else ''} whose file has no such "
                  "dimension" for what, n in fell_back.items()]
    return points, problems


def arrays(points: list[Point]) -> dict[str, np.ndarray]:
    return {k: np.array([getattr(p, k) for p in points], dtype=float)
            for k in ("H", "f", "theta", "phi")}


# ───────────────────────────── the energy and its minimum ─────────────────────

def _direction(theta_deg, phi_deg) -> np.ndarray:
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)


def _terms(v: dict) -> dict[int, tuple[float, float]]:
    """{n: (Bn cos n phi_n, Bn sin n phi_n)} for n = 2 (uniaxial), 4, 6, whichever
    form the parameters are in (Cartesian while fitting, polar when typed)."""
    out = {}
    for n, key in ((2, "u"), (4, "4"), (6, "6")):
        if f"B{key}_c" in v:
            out[n] = (v[f"B{key}_c"], v[f"B{key}_s"])
        else:
            b, ph = v.get(f"B{key}", 0.0), np.deg2rad(v.get(f"phi_{key}", 0.0))
            out[n] = (b * np.cos(n * ph), b * np.sin(n * ph))
    return out


def energy(m: np.ndarray, hdir: np.ndarray, H: np.ndarray, v: dict) -> np.ndarray:
    """E / Ms in mT for magnetisation directions m (..., 3). See the module note."""
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
    w = mx + 1j * my                        # sin(theta) e^{i phi}
    s2 = mx * mx + my * my
    w2 = w * w
    E = -H * np.sum(m * hdir, axis=-1) + 0.5 * v["Meff"] * mz * mz
    t = _terms(v)
    c, s = t[2]
    E = E - 0.25 * (np.hypot(c, s) * s2 + c * w2.real + s * w2.imag)
    c, s = t[4]
    w4 = w2 * w2
    E = E - (3 * np.hypot(c, s) * s2 * s2 + c * w4.real + s * w4.imag) / 16
    c, s = t[6]
    w6 = w4 * w2
    E = E - (c * w6.real + s * w6.imag) / 36
    return E


def _frame(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to m and to each other."""
    a = np.zeros_like(m)
    near_pole = np.abs(m[..., 2]) > 0.9
    a[..., 2] = np.where(near_pole, 0.0, 1.0)
    a[..., 0] = np.where(near_pole, 1.0, 0.0)
    e1 = a - np.sum(a * m, axis=-1, keepdims=True) * m
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(m, e1)


_H = 1e-3          # rad: finite-difference step on the sphere. Not smaller: the
#                    roundoff of E / h^2 would make the model noisy at 1e-8, as small
#                    as the fit's own parameter steps (the truncation error is smooth)


def _local(m, hdir, H, v):
    """Energy, gradient (2) and Hessian (2x2) of E around m in its local frame.
    At an equilibrium the Hessian's determinant does not depend on the frame."""
    e1, e2 = _frame(m)

    def E(a, b):
        mm = m + a * e1 + b * e2
        mm = mm / np.linalg.norm(mm, axis=-1, keepdims=True)
        return energy(mm, hdir, H, v)

    h = _H
    e0 = E(0, 0)
    ea, eA = E(h, 0), E(-h, 0)
    eb, eB = E(0, h), E(0, -h)
    g = np.stack([(ea - eA) / (2 * h), (eb - eB) / (2 * h)], axis=-1)
    haa = (ea - 2 * e0 + eA) / h ** 2
    hbb = (eb - 2 * e0 + eB) / h ** 2
    hab = (E(h, h) - E(h, -h) - E(-h, h) + E(-h, -h)) / (4 * h * h)
    return e0, g, haa, hbb, hab, e1, e2


_STARTS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                    [.7071, .7071, 0], [-.7071, .7071, 0], [.7071, -.7071, 0],
                    [-.7071, -.7071, 0]], dtype=float)


def equilibrium(H, hdir, v, start: np.ndarray | None = None) -> np.ndarray:
    """Direction of m at the lowest energy minimum, for every point (N, 3).

    Newton steps on the sphere from several starts (the field direction plus
    the axes and in-plane diagonals), made safe near a saddle by shifting the
    Hessian; the start that ends lowest wins. `start` (N, 3) = a warm start
    per point instead (a small step in H, angle or the parameters): that one
    and the field direction only.
    """
    H = np.asarray(H, dtype=float)
    hdir = np.asarray(hdir, dtype=float)
    n = H.size
    if start is not None:
        cands = np.stack([start, hdir + 1e-3 * np.array([0.31, 0.47, 0.23])])
    else:
        cands = np.concatenate([hdir[None], np.broadcast_to(_STARTS[:, None, :],
                                                            (len(_STARTS), n, 3))])
        # nudge off exact symmetry points, where the gradient is zero by symmetry
        cands = cands + 1e-3 * np.array([0.31, 0.47, 0.23])
    S = cands.shape[0]
    m = (cands / np.linalg.norm(cands, axis=-1, keepdims=True)).reshape(S * n, 3)
    HH = np.tile(H, S)
    hd = np.tile(hdir, (S, 1))
    scale = np.abs(HH) + abs(v["Meff"]) + 1.0
    act = np.arange(S * n)                 # only the ones not converged yet
    for _ in range(100):
        e0, g, haa, hbb, hab, e1, e2 = _local(m[act], hd[act], HH[act], v)
        moving = np.hypot(g[:, 0], g[:, 1]) >= 1e-9 * scale[act]
        if not moving.any():
            break
        act = act[moving]
        g, haa, hbb, hab = g[moving], haa[moving], hbb[moving], hab[moving]
        e1, e2 = e1[moving], e2[moving]
        sc = scale[act]
        tr, det = haa + hbb, haa * hbb - hab * hab
        emin = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        lam = np.where(emin > 1e-6 * sc, 0.0, 1e-6 * sc - 1.5 * emin)
        a, b, d = haa + lam, hab, hbb + lam
        dd = a * d - b * b
        sa = -(d * g[:, 0] - b * g[:, 1]) / dd
        sb = -(-b * g[:, 0] + a * g[:, 1]) / dd
        step = np.hypot(sa, sb)
        k = np.minimum(1.0, 0.3 / np.maximum(step, 1e-300))
        mm = m[act] + (k * sa)[:, None] * e1 + (k * sb)[:, None] * e2
        m[act] = mm / np.linalg.norm(mm, axis=-1, keepdims=True)
    E = energy(m, hd, HH, v).reshape(S, n)
    best = np.argmin(E, axis=0)
    return m.reshape(S, n, 3)[best, np.arange(n)]


def modes(H, theta, phi, v, hex_=0.0, start=None):
    """Resonance frequency (GHz), frequency FWHM per unit alpha (GHz), and m."""
    H = np.atleast_1d(np.asarray(H, dtype=float))
    hdir = _direction(np.broadcast_to(theta, H.shape), np.broadcast_to(phi, H.shape))
    m = equilibrium(H, hdir, v, start)
    _, _, haa, hbb, hab, _, _ = _local(m, hdir, H, v)
    hx = np.broadcast_to(np.asarray(hex_, dtype=float), H.shape)
    det = (haa + hx) * (hbb + hx) - hab * hab
    k = v["gamma"] * 1e-3                    # GHz per mT
    f = k * np.sqrt(np.maximum(det, 0.0))
    return f, k * (haa + hbb + 2 * hx), m


def frequency(H, theta, phi, v, hex_=0.0) -> np.ndarray:
    return modes(H, theta, phi, v, hex_)[0]


def derivative(coord: str, H, theta, phi, v, hex_=0.0, m=None) -> np.ndarray:
    """d f / d coord (GHz per mT or per deg), central difference, warm-started."""
    step = {"H": 0.05, "theta": 0.02, "phi": 0.02}[coord]
    args = {"H": np.atleast_1d(np.asarray(H, float)), "theta": np.asarray(theta, float),
            "phi": np.asarray(phi, float)}
    hi, lo = dict(args), dict(args)
    hi[coord] = args[coord] + step
    lo[coord] = args[coord] - step
    fh = modes(hi["H"], hi["theta"], hi["phi"], v, hex_, start=m)[0]
    fl = modes(lo["H"], lo["theta"], lo["phi"], v, hex_, start=m)[0]
    return (fh - fl) / (2 * step)


def resonance_field(f, theta, phi, v, hex_=0.0, n_grid: int = 61) -> np.ndarray:
    """mu0 H (mT) at which the mode sits at frequency f -- for drawing field-swept
    dispersions and making test data. The LOWEST field where f(H) crosses f:
    bracketed on a coarse grid, then bisected (Newton overshoots to negative
    fields where the resonance is near zero field, e.g. 4 GHz with anisotropy).
    NaN where there is no crossing below the grid's end."""
    f = np.atleast_1d(np.asarray(f, float))
    theta = np.broadcast_to(np.asarray(theta, float), f.shape)
    phi = np.broadcast_to(np.asarray(phi, float), f.shape)
    hx = np.broadcast_to(np.asarray(hex_, float), f.shape)
    M = abs(v["Meff"])
    b = f / (v["gamma"] * 1e-3)
    kittel = np.maximum(-M + np.sqrt(M * M + 4 * b * b), 0) / 2
    anis = sum(abs(v.get(k, 0.0)) for k in ("Bu", "B4", "B6", "Bu_c", "Bu_s", "B4_c",
                                            "B4_s", "B6_c", "B6_s"))
    # perpendicular fields need up to Meff more than the in-plane estimate
    hmax = 1.5 * (kittel + b + M * np.abs(np.cos(np.deg2rad(theta)))) + 2 * anis + 50
    s = np.linspace(0.0, 1.0, n_grid)
    grid = hmax[:, None] * s[None, :]                                # (N, G)
    fg, _, mg = modes(grid.ravel(), np.repeat(theta, n_grid), np.repeat(phi, n_grid), v,
                      np.repeat(hx, n_grid))
    fg, mg = fg.reshape(grid.shape), mg.reshape(grid.shape + (3,))
    d = fg - f[:, None]
    cross = (np.sign(d[:, :-1]) != np.sign(d[:, 1:])) & np.isfinite(d[:, 1:])
    has = cross.any(axis=1)
    k = np.argmax(cross, axis=1)
    rows = np.arange(f.size)
    lo, hi = grid[rows, k], grid[rows, np.minimum(k + 1, n_grid - 1)]
    dlo = d[rows, k]
    m = mg[rows, k]                  # warm start: the grid point's equilibrium
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        fm, _, m = modes(mid, theta, phi, v, hx, start=m)
        dm = fm - f
        same = np.sign(dm) == np.sign(dlo)
        lo, dlo = np.where(same, mid, lo), np.where(same, dm, dlo)
        hi = np.where(same, hi, mid)
    return np.where(has, 0.5 * (lo + hi), np.nan)


# ──────────────────────────────── the settings ────────────────────────────────

@dataclass
class Settings:
    uniaxial: bool = False
    fourfold: bool = False
    sixfold: bool = False
    pssw: str = "free"            # "free": Hex per mode; "A": through the exchange stiffness
    d_nm: float | None = None     # film thickness, for A
    Ms_mT: float | None = None    # mu0 Ms, for A (NOT Meff: PMA changes Meff, not Ms)


#: the parameters as the operator sees them: (name, unit, meaning)
PARAMS = [("gamma", "GHz/T", "gamma/2pi"), ("Meff", "mT", "mu0 Meff"),
          ("Bu", "mT", "in-plane uniaxial, 2Ku/Ms"), ("phi_u", "deg", "uniaxial easy axis"),
          ("B4", "mT", "4-fold, 2K4/Ms"), ("phi_4", "deg", "4-fold easy axis"),
          ("B6", "mT", "6-fold, 2K6/Ms"), ("phi_6", "deg", "6-fold easy axis"),
          ("A", "pJ/m", "exchange stiffness")] + \
         [(f"Hex{n}", "mT", f"PSSW n={n}: exchange field") for n in (1, 2, 3, 4)]
DAMPING = [("alpha", "", "Gilbert damping"), ("dH0", "mT", "inhomogeneous FWHM")]
UNITS = {n: u for n, u, _ in PARAMS + DAMPING} | {"g": ""}
PERIOD = {"u": 180.0, "4": 90.0, "6": 60.0}
ORDER = {"u": 2, "4": 4, "6": 6}


def pssw_orders(points: list[Point]) -> list[int]:
    return sorted({p.role for p in points if p.use and p.role > 0})


def _kittel_meff(points, gamma=28.0) -> float:
    """mu0 Meff from in-plane uniform points, no anisotropy: f^2 = g^2 H (H + M)."""
    ms = []
    for p in points:
        if p.use and p.role == 0 and abs(p.theta - 90) < 20 and p.H > 1 and p.f > 0:
            ms.append((p.f / (gamma * 1e-3)) ** 2 / p.H - p.H)
    return float(np.median(ms)) if ms else 1000.0


GAMMA_FREE = 2.0023 * GAMMA_E      # 28.02 GHz/T, the free electron


def one_kittel_point(points: list[Point]) -> bool:
    """True when every uniform-mode resonance was taken at the same frequency
    (field sweeps) or the same field (frequency sweeps): then only ONE point of
    the Kittel curve is known per angle, and gamma and Meff cannot both be fitted."""
    use = [p for p in points if p.use and p.role == ROLE_UNIFORM]
    held = set()
    for p in use:
        if p.swept == "H":
            held.add(("f", round(p.f, 6)))
        elif p.swept == "f":
            held.add(("H", round(p.H, 6)))
        else:
            held.add(("Hf", round(p.H, 6), round(p.f, 6)))
    return len(held) <= 1


def default_specs(points: list[Point], settings: Settings) -> dict[str, Spec]:
    """Start values: gamma of the free electron (FIXED when the data have only
    one point of the Kittel curve, see one_kittel_point), Meff from the in-plane
    Kittel formula, anisotropies 0, each PSSW's exchange field from how far it
    sits from the uniform line."""
    meff = _kittel_meff(points)
    sp = {"gamma": Spec(GAMMA_FREE, vary=not one_kittel_point(points), min=10.0, max=60.0),
          "Meff": Spec(meff),
          "Bu": Spec(0.0), "phi_u": Spec(0.0), "B4": Spec(0.0), "phi_4": Spec(0.0),
          "B6": Spec(0.0), "phi_6": Spec(0.0), "A": Spec(10.0, min=0.0),
          "alpha": Spec(0.01, min=0.0), "dH0": Spec(0.0)}
    for n in (1, 2, 3, 4):
        guesses = []
        for p in points:
            if p.use and p.role == n and abs(p.theta - 90) < 20:
                b = p.f / 28e-3
                heff = (-meff + np.sqrt(meff * meff + 4 * b * b)) / 2
                guesses.append(heff - p.H)
        sp[f"Hex{n}"] = Spec(float(np.median(guesses)) if guesses else 100.0, min=0.0)
    return sp


def _a_coefficient(n: int, settings: Settings) -> float:
    """Hex_n (mT) per A (pJ/m): 2 A (n pi / d)^2 / Ms, unpinned surfaces."""
    k = n * np.pi / (settings.d_nm * 1e-9)
    return 2 * MU0 * 1e-12 * k * k / (settings.Ms_mT * 1e-3) * 1e3


def _use_A(settings: Settings) -> bool:
    return settings.pssw == "A" and bool(settings.d_nm) and bool(settings.Ms_mT)


# ─────────────────────────────── fitting positions ────────────────────────────

@dataclass
class DResult:
    values: dict[str, float]            # physical: gamma, g, Meff, Bu, phi_u, ..., Hex_n, A_n
    errors: dict[str, float | None]
    vary: dict[str, bool]
    internal: dict[str, float]          # what energy() / modes() take
    settings: Settings
    redchi: float
    ndata: int
    success: bool
    message: str
    orders: list[int] = field(default_factory=list)

    def hex(self, role: int) -> float:
        return 0.0 if role <= 0 else self.internal.get(f"Hex{role}", 0.0)


def _params(points, settings: Settings, specs: dict[str, Spec]):
    import lmfit
    P = lmfit.Parameters()

    def add(name, sp: Spec, vary=None, **kw):
        v = float(np.clip(sp.value, sp.min, sp.max))
        P.add(name, value=v, vary=sp.vary if vary is None else vary,
              min=sp.min, max=sp.max, **kw)

    add("gamma", specs["gamma"])
    add("Meff", specs["Meff"])
    P.add("g", expr=f"gamma / {GAMMA_E}")
    for key, on in (("u", settings.uniaxial), ("4", settings.fourfold), ("6", settings.sixfold)):
        n = ORDER[key]
        b, ph = specs[f"B{key}"], specs[f"phi_{key}"]
        if not on or not b.vary:
            # a fixed term: in the energy with the typed values (0 when switched off)
            val = b.value if on else 0.0
            P.add(f"B{key}", value=val, vary=False)
            P.add(f"phi_{key}", value=ph.value, vary=False)
        elif not ph.vary:
            # the axis is known: fit the strength along it (a sign flip is allowed)
            P.add(f"B{key}", value=b.value, vary=True)
            P.add(f"phi_{key}", value=ph.value, vary=False)
        else:
            # both free: Cartesian components, no angle wrapping, no kink at the axis
            r = b.value if abs(b.value) > 0.05 else 0.5
            a = np.deg2rad(n * ph.value)
            P.add(f"B{key}_c", value=r * np.cos(a))
            P.add(f"B{key}_s", value=r * np.sin(a))
            P.add(f"B{key}", expr=f"sqrt(B{key}_c**2 + B{key}_s**2)")
            P.add(f"phi_{key}", expr=f"degrees(arctan2(B{key}_s, B{key}_c)) / {n}")
    orders = pssw_orders(points)
    if orders:
        if _use_A(settings):
            add("A", specs["A"])
            for n in orders:
                P.add(f"Hex{n}", expr=f"A * {_a_coefficient(n, settings)!r}")
        else:
            for n in orders:
                add(f"Hex{n}", specs[f"Hex{n}"])
                if settings.d_nm and settings.Ms_mT:
                    P.add(f"A{n}", expr=f"Hex{n} / {_a_coefficient(n, settings)!r}")
    return P, orders


def _swept_derivative(pts_H, pts_t, pts_p, swept, v, hexes, m):
    """df/d(swept coordinate) per point (1 for frequency-swept points)."""
    d = np.ones(pts_H.size)
    for coord in ("H", "theta", "phi"):
        sel = swept == coord
        if np.any(sel):
            d[sel] = derivative(coord, pts_H[sel], pts_t[sel], pts_p[sel], v, hexes[sel],
                                m[sel])
    return d


def position_residuals(points: list[Point], v: dict, hexes=None,
                       cache: dict | None = None) -> np.ndarray:
    """Model minus data, in the unit of each point's sweep and in sigmas:
    frequency-swept points directly, the others through df/dx
    ((f_model - f) / (df/dx) is the distance along the sweep, to first order).
    `cache` keeps each point's equilibrium between calls of one fit, so only
    the first call searches from every start."""
    a = arrays(points)
    swept = np.array([p.swept for p in points])
    if hexes is None:
        hexes = np.array([_hex_of(v, p.role) for p in points])
    start = cache.get("m") if cache is not None else None
    f, _, m = modes(a["H"], a["theta"], a["phi"], v, hexes, start=start)
    if cache is not None:
        cache["m"] = m
    d = _swept_derivative(a["H"], a["theta"], a["phi"], swept, v, hexes, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (f - a["f"]) / d
    err = np.array([p.err for p in points])
    w = np.where(np.isfinite(err) & (err > 0), err, 1.0)
    return np.where(np.isfinite(r), r / w, 1e6)


def _hex_of(v: dict, role: int) -> float:
    return 0.0 if role <= 0 else float(v.get(f"Hex{role}", 0.0))


def _wrap(ph: float, period: float) -> float:
    w = (ph + period / 2) % period - period / 2
    return period / 2 if np.isclose(w, -period / 2) else w


def fit_positions(points: list[Point], settings: Settings,
                  specs: dict[str, Spec] | None = None) -> DResult:
    """gamma, Meff, anisotropies (and PSSW exchange) from the resonance positions."""
    import lmfit
    use = [p for p in points if p.use and p.role != ROLE_IGNORE]
    if len(use) < 2:
        raise ValueError("tick at least two resonances to fit")
    specs = specs or default_specs(use, settings)
    P, orders = _params(use, settings, specs)
    nfree = sum(1 for p in P.values() if p.vary)
    if len(use) < nfree:
        raise ValueError(f"{len(use)} points for {nfree} free parameters -- fix some, or "
                         "add points")
    cache: dict = {}
    free = [n for n, p in P.items() if p.vary]
    anis = [n for n in free if n.startswith(("Bu", "B4", "B6"))]
    # In stages, gamma LAST: (1) Meff and the exchange fields, (2) + the
    # anisotropy, (3) everything. Found on the demo film, 2026-09-29:
    # * all at once from zero anisotropy, a start 0.02 GHz/T different in gamma
    #   ended in a false minimum (gamma 23.5, chi2 150x the truth's);
    # * gamma free in the first stage, with the anisotropy not yet modelled, ran
    #   along the gamma-Meff valley to gamma = 10 GHz/T and Meff = 12 T -- two
    #   points of the Kittel curve per angle pin the RATIO poorly -- crawling for
    #   thousands of evaluations. gamma is close to its start (g ~ 2) for a metal
    #   film, so holding it until the rest is right is the physical order.
    s1 = [n for n in free if n not in anis and n != "gamma"]
    s2 = s1 + anis
    stages = [st for st in (s1, s2, free) if st]
    stages = [st for i, st in enumerate(stages) if i == 0 or st != stages[i - 1]]
    out = None
    for stage in stages:
        for n in free:
            P[n].set(vary=n in stage)
        # epsfcn: parameter steps of 1e-4 (relative) for the Jacobian. MINPACK's
        # default (~1e-8) is as small as the model's numerical noise, and the fit
        # then does not move at all (it stayed at the start values, 2026-09-29).
        out = lmfit.minimize(lambda p: position_residuals(use, p.valuesdict(), cache=cache),
                             P, method="leastsq", epsfcn=1e-8,
                             max_nfev=300 * (len(stage) + 1))
        P = out.params
    return _result(out, settings, orders)


def _result(out, settings, orders) -> DResult:
    vals = {n: float(p.value) for n, p in out.params.items()}
    # an expression of fixed parameters (g while gamma is held) gets stderr 0:
    # that is "not fitted", not "exact"
    errs = {n: (float(p.stderr) if p.stderr is not None and np.isfinite(p.stderr)
                and (p.vary or (p.expr and p.stderr > 0)) else None)
            for n, p in out.params.items()}
    internal = dict(vals)
    phys = {k: v for k, v in vals.items() if not k.endswith(("_c", "_s"))}
    # one canonical form: strength >= 0, axis inside one period
    for key in ("u", "4", "6"):
        b, ph = phys.get(f"B{key}", 0.0), phys.get(f"phi_{key}", 0.0)
        if b < 0:
            b, ph = -b, ph + PERIOD[key] / ORDER[key] * ORDER[key] / 2
        phys[f"B{key}"], phys[f"phi_{key}"] = b, _wrap(ph, PERIOD[key])
    vary = {n: bool(p.vary or p.expr) for n, p in out.params.items()}
    return DResult(values=phys, errors=errs, vary=vary, internal=internal, settings=settings,
                   redchi=float(out.redchi), ndata=int(out.ndata), success=bool(out.success),
                   message=str(out.message), orders=orders)


# ─────────────────────────────── what to draw ─────────────────────────────────

AXIS = {"H": ("μ0H", "mT"), "f": ("f", "GHz"), "phi": ("φ_H", "deg"), "theta": ("θ_H", "deg")}


@dataclass
class Series:
    label: str
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None = None
    role: int = ROLE_UNIFORM


@dataclass
class Plot:
    x: str                              # coordinate on the x axis: H, f, phi, theta
    y: str                              # ... on the y axis, or "fwhm"
    x_title: str
    y_title: str
    data: list[Series] = field(default_factory=list)
    lines: list[Series] = field(default_factory=list)


def role_name(role: int) -> str:
    return {ROLE_UNIFORM: "uniform", ROLE_IGNORE: "ignored"}.get(role, f"PSSW n={role}")


def layout(points: list[Point]) -> tuple[str, str, list[Point]]:
    """(x coordinate, swept coordinate, the points shown).

    A dispersion is drawn against what CHANGES between the curves: the angle if
    it does (resonance vs phi_H), otherwise in the field-frequency plane (f vs
    mu0H, the Kittel plot). Points of the most common sweep type are shown; a
    mixed set (field and frequency sweeps) is fitted together all the same.
    """
    use = [p for p in points if p.use and p.role != ROLE_IGNORE]
    if not use:
        return "H", "H", []
    kinds = [p.swept for p in use]
    swept = max(set(kinds), key=kinds.count)
    sel = [p for p in use if p.swept == swept]
    for k in ("phi", "theta"):
        if k != swept and len({round(getattr(p, k), 6) for p in sel}) > 1:
            return k, swept, sel
    return "H", swept, sel


def _groups(sel: list[Point], keys: tuple[str, ...]) -> dict[tuple, list[Point]]:
    out: dict[tuple, list[Point]] = {}
    for p in sel:
        out.setdefault((p.role,) + tuple(round(getattr(p, k), 6) for k in keys), []).append(p)
    return out


def _title(k: str) -> str:
    n, u = AXIS[k]
    return f"{n} ({u})"


def position_plot(points: list[Point], dres: DResult | None = None, n: int = 200) -> Plot:
    """Resonances and the fitted model, ready to draw."""
    x, swept, sel = layout(points)
    if x == "H":                           # the Kittel plane: f against mu0H
        plot = Plot("H", "f", _title("H"), _title("f"))
        other = ("phi", "theta")
    else:
        plot = Plot(x, swept, _title(x), _title(swept))
        other = tuple(k for k in ("H", "f", "phi", "theta") if k not in (x, swept))
    for key, pts in _groups(sel, other).items():
        role = key[0]
        xs = np.array([getattr(p, plot.x) for p in pts])
        ys = np.array([getattr(p, plot.y) for p in pts])
        # error bars only along y, and only when y is the swept coordinate
        err = (np.array([p.err for p in pts]) if plot.y == swept else None)
        tag = ", ".join(f"{AXIS[k][0]} = {v:g} {AXIS[k][1]}" for k, v in zip(other, key[1:])
                        if len({round(getattr(q, k), 6) for q in sel}) > 1)
        plot.data.append(Series(role_name(role) + (f" ({tag})" if tag else ""), xs, ys,
                                err, role))
        if dres is None:
            continue
        hx = dres.hex(role)
        c = {k: v for k, v in zip(other, key[1:])}
        lo, hi = float(np.min(xs)), float(np.max(xs))
        if plot.x == "H":
            grid = np.linspace(0.0, max(hi, 1.0) * 1.15, n)
            yy = frequency(grid, c["theta"], c["phi"], dres.internal, hx)
        else:
            pad = 0.02 * (hi - lo) if hi > lo else 1.0
            grid = np.linspace(lo - pad, hi + pad, n)
            args = {"H": c.get("H"), "f": c.get("f"), "theta": c.get("theta"),
                    "phi": c.get("phi"), plot.x: grid}
            if swept == "f":
                yy = frequency(np.broadcast_to(args["H"], grid.shape), args["theta"],
                               args["phi"], dres.internal, hx)
            else:
                yy = resonance_field(np.broadcast_to(args["f"], grid.shape), args["theta"],
                                     args["phi"], dres.internal, hx)
        plot.lines.append(Series(f"model, {role_name(role)}", grid, yy, None, role))
    return plot


def width_plot(points: list[Point], dres: DResult | None = None,
               damp: "DampResult | None" = None, n: int = 120) -> Plot | None:
    """The uniform mode's FWHM against the frequency (Kittel plane) or the angle,
    with the damping model."""
    x, swept, sel = layout(points)
    sel = [p for p in sel if p.role == ROLE_UNIFORM and np.isfinite(p.fwhm)]
    if not sel:
        return None
    kittel = x == "H"                      # the Kittel plane: FWHM against f
    xk = "f" if kittel else x
    unit = AXIS[swept][1]
    plot = Plot(xk, "fwhm", _title(xk), f"FWHM ({unit})")
    other = (("phi", "theta") if kittel
             else tuple(k for k in ("H", "f", "phi", "theta") if k not in (xk, swept)))
    for key, pts in _groups(sel, other).items():
        xs = np.array([getattr(p, xk) for p in pts])
        plot.data.append(Series("uniform", xs, np.array([p.fwhm for p in pts]),
                                np.array([p.fwhm_err for p in pts])))
        if dres is None or damp is None:
            continue
        c = {k: v for k, v in zip(other, key[1:])}
        if kittel and swept == "f":
            # frequency sweeps at many fields: walk along the field
            hs = np.array([p.H for p in pts])
            H = np.linspace(max(hs.min() * 0.95, 0.0), hs.max() * 1.05, n)
            th, ph = np.full(n, c["theta"]), np.full(n, c["phi"])
            f = frequency(H, th, ph, dres.internal)
            grid = f
        else:
            lo, hi = float(np.min(xs)), float(np.max(xs))
            pad = 0.02 * (hi - lo) if hi > lo else 1.0
            grid = np.linspace(max(lo - pad, 0.0) if xk == "f" else lo - pad, hi + pad, n)
            coords = {**c, xk: grid}
            f = np.broadcast_to(np.asarray(coords.get("f", 0.0), float), grid.shape)
            th = np.broadcast_to(np.asarray(coords["theta"], float), grid.shape)
            ph = np.broadcast_to(np.asarray(coords["phi"], float), grid.shape)
            if swept == "H":
                H = resonance_field(f, th, ph, dres.internal)
            else:
                H = np.broadcast_to(np.asarray(coords["H"], float), grid.shape)
                f = frequency(H, th, ph, dres.internal)
        ok = np.isfinite(H) & np.isfinite(f)
        model_pts = [Point(H=h, f=ff, theta=t, phi=q, swept=swept)
                     for h, ff, t, q in zip(H[ok], f[ok], th[ok], ph[ok])]
        yy = np.full(grid.shape, np.nan)
        if model_pts:
            yy[ok] = linewidth(model_pts, dres.internal, damp.alpha, damp.dH0)
        plot.lines.append(Series("model", np.asarray(grid), yy))
    return plot


def result_rows(dres: DResult, damp: "DampResult | None" = None) -> list[tuple]:
    """(name, value, error, unit, meaning) for the results table and the file."""
    rows = []
    meaning = {n: m for n, _, m in PARAMS + DAMPING}
    order = ["gamma", "g", "Meff"]
    for key, on in (("u", dres.settings.uniaxial), ("4", dres.settings.fourfold),
                    ("6", dres.settings.sixfold)):
        if on:
            order += [f"B{key}", f"phi_{key}"]
    for n in dres.orders:
        order.append(f"Hex{n}")
        if f"A{n}" in dres.values:
            order.append(f"A{n}")
    if "A" in dres.values:
        order.append("A")
    for n in order:
        if n in dres.values:
            unit = "pJ/m" if n.startswith("A") else UNITS.get(n, UNITS.get(n.rstrip("0123456789"), ""))
            text = "g-factor" if n == "g" else meaning.get(n, meaning.get(
                n.rstrip("0123456789"), "")) or (f"PSSW n={n[1:]}: exchange stiffness"
                                                  if n.startswith("A") else "")
            rows.append((n, dres.values[n], dres.errors.get(n), unit, text))
    if damp is not None:
        rows.append(("alpha", damp.alpha, damp.alpha_err, "", "Gilbert damping"))
        rows.append(("dH0", damp.dH0, damp.dH0_err, "mT", "inhomogeneous FWHM"))
    return rows


def figure_dispersion(pos: Plot, wid: Plot | None, title: str = "", size=(6.4, 7.0)):
    """Both plots, publication style (white), like AaltoView's figures."""
    from aaltoview.export import _figure
    fig = _figure(size)
    axes = fig.subplots(2 if wid else 1, 1, squeeze=False)[:, 0]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for ax, plot in zip(axes, [pos] + ([wid] if wid else [])):
        for k, s in enumerate(plot.data):
            col = colors[max(s.role, 0) % len(colors)] if plot is pos else colors[k % 6]
            ax.errorbar(s.x, s.y, yerr=s.yerr, fmt="o", ms=3.5, color=col, label=s.label,
                        capsize=0, lw=1)
        for s in plot.lines:
            col = colors[max(s.role, 0) % len(colors)] if plot is pos else "0.2"
            ax.plot(s.x, s.y, "-", color=col, lw=1.3)
        ax.set_xlabel(plot.x_title)
        ax.set_ylabel(plot.y_title)
        ax.legend(fontsize=8, frameon=False)
    if title:
        axes[0].set_title(title, fontsize=9)
    return fig


# ───────────────────────────────── damping ────────────────────────────────────

@dataclass
class DampResult:
    alpha: float
    alpha_err: float | None
    dH0: float
    dH0_err: float | None
    redchi: float
    ndata: int


def _linewidth_terms(points: list[Point], v: dict):
    """Per point: frequency FWHM per unit alpha (GHz), df/dH, and df/d(swept)."""
    a = arrays(points)
    swept = np.array([p.swept for p in points])
    hexes = np.array([_hex_of(v, p.role) for p in points])
    _, k_alpha, m = modes(a["H"], a["theta"], a["phi"], v, hexes)
    dfdH = derivative("H", a["H"], a["theta"], a["phi"], v, hexes, m)
    dsw = _swept_derivative(a["H"], a["theta"], a["phi"], swept, v, hexes, m)
    return k_alpha, dfdH, dsw


def linewidth(points: list[Point], v: dict, alpha: float, dH0: float,
              terms=None) -> np.ndarray:
    """Model FWHM of each point, in its sweep's unit:
    FWHM_f = alpha (gamma/2pi)(E_aa + E_bb + 2Hex) + dH0 df/dH, divided by |df/dx|."""
    k_alpha, dfdH, dsw = terms if terms is not None else _linewidth_terms(points, v)
    return (alpha * k_alpha + dH0 * np.abs(dfdH)) / np.abs(dsw)


def fit_damping(points: list[Point], dres: DResult,
                start: dict[str, Spec] | None = None) -> DampResult:
    """alpha and the inhomogeneous width dH0 from the FWHMs of the uniform mode,
    with the positions' parameters held (the linewidth is linear in both)."""
    import lmfit
    use = [p for p in points if p.use and p.role == ROLE_UNIFORM and np.isfinite(p.fwhm)]
    if len(use) < 2:
        raise ValueError("need linewidths of at least two uniform-mode resonances")
    terms = _linewidth_terms(use, dres.internal)
    y = np.array([p.fwhm for p in use])
    e = np.array([p.fwhm_err for p in use])
    w = np.where(np.isfinite(e) & (e > 0), e, 1.0)
    start = start or {"alpha": Spec(0.01, min=0.0), "dH0": Spec(0.0)}
    P = lmfit.Parameters()
    for n in ("alpha", "dH0"):
        sp = start[n]
        P.add(n, value=float(np.clip(sp.value, sp.min, sp.max)), vary=sp.vary,
              min=sp.min, max=sp.max)
    out = lmfit.minimize(lambda p: (linewidth(use, dres.internal, p["alpha"].value,
                                              p["dH0"].value, terms) - y) / w, P,
                         method="leastsq")
    pe = out.params

    def err(n):
        s = pe[n].stderr
        return float(s) if pe[n].vary and s is not None and np.isfinite(s) else None
    return DampResult(alpha=float(pe["alpha"].value), alpha_err=err("alpha"),
                      dH0=float(pe["dH0"].value), dH0_err=err("dH0"),
                      redchi=float(out.redchi), ndata=int(out.ndata))
