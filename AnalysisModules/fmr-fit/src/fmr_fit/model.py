"""model.py -- the FMR lineshape and fitting it. No Qt: tested headless.

What a VNA measures near a resonance (field sweep at a fixed frequency, or a
frequency sweep at a fixed field -- the same maths with x = f):

    S(x) = sum_k  A_k e^{i phi_k} chi(x; x0_k, dx_k)  +  b0 + b1 (x - xc)

    chi(x; x0, dx) = dx / (x0 - x - i dx)       the complex Lorentzian

* Im chi is the SYMMETRIC (absorption) peak, Re chi the ANTISYMMETRIC
  (dispersion) one; |chi| = 1 on resonance. A VNA that is not calibrated at the
  sample rotates the signal by an unknown phase, which mixes the two shapes.
  phi is fitted, so the mixing costs nothing: the width comes out right whatever
  the calibration.
* dx is the HALF width at half maximum (HWHM) of the absorption peak, in the
  unit of x. FWHM = 2 dx. Keep the factor 2 in mind: it goes straight into the
  damping (alpha from the slope of linewidth vs frequency) later.
* NOT the derivative lineshape. A field-modulated lock-in measures d chi/dH, a
  different curve with a different width; this model is for direct detection.
* b0, b1: a background that is constant or linear in x (complex when the data
  is complex), measured from the centre of the fit range xc.

Two ways to fit (Setup.mode):
* "complex": Re and Im TOGETHER, when the curve carries complex data. Twice
  the data for the same parameters, and the phase is pinned by the data rather
  than by the shape alone: the most reliable x0 and dx.
* "real": one real channel (what the viewer showed: Re, Im, |S21|). The model
  is Re(S). |S21| works too: a small resonance on a large background changes
  the magnitude by Re(signal x e^{-i arg background}), which is again a mixed
  Lorentzian.

Handedness: whether the signal turns one way or the other through the
resonance depends on the sweep (field or frequency) and on the instrument's
sign convention. Complex conjugation cannot be undone by a phase, so in complex
mode the fit tries both (Setup.hand = 0) and keeps the better one; the result
says which it was. In real mode it does not matter (the phase absorbs it).

Options for FREQUENCY sweeps, where a VNA is harder to fit than in field:

* Setup.lineshape = "oscillator": the damped-oscillator form, exact in f,
      chi(f) = 2 f0 dx / (f0^2 - f^2 - i f 2 dx)
  instead of the Lorentzian, which is its near-resonance limit (same x0 and
  HWHM dx). The Lorentzian is off by ~ dx/f0: a few % in f0 and the width for a
  broad line at low frequency. For field sweeps keep the Lorentzian.
* Setup.delay: the cable's electrical delay, the whole complex signal times
  e^{-i 2 pi tau (x - xc)} -- the phase that winds with frequency. tau in 1/x
  (ns for x in GHz), started from the phase slope at the ends of the range.
* Setup.dd = k > 0: DERIVATIVE-DIVIDE (Maier-Flaig et al., Rev. Sci. Instrum.
  89, 076101 (2018)). The data become
      D(x) = (S(x+) - S(x-)) / ((x+ - x-) S(x)),    x+- = k points either side
  which removes a background that MULTIPLIES the signal and varies slowly
  (cable loss, impedance mismatch); the model goes through the same
  transformation exactly: with S = B (1 + P), P the peaks,
      D = (P(x+) - P(x-)) / ((x+ - x-)(1 + P(x))) + c0 + c1 (x - xc)
  so the amplitude is RELATIVE to the background and (c0, c1) is what is left
  of d ln B / dx. Complex data only.
  A cable DELAY does not vary slowly: its phase turns by 2 pi tau (x+ - x) between
  neighbours (0.6 rad for 3.2 ns and 30 MHz), which rotates the peaks in D. With
  Setup.delay the model carries that exactly:
      D = (e^{-i2pi tau (x+ - x)} (1 + P(x+)) - e^{-i2pi tau (x- - x)} (1 + P(x-)))
          / ((x+ - x-)(1 + P(x))) + c0 + c1 (x - xc)
  -- use both options together for a VNA frequency sweep.
* reference(): subtract or divide by another sweep -- one at a field where
  nothing resonates in the band -- before fitting (done by the caller).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODES = ("complex", "real")
BASELINES = ("constant", "linear")
PEAK_PARAMS = ("center", "hwhm", "amp", "phase")


LINESHAPES = ("lorentzian", "oscillator")


def chi(x, center, hwhm, hand: int = 1, lineshape: str = "lorentzian") -> np.ndarray:
    """The complex susceptibility; hand = -1 is its mirror image. |chi| = 1 and
    chi = +i on resonance for both lineshapes."""
    x = np.asarray(x, dtype=float)
    if lineshape == "oscillator":
        c = 2 * center * hwhm / (center * center - x * x - 1j * x * 2 * hwhm)
    else:
        c = hwhm / (center - x - 1j * hwhm)
    return c if hand >= 0 else np.conj(c)


# ─────────────────────────────── what to fit ──────────────────────────────────

@dataclass
class Setup:
    n_peaks: int = 1
    mode: str = "complex"
    baseline: str = "linear"
    hand: int = 0                       # +1, -1, or 0 = try both (complex mode)
    xmin: float | None = None           # the fit range; None = the data's end
    xmax: float | None = None
    lineshape: str = "lorentzian"       # or "oscillator" (exact in frequency)
    delay: bool = False                 # fit the electrical delay (complex, no dd)
    dd: int = 0                         # derivative-divide, k points either side; 0 = off


@dataclass
class Spec:
    """One parameter as the operator set it: start value, free or fixed, bounds."""
    value: float
    vary: bool = True
    min: float = -np.inf
    max: float = np.inf


def peak_names(k: int) -> list[str]:
    return [f"p{k}_{p}" for p in PEAK_PARAMS]


def param_names(setup: Setup) -> list[str]:
    names = []
    for k in range(1, setup.n_peaks + 1):
        names += peak_names(k)
    if setup.mode == "complex":
        names += ["bg_re", "bg_im"] + (["slope_re", "slope_im"]
                                       if setup.baseline == "linear" else [])
        if setup.delay:
            names.append("delay")
    else:
        names += ["bg"] + (["slope"] if setup.baseline == "linear" else [])
    return names


_INVERSE = {"GHz": "ns", "MHz": "us", "kHz": "ms", "Hz": "s"}


def param_unit(name: str, x_unit: str, y_unit: str, setup: Setup | None = None) -> str:
    """The unit a parameter is in, for tables and file headers."""
    inv = _INVERSE.get(x_unit, f"1/{x_unit}" if x_unit else "")
    dd = setup is not None and setup.dd
    if name.endswith(("_center", "_hwhm", "_fwhm")):
        return x_unit
    if name.endswith("_phase"):
        return "deg"
    if name == "delay":
        return inv
    if dd:
        # derivative-divide: amplitude relative to the background, the rest 1/x
        if name.endswith("_amp"):
            return ""
        if name.startswith("slope"):
            return f"1/{x_unit}^2" if x_unit else ""
        return f"1/{x_unit}" if x_unit else ""
    if name.startswith("slope"):
        return f"{y_unit}/{x_unit}" if (y_unit or x_unit) else ""
    return y_unit                       # amplitude, background


# ──────────────────────────────── the data ────────────────────────────────────

@dataclass
class Data:
    x: np.ndarray
    y: np.ndarray                       # complex (complex mode) or real
    xc: float                           # centre of the range: the baseline's origin
    xp: np.ndarray | None = None        # derivative-divide: the neighbours of each x
    xm: np.ndarray | None = None
    dd_half: float | None = None        # ... their typical half-distance (for drawing)


def reference(x, y, xr, yr, how: str = "divide") -> np.ndarray:
    """y with a reference sweep taken out: "subtract" (y - ref) or "divide"
    (y / ref), the reference interpolated onto x (Re and Im separately); NaN
    where the reference does not reach. Divide for a VNA: the background
    multiplies the signal (cable loss, delay, mismatch)."""
    x = np.asarray(x, dtype=float)
    xr = np.asarray(xr, dtype=float)
    yr = np.asarray(yr)
    ok = np.isfinite(xr) & np.isfinite(np.abs(yr))
    xr, yr = xr[ok], yr[ok]
    order = np.argsort(xr)
    xr, yr = xr[order], yr[order]
    out = np.interp(x, xr, np.real(yr), left=np.nan, right=np.nan)
    if np.iscomplexobj(yr):
        out = out + 1j * np.interp(x, xr, np.imag(yr), left=np.nan, right=np.nan)
    if how == "subtract":
        return np.asarray(y) - out
    if how == "divide":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(y) / out
    raise ValueError(f"reference: subtract or divide, not '{how}'")


def prepare(x, y, setup: Setup) -> Data:
    """The points that take part: finite, sorted by x, derivative-divided if
    asked (on the whole sweep, so the range does not eat the neighbours),
    inside the range. NaN-aware: a running or aborted scan has holes, and they
    are left out."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if setup.mode == "complex" and not np.iscomplexobj(y):
        raise ValueError("complex fit needs complex data -- this curve is real; "
                         "use 'one channel'")
    if setup.mode == "real" and np.iscomplexobj(y):
        raise ValueError("'one channel' fits real data; pass Re, Im or |z|")
    if setup.dd and setup.mode != "complex":
        raise ValueError("derivative-divide needs complex data (Re + Im together)")
    ok = np.isfinite(x) & np.isfinite(np.abs(y))
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    xp = xm = None
    if setup.dd:
        k = int(setup.dd)
        if x.size <= 2 * k + 5:
            raise ValueError(f"derivative-divide with k = {k} needs more than {2 * k + 5} "
                             "points")
        xp, xm = x[2 * k:], x[:-2 * k]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = (y[2 * k:] - y[:-2 * k]) / ((xp - xm) * y[k:-k])
        x = x[k:-k]
        ok = np.isfinite(np.abs(y))
        x, y, xp, xm = x[ok], y[ok], xp[ok], xm[ok]
    sel = np.ones(x.size, dtype=bool)
    if setup.xmin is not None:
        sel &= x >= setup.xmin
    if setup.xmax is not None:
        sel &= x <= setup.xmax
    x, y = x[sel], y[sel]
    if xp is not None:
        xp, xm = xp[sel], xm[sel]
    n_free = len(param_names(setup)) * (1 if setup.mode == "real" else 0.5)
    if x.size < max(5, n_free + 2):
        raise ValueError(f"only {x.size} measured points in the fit range -- too few")
    return Data(x=x, y=y, xc=0.5 * (x[0] + x[-1]), xp=xp, xm=xm,
                dd_half=None if xp is None else float(np.median(xp - xm) / 2))


def displayed(x, y, setup: Setup) -> Data:
    """What the fit sees over the WHOLE sweep (the range ignored): the data to
    draw -- derivative-divided when that is on."""
    return prepare(x, y, Setup(**{**setup.__dict__, "xmin": None, "xmax": None}))


def _peaks(values: dict, x, setup: Setup, hand: int) -> np.ndarray:
    s = np.zeros(np.shape(x), dtype=complex)
    for k in range(1, setup.n_peaks + 1):
        s += (values[f"p{k}_amp"] * np.exp(1j * np.deg2rad(values[f"p{k}_phase"]))
              * chi(x, values[f"p{k}_center"], values[f"p{k}_hwhm"], hand, setup.lineshape))
    return s


def evaluate(values: dict, x, setup: Setup, hand: int, xc: float, xp=None, xm=None,
             dd_half: float | None = None) -> np.ndarray:
    """The model at x: complex in complex mode, real (= Re S) in real mode.
    Derivative-divide needs each point's neighbours xp, xm -- or, for a smooth
    line to draw, their typical half-distance dd_half."""
    x = np.asarray(x, dtype=float)
    if setup.dd:
        if xp is None:
            h = dd_half if dd_half else 1e-6 * max(1.0, float(np.max(np.abs(x))))
            xp, xm = x + h, x - h
        pp, pm = 1 + _peaks(values, xp, setup, hand), 1 + _peaks(values, xm, setup, hand)
        if setup.delay:
            tau = values["delay"]
            pp = pp * np.exp(-2j * np.pi * tau * (xp - x))
            pm = pm * np.exp(-2j * np.pi * tau * (xm - x))
        s = (pp - pm) / ((xp - xm) * (1 + _peaks(values, x, setup, hand)))
        s = s + values["bg_re"] + 1j * values["bg_im"]
        if setup.baseline == "linear":
            s = s + (values["slope_re"] + 1j * values["slope_im"]) * (x - xc)
        return s
    s = _peaks(values, x, setup, hand)
    if setup.mode == "complex":
        s += values["bg_re"] + 1j * values["bg_im"]
        if setup.baseline == "linear":
            s += (values["slope_re"] + 1j * values["slope_im"]) * (x - xc)
        if setup.delay:
            s = s * np.exp(-2j * np.pi * values["delay"] * (x - xc))
        return s
    out = s.real + values["bg"]
    if setup.baseline == "linear":
        out = out + values["slope"] * (x - xc)
    return out


# ─────────────────────────────── start values ─────────────────────────────────

def _smooth(v: np.ndarray, n: int = 5) -> np.ndarray:
    if v.size < 2 * n:
        return v
    k = np.ones(n) / n
    return np.convolve(np.pad(v, n // 2, mode="edge"), k, mode="valid")[: v.size]


def _analytic(r: np.ndarray) -> np.ndarray:
    """A real channel -> a complex signal with the same shape (Hilbert transform).
    Re and Im of a Lorentzian are a Hilbert pair (Kramers-Kronig), so this gives
    back chi up to a phase and a mirror: good enough for a START value."""
    from scipy.signal import hilbert
    return hilbert(r - np.mean(r))


def _guess_peak(x, z, hand: int, allowed=None, bounds=None) -> dict[str, float]:
    """Largest bump of |z| (where `allowed`): its position, its width at
    1/sqrt(2) of the top (|chi| = 1/sqrt(2) at x0 +- dx, measured within
    `bounds` = (first, last) index), its height and phase."""
    mag = _smooth(np.abs(z))
    search = mag if allowed is None else np.where(allowed, mag, -np.inf)
    i = int(np.argmax(search))
    top = mag[i]
    first, last = bounds if bounds is not None else (0, x.size - 1)
    lo = i
    while lo > first and mag[lo] > top / np.sqrt(2):
        lo -= 1
    hi = i
    while hi < last and mag[hi] > top / np.sqrt(2):
        hi += 1
    step = np.median(np.diff(x)) if x.size > 1 else 1.0
    w = max(0.5 * (x[hi] - x[lo]), step)
    # on resonance chi = +i (hand +1) or -i (hand -1)
    phase = np.rad2deg(np.angle(z[i]) - hand * np.pi / 2)
    return {"center": float(x[i]), "hwhm": float(w), "amp": float(top),
            "phase": float(_wrap(phase))}


def _robust_background(A: np.ndarray, y: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Background coefficients: from the ends of the range first, then refitted
    on every point that does not stand out (|residual| < 3 median deviations),
    a few times. The ends alone fail when a resonance sits there: a 16 GHz YIG
    sweep has its two strongest lines in the top 10 % of the field range, the
    tilted "background" made a false peak and the weakest mode was lost."""
    coef, *_ = np.linalg.lstsq(A[edge], y[edge], rcond=None)
    for _ in range(5):
        r = np.abs(y - A @ coef)
        mad = np.median(r)
        keep = r < 3 * mad if mad > 0 else np.ones(y.size, dtype=bool)
        if keep.sum() < max(10, 2 * A.shape[1]):
            break
        new, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        if np.allclose(new, coef, rtol=1e-6, atol=1e-12 * (np.abs(coef).max() + 1)):
            break
        coef = new
    return coef


def _candidates(x, z) -> list[int]:
    """Indices of local maxima of the smoothed |z|, most prominent first,
    keeping only those that stand out of the noise (5 x its point-to-point
    scatter)."""
    from scipy.signal import find_peaks
    mag = _smooth(np.abs(z))
    if mag.size < 5:
        return []
    noise = 1.4826 * np.median(np.abs(np.diff(np.abs(z)))) / np.sqrt(2)
    idx, props = find_peaks(mag, prominence=max(5 * noise, 1e-300))
    order = np.argsort(props["prominences"])[::-1]
    return [int(i) for i in idx[order]]


def _wrap(deg: float) -> float:
    """An angle in (-180, 180]."""
    w = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def guess(x, y, setup: Setup, hand: int = 1,
          keep: dict[str, Spec] | None = None) -> dict[str, Spec]:
    """Start values for every parameter.

    Background from the outer 10 % of the range on each side (a resonance is
    in the middle of a sensible sweep). Peaks from the largest remaining bump,
    one after the other. Peaks the operator already has in `keep` are kept as
    they are -- adding a second peak must not throw away a tuned first one.
    """
    d = prepare(x, y, setup)
    if setup.dd and setup.delay:
        return _guess_dd_from_raw(x, y, setup, hand, keep)
    if setup.dd:
        return _guess_dd(d, setup, hand, keep)
    n = d.x.size
    m = max(3, n // 10)
    edge = np.r_[np.arange(m), np.arange(n - m, n)]
    specs: dict[str, Spec] = {}
    yw = d.y
    if setup.mode == "complex" and setup.delay:
        # the delay winds the phase linearly with x: its slope at the two ends
        if keep and "delay" in keep:
            tau = keep["delay"].value
            specs["delay"] = keep["delay"]
        else:
            tau = _delay_from_phase(d)
            specs["delay"] = Spec(float(tau))
        yw = d.y * np.exp(2j * np.pi * tau * (d.x - d.xc))
    cols = [np.ones(n)] + ([d.x - d.xc] if setup.baseline == "linear" else [])
    A = np.stack(cols, axis=1)
    coef = _robust_background(A, yw, edge)
    bg = A @ coef
    if setup.mode == "complex":
        specs["bg_re"], specs["bg_im"] = Spec(float(coef[0].real)), Spec(float(coef[0].imag))
        if setup.baseline == "linear":
            specs["slope_re"] = Spec(float(coef[1].real))
            specs["slope_im"] = Spec(float(coef[1].imag))
    else:
        specs["bg"] = Spec(float(coef[0].real))
        if setup.baseline == "linear":
            specs["slope"] = Spec(float(coef[1].real))

    r = yw - bg
    z = r if setup.mode == "complex" else _analytic(r)
    span = d.x[-1] - d.x[0]
    # a peak found is not looked for again within a few widths of it: with
    # narrow lines (YIG, ~9 points) the first estimate is not exact, what it
    # leaves behind is still the biggest bump, and all five "peaks" of a YIG
    # sweep landed within 0.5 mT of the uniform mode (2026-09-29)
    allowed = np.ones(d.x.size, dtype=bool)
    # Candidates: local maxima of |z| ranked by PROMINENCE. The tails of a line
    # have no local maxima, so what the strong line leaves behind is never
    # taken for a weak mode (a 0.3 uV PSSW next to a 10 uV uniform line was
    # missed that way); noise has little prominence. Taken in order, then the
    # residual search below when they run out.
    cands = _candidates(d.x, z)[: setup.n_peaks]
    cand_pos = list(cands)
    w_first = None
    for k in range(1, setup.n_peaks + 1):          # kept peaks: their lines are taken
        if keep and all(nm in keep for nm in peak_names(k)):
            c0, w0 = keep[f"p{k}_center"].value, keep[f"p{k}_hwhm"].value
            allowed &= np.abs(d.x - c0) > 4 * max(w0, 1e-12)
    for k in range(1, setup.n_peaks + 1):
        names = peak_names(k)
        if keep and all(nm in keep for nm in names):
            vals = {p: keep[f"p{k}_{p}"].value for p in PEAK_PARAMS}
            specs.update({nm: keep[nm] for nm in names})
        else:
            if not allowed.any():
                allowed[:] = True
            # not a line already taken -- by a peak found, or KEPT: going from 1
            # to 5 peaks kept peak 1 on the uniform line, and peak 2 was put on
            # the same line again (YIG screenshot, 2026-09-29)
            while cands and not allowed[cands[0]]:
                cands.pop(0)
            if cands:
                # a candidate is its own local maximum: used as it is, its width
                # measured no further than halfway to the neighbouring ones (a
                # weak line's half-height walk ran on through the noise, and the
                # zone around it hid the next mode)
                i = cands.pop(0)
                only = np.zeros(d.x.size, dtype=bool)
                only[i] = True
                left = max([j for j in cand_pos if j < i], default=None)
                right = min([j for j in cand_pos if j > i], default=None)
                bounds = (0 if left is None else (left + i) // 2,
                          d.x.size - 1 if right is None else (i + right) // 2)
                vals = _guess_peak(d.x, z, hand, only, bounds)
            else:
                vals = _guess_peak(d.x, z, hand, allowed)
            # the modes of one sweep have similar widths: a weak line's start
            # width is held to 5x the strongest one's (its half-height walk ran
            # into the background, and from ~140 mT wide the fit carried the
            # 16 GHz YIG PSSW 4 off to another line)
            if w_first is None:
                w_first = vals["hwhm"]
            else:
                vals["hwhm"] = min(vals["hwhm"], 5 * w_first)
            specs[f"p{k}_center"] = Spec(vals["center"], min=d.x[0] - 0.25 * span,
                                         max=d.x[-1] + 0.25 * span)
            # narrower than the point spacing cannot be resolved; letting the
            # width go to 0 turns a failed fit into a spike on one point
            step = float(np.median(np.diff(d.x)))
            specs[f"p{k}_hwhm"] = Spec(max(vals["hwhm"], step), min=0.25 * step, max=span)
            specs[f"p{k}_amp"] = Spec(vals["amp"], min=0.0)
            specs[f"p{k}_phase"] = Spec(vals["phase"])
        # the next peak is looked for in what this one leaves, away from it
        z = z - vals["amp"] * np.exp(1j * np.deg2rad(vals["phase"])) * chi(
            d.x, vals["center"], vals["hwhm"], hand, setup.lineshape)
        allowed &= np.abs(d.x - vals["center"]) > 4 * max(vals["hwhm"], 1e-12)
    return specs


def _delay_from_phase(d: Data) -> float:
    """tau from the phase slope at the two ends of the range (1/x units)."""
    n = d.x.size
    m = max(3, n // 10)
    edge = np.r_[np.arange(m), np.arange(n - m, n)]
    ph = np.unwrap(np.angle(d.y))
    return float(-np.polyfit(d.x[edge] - d.xc, ph[edge], 1)[0] / (2 * np.pi))


def _guess_dd_from_raw(x, y, setup: Setup, hand: int, keep) -> dict[str, Spec]:
    """Start values for derivative-divide WITH the delay, from the raw sweep:
    the delay unwound there (phase slope), the peak guessed relative to the
    background there. In D itself the peaks are rotated by e^{-+i 2pi tau h},
    2 rad for 3.2 ns and 100 MHz steps, and integrating D back gives nothing
    useful to guess from (tried first, 2026-09-29)."""
    raw = Setup(**{**setup.__dict__, "dd": 0, "delay": True, "baseline": "linear"})
    g = guess(x, y, raw, hand=hand, keep=keep)
    c = g["bg_re"].value + 1j * g["bg_im"].value
    c = c if abs(c) > 0 else 1.0
    s1 = g["slope_re"].value + 1j * g["slope_im"].value
    lnb = s1 / c                              # what is left of d ln B / dx
    specs = {"bg_re": Spec(float(lnb.real)), "bg_im": Spec(float(lnb.imag)),
             "delay": g["delay"]}
    if setup.baseline == "linear":
        specs["slope_re"], specs["slope_im"] = Spec(0.0), Spec(0.0)
    for k in range(1, setup.n_peaks + 1):
        names = peak_names(k)
        if keep and all(nm in keep for nm in names):
            specs.update({nm: keep[nm] for nm in names})
            continue
        specs[f"p{k}_center"], specs[f"p{k}_hwhm"] = g[f"p{k}_center"], g[f"p{k}_hwhm"]
        specs[f"p{k}_amp"] = Spec(g[f"p{k}_amp"].value / abs(c), min=0.0)
        specs[f"p{k}_phase"] = Spec(_wrap(g[f"p{k}_phase"].value - np.rad2deg(np.angle(c))))
    return specs


def _guess_dd(d: Data, setup: Setup, hand: int, keep, tau=None) -> dict[str, Spec]:
    """Start values for derivative-divided data: D is d ln S / dx, so its
    integral gives S back up to a constant (the background, if it varies
    slowly). Guess on that, then make the amplitude relative and the phase
    relative to the background. The delay's part of D, known exactly for a
    given tau, is taken out first."""
    y = d.y
    if tau is not None:
        y = y - ((np.exp(-2j * np.pi * tau * (d.xp - d.x))
                  - np.exp(-2j * np.pi * tau * (d.xm - d.x))) / (d.xp - d.xm))
        d = Data(x=d.x, y=y, xc=d.xc, xp=d.xp, xm=d.xm, dd_half=d.dd_half)
    # The background's own d ln B / dx first, from the ends of the range: for a
    # VNA it is dominated by the delay, -i 2 pi tau (-20i per GHz for 3.2 ns),
    # far larger than the resonance. Integrated along, it would wind the phase
    # back up and hide the peak from the guess (it did, 2026-09-29).
    n = d.x.size
    m = max(3, n // 10)
    edge = np.r_[np.arange(m), np.arange(n - m, n)]
    cols = [np.ones(n)] + ([d.x - d.xc] if setup.baseline == "linear" else [])
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A[edge], d.y[edge], rcond=None)
    rest = d.y - A @ coef
    step = np.diff(d.x)
    L = np.concatenate([[0.0], np.cumsum(0.5 * (rest[1:] + rest[:-1]) * step)])
    s = np.exp(L)
    plain = Setup(**{**setup.__dict__, "dd": 0, "delay": False, "baseline": "linear",
                     "xmin": None, "xmax": None})
    g = guess(d.x, s, plain, hand=hand, keep=keep)
    c = g["bg_re"].value + 1j * g["bg_im"].value
    specs = {"bg_re": Spec(float(coef[0].real)), "bg_im": Spec(float(coef[0].imag))}
    if setup.baseline == "linear":
        specs["slope_re"] = Spec(float(coef[1].real))
        specs["slope_im"] = Spec(float(coef[1].imag))
    for k in range(1, setup.n_peaks + 1):
        names = peak_names(k)
        if keep and all(nm in keep for nm in names):
            specs.update({nm: keep[nm] for nm in names})
            continue
        specs[f"p{k}_center"], specs[f"p{k}_hwhm"] = g[f"p{k}_center"], g[f"p{k}_hwhm"]
        amp = g[f"p{k}_amp"].value / max(abs(c), 1e-300)
        specs[f"p{k}_amp"] = Spec(amp, min=0.0)
        specs[f"p{k}_phase"] = Spec(_wrap(g[f"p{k}_phase"].value - np.rad2deg(np.angle(c))))
    if tau is not None:
        specs["delay"] = Spec(float(tau))
    return specs


# ─────────────────────────────────── the fit ──────────────────────────────────

@dataclass
class Result:
    setup: Setup
    hand: int
    xc: float
    values: dict[str, float]
    errors: dict[str, float | None]     # 1 sigma; None = fixed, or not estimable
    vary: dict[str, bool]
    chisqr: float
    redchi: float
    ndata: int
    nvarys: int
    success: bool
    message: str
    xrange: tuple[float, float]
    dd_half: float | None = None

    def evaluate(self, x) -> np.ndarray:
        return evaluate(self.values, x, self.setup, self.hand, self.xc, dd_half=self.dd_half)

    def fwhm(self, k: int) -> tuple[float, float | None]:
        e = self.errors.get(f"p{k}_hwhm")
        return 2 * self.values[f"p{k}_hwhm"], None if e is None else 2 * e

    def specs(self) -> dict[str, Spec]:
        """The fitted values as start values: 'fit again from here'."""
        return {n: Spec(v, self.vary.get(n, True)) for n, v in self.values.items()}


def _run(d: Data, setup: Setup, hand: int, start: dict[str, Spec]) -> Result:
    import lmfit
    params = lmfit.Parameters()
    for name in param_names(setup):
        sp = start[name]
        v = float(np.clip(sp.value, sp.min, sp.max))
        params.add(name, value=v, vary=sp.vary, min=sp.min, max=sp.max)

    def resid(p):
        diff = evaluate(p.valuesdict(), d.x, setup, hand, d.xc, d.xp, d.xm) - d.y
        return np.concatenate([diff.real, diff.imag]) if setup.mode == "complex" else diff

    out = lmfit.minimize(resid, params, method="leastsq")
    values = {n: float(p.value) for n, p in out.params.items()}
    errors = {n: (float(p.stderr) if (p.vary and p.stderr is not None
                                      and np.isfinite(p.stderr)) else None)
              for n, p in out.params.items()}
    for k in range(1, setup.n_peaks + 1):
        values[f"p{k}_phase"] = _wrap(values[f"p{k}_phase"])
    return Result(setup=setup, hand=hand, xc=d.xc, values=values, errors=errors,
                  vary={n: p.vary for n, p in out.params.items()},
                  chisqr=float(out.chisqr), redchi=float(out.redchi),
                  ndata=int(out.ndata), nvarys=int(out.nvarys),
                  success=bool(out.success), message=str(out.message),
                  xrange=(float(d.x[0]), float(d.x[-1])), dd_half=d.dd_half)


def fit(x, y, setup: Setup, start: dict[str, Spec] | None = None) -> Result:
    """Fit, and return the best of the candidates tried.

    Candidates: in complex mode with hand = 0, both handednesses. With no
    `start`, the start values are guessed; in real mode from both mirror images
    of the Hilbert-transformed channel (the guess cannot tell them apart; the
    fit can). The one with the smallest chi-square wins.
    """
    d = prepare(x, y, setup)
    runs = []
    if setup.mode == "complex":
        hands = [setup.hand] if setup.hand else [1, -1]
        if not setup.hand and not start and setup.n_peaks > 1:
            # several peaks: decide the hand on the strongest one alone, then fit
            # everything once. The wrong hand's full fit starts from junk and
            # wandered for ~30 s on a 10 000-point YIG sweep before losing.
            picked = _pick_hand(x, y, setup)
            hands = [picked] if picked else hands
        for h in hands:
            if not start:
                runs.append(_run(d, setup, h, guess(x, y, setup, hand=h)))
                continue
            runs.append(_run(d, setup, h, start))
            if not setup.hand:
                # the operator's start was made for ONE hand, and nobody knows
                # which: for the mirror image, turn every peak by 180 deg, which
                # keeps the value on resonance (chi -> conj(chi): +i -> -i)
                runs.append(_run(d, setup, h, _turned(start, setup)))
        if start and not setup.hand:
            # and a fresh guess as a safety net: a poor start must not give a
            # worse result than no start (a start mixing both hands ended at
            # chi2 221 where the guess reaches 27, 2026-09-29). Parameters the
            # operator FIXED stay fixed.
            h = _pick_hand(x, y, setup) or 1
            fresh = guess(x, y, setup, hand=h)
            fresh.update({n: sp for n, sp in start.items() if n in fresh and not sp.vary})
            runs.append(_run(d, setup, h, fresh))
    else:
        starts = [start] if start else []
        if not start:
            starts.append(guess(x, y, setup, hand=1))
            mirror = guess(x, y, setup, hand=-1)
            # Re(A e^{i phi} conj(chi)) = Re(A e^{-i phi} chi): same model, phase negated
            for k in range(1, setup.n_peaks + 1):
                sp = mirror[f"p{k}_phase"]
                mirror[f"p{k}_phase"] = Spec(-sp.value, sp.vary, -sp.max, -sp.min)
            starts.append(mirror)
        runs = [_run(d, setup, 1, st) for st in starts]
    return min(runs, key=lambda r: r.chisqr if np.isfinite(r.chisqr) else np.inf)


# ─────────────────────────── a range that follows the peak ────────────────────

def locate(x, y, setup: Setup) -> tuple[float, float]:
    """Where the main resonance is and how wide: (x0, HWHM) from a quick
    one-peak fit over the WHOLE sweep. The same measure for every curve, so a
    range drawn on one curve can be carried to the others (follow_range)."""
    quick = Setup(**{**setup.__dict__, "n_peaks": 1, "xmin": None, "xmax": None})
    r = fit(x, y, quick)
    return r.values["p1_center"], r.values["p1_hwhm"]


def range_in_widths(x0: float, hwhm: float, xmin: float, xmax: float) -> tuple[float, float]:
    """A range as seen from the peak: (xmin - x0) / HWHM, (xmax - x0) / HWHM."""
    return (xmin - x0) / hwhm, (xmax - x0) / hwhm


def follow_range(x, y, setup: Setup, lo: float, hi: float) -> tuple[float, float]:
    """The range x0 + lo*HWHM ... x0 + hi*HWHM on THIS curve, inside its data.

    Why in widths and not in field: the resonance moves with frequency (44 mT
    at 6 GHz, 158 mT at 12 GHz for the demo film) and gets broader; a window of
    fixed field values drawn at 6 GHz would miss the 12 GHz peak entirely.
    """
    x0, w = locate(x, y, setup)
    xs = np.asarray(x, dtype=float)
    xs = xs[np.isfinite(xs)]
    return max(x0 + lo * w, float(xs.min())), min(x0 + hi * w, float(xs.max()))


def _pick_hand(x, y, setup: Setup) -> int:
    """The handedness from a quick one-peak fit, both ways, in a window of ten
    widths around the strongest line; 0 if that cannot be done."""
    try:
        one = Setup(**{**setup.__dict__, "n_peaks": 1, "hand": 0})
        g = guess(x, y, one, hand=1)
        x0, w = g["p1_center"].value, g["p1_hwhm"].value
        lo = x0 - 10 * w if setup.xmin is None else max(setup.xmin, x0 - 10 * w)
        hi = x0 + 10 * w if setup.xmax is None else min(setup.xmax, x0 + 10 * w)
        win = Setup(**{**one.__dict__, "xmin": lo, "xmax": hi})
        dw = prepare(x, y, win)
        runs = [(_run(dw, win, h, guess(x, y, win, hand=h)), h) for h in (1, -1)]
        return min(runs, key=lambda rh: rh[0].chisqr)[1]
    except Exception:
        return 0


def _turned(start: dict[str, Spec], setup: Setup) -> dict[str, Spec]:
    out = dict(start)
    for k in range(1, setup.n_peaks + 1):
        sp = start[f"p{k}_phase"]
        out[f"p{k}_phase"] = Spec(_wrap(sp.value + 180.0), sp.vary, sp.min, sp.max)
    return out


def suspicious(res: Result) -> str:
    """Why a converged fit should still be looked at, or "" -- a peak whose
    width or amplitude is smaller than its own error bar was not found."""
    if not res.success:
        return "did not converge"
    for k in range(1, res.setup.n_peaks + 1):
        for p in ("hwhm", "amp"):
            v, e = res.values[f"p{k}_{p}"], res.errors.get(f"p{k}_{p}")
            if res.vary.get(f"p{k}_{p}") and (e is None or e > abs(v)):
                return f"peak {k}: {p} not determined (error {'?' if e is None else f'{e:.2g}'})"
    return ""


def residuals(res: Result, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Data minus model inside the fit range: (x, residual) -- complex in complex mode."""
    d = prepare(x, y, Setup(**{**res.setup.__dict__, "xmin": res.xrange[0],
                               "xmax": res.xrange[1]}))
    return d.x, d.y - evaluate(res.values, d.x, res.setup, res.hand, res.xc, d.xp, d.xm)


# ──────────────────────────────── the results ─────────────────────────────────

@dataclass
class Column:
    name: str
    unit: str
    comment: str
    values: list = field(default_factory=list)
    kind: str = "Y"                     # Origin designation: X, Y, E (error of the Y before), L


def results_table(rows: list) -> list[Column]:
    """Fitted curves -> columns, one row per curve.

    rows: (Curve, Result) pairs. The coordinates the curves were held at come
    first (the natural X: "linewidth vs frequency"); then, per peak, centre,
    HWHM, FWHM, amplitude and phase, each followed by its 1-sigma error; then
    reduced chi-square, handedness, label and file. Curves with fewer peaks, or
    held at other dims, leave cells empty rather than inventing zeros.
    """
    if not rows:
        return []
    c0 = rows[0][0]
    held_dims = []
    for c, _ in rows:
        for d in c.held:
            if d not in held_dims:
                held_dims.append(d)
    cols = []
    for i, dim in enumerate(held_dims):
        unit = next((c.held[dim][1] for c, _ in rows if dim in c.held), "")
        cols.append(Column(dim, unit, "the curve was taken at", kind="X" if i == 0 else "Y",
                           values=[c.held[dim][0] if dim in c.held else np.nan
                                   for c, _ in rows]))
    n_peaks = max(r.setup.n_peaks for _, r in rows)
    x_unit, y_unit = c0.x_unit, c0.y_unit
    for k in range(1, n_peaks + 1):
        for p, label in (("center", "resonance"), ("hwhm", "HWHM"), ("fwhm", "FWHM = 2 HWHM"),
                         ("amp", "amplitude"), ("phase", "mixing phase")):
            vals, errs = [], []
            for _, r in rows:
                if k > r.setup.n_peaks:
                    vals.append(np.nan); errs.append(np.nan)
                    continue
                if p == "fwhm":
                    v, e = r.fwhm(k)
                else:
                    v, e = r.values[f"p{k}_{p}"], r.errors.get(f"p{k}_{p}")
                vals.append(v); errs.append(np.nan if e is None else e)
            unit = param_unit(f"p{k}_{p}", x_unit, y_unit, rows[0][1].setup)
            cols.append(Column(f"p{k}_{p}", unit, f"peak {k}: {label}", vals))
            cols.append(Column(f"p{k}_{p}_err", unit, f"peak {k}: {label}, 1 sigma", errs,
                               kind="E"))
    cols.append(Column("redchi", "", "reduced chi-square", [r.redchi for _, r in rows]))
    cols.append(Column("hand", "", "+1 / -1: which way the signal turns",
                       [r.hand for _, r in rows]))
    cols.append(Column("label", "", "curve", [c.label for c, _ in rows], kind="L"))
    cols.append(Column("source", "", "measurement file",
                       [Path(c.source).name if c.source else "" for c, _ in rows], kind="L"))
    if not held_dims and cols:
        cols.insert(0, Column("index", "", "curve number", list(range(1, len(rows) + 1)),
                              kind="X"))
    return cols


def write_results(path, columns: list[Column]) -> Path:
    """Columns as text, with the three header rows AaltoView's exports use
    (Long Name / Units / Comments -- Origin's import recognises them)."""
    from aaltoview.export import _delimiter, _encoding
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
            w.writerow([cell(c.values[i]) for c in columns])
    return path


def figure_fit(curve, res: Result, y, size=(6.4, 5.6)):
    """Data, fit and residuals, publication style (white, like AaltoView's figures)."""
    from aaltoview.export import _figure, axis_title
    fig = _figure(size)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    xx = np.linspace(res.xrange[0], res.xrange[1], 800)
    model = res.evaluate(xx)
    rx, rr = residuals(res, curve.x, y)
    shown = displayed(curve.x, y, res.setup)        # derivative-divided when that is on
    un = unwinder(res.setup, res.values, res.xc)    # the delay taken out, when fitted
    if un is not None:
        shown = Data(x=shown.x, y=shown.y * un(shown.x), xc=shown.xc)
        model, rr = model * un(xx), rr * un(rx)
    if res.setup.mode == "complex":
        for part, fn, col in (("Re", np.real, "#1f77b4"), ("Im", np.imag, "#d62728")):
            ax.plot(shown.x, fn(shown.y), "o", ms=2.5, color=col, alpha=0.6,
                    label=f"{part} data")
            ax.plot(xx, fn(model), "-", color=col, lw=1.5, label=f"{part} fit")
            axr.plot(rx, fn(rr), "o", ms=2, color=col)
    else:
        ax.plot(shown.x, shown.y, "o", ms=2.5, color="#1f77b4", alpha=0.6, label="data")
        ax.plot(xx, model, "-", color="#d62728", lw=1.5, label="fit")
        axr.plot(rx, rr, "o", ms=2, color="#1f77b4")
    axr.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel(y_title(curve, res.setup, unwound=un is not None))
    axr.set_ylabel("residual")
    axr.set_xlabel(axis_title(curve.x_name, curve.x_unit))
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=8, frameon=False)
    bits = []
    for k in range(1, res.setup.n_peaks + 1):
        c, ce = res.values[f"p{k}_center"], res.errors.get(f"p{k}_center")
        w, we = res.fwhm(k)
        bits.append(f"x0 = {fmt_pm(c, ce)}, FWHM = {fmt_pm(w, we)} {curve.x_unit}")
    ax.set_title(curve.label + "\n" + "; ".join(bits), fontsize=9)
    return fig


def y_title(curve, setup: Setup, unwound: bool = False) -> str:
    """The y axis: the channel, S (with the delay taken out, if so drawn), or
    the derivative-divided d ln S / dx."""
    from aaltoview.export import axis_title
    if setup.dd:
        return axis_title("∂_D S = ΔS / (Δx S)", f"1/{curve.x_unit}" if curve.x_unit else "")
    if unwound:
        return axis_title("S · e^{+i2πτ(x − xc)}", curve.y_unit)
    return axis_title(curve.y_name if setup.mode == "real" else "S", curve.y_unit)


def unwinder(setup: Setup, values: dict | None, xc: float | None):
    """x -> e^{+i 2 pi tau (x - xc)}, the cable's phase taken back out, for
    DRAWING a fit with the delay option (None when there is nothing to unwind).
    Raw, a 3.2 ns delay spins Re and Im through 60 turns over 1-20 GHz and the
    data look like noise; unwound, the resonance is what is seen."""
    if (setup.mode != "complex" or not setup.delay or setup.dd or not values
            or "delay" not in values or xc is None):
        return None
    tau = values["delay"]
    return lambda x: np.exp(2j * np.pi * tau * (np.asarray(x, dtype=float) - xc))


def phase_turns(x, z) -> float:
    """How many full turns the phase of z makes over the sweep (0 for real data)."""
    z = np.asarray(z)
    if not np.iscomplexobj(z):
        return 0.0
    ok = np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(np.abs(z))
    if ok.sum() < 3:
        return 0.0
    order = np.argsort(np.asarray(x, dtype=float)[ok])
    ph = np.unwrap(np.angle(z[ok][order]))
    return float(abs(ph[-1] - ph[0]) / (2 * np.pi))


def fmt_pm(v: float, e: float | None) -> str:
    """'12.34(5)'-style is compact but opaque; '12.345 ± 0.052' reads anywhere.
    Two significant digits of the error set the digits of the value."""
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.6g}"
    digits = max(0, 1 - int(np.floor(np.log10(e))))
    return f"{v:.{digits}f} ± {e:.{digits}f}"
