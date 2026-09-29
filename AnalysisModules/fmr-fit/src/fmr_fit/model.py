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
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODES = ("complex", "real")
BASELINES = ("constant", "linear")
PEAK_PARAMS = ("center", "hwhm", "amp", "phase")


def chi(x, center, hwhm, hand: int = 1) -> np.ndarray:
    """The complex Lorentzian susceptibility; hand = -1 is its mirror image."""
    c = hwhm / (center - np.asarray(x, dtype=float) - 1j * hwhm)
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
    else:
        names += ["bg"] + (["slope"] if setup.baseline == "linear" else [])
    return names


def param_unit(name: str, x_unit: str, y_unit: str) -> str:
    """The unit a parameter is in, for tables and file headers."""
    if name.endswith(("_center", "_hwhm", "_fwhm")):
        return x_unit
    if name.endswith("_phase"):
        return "deg"
    if name.startswith("slope"):
        return f"{y_unit}/{x_unit}" if (y_unit or x_unit) else ""
    return y_unit                       # amplitude, background


# ──────────────────────────────── the data ────────────────────────────────────

@dataclass
class Data:
    x: np.ndarray
    y: np.ndarray                       # complex (complex mode) or real
    xc: float                           # centre of the range: the baseline's origin


def prepare(x, y, setup: Setup) -> Data:
    """The points that take part: finite, inside the range, sorted by x.
    NaN-aware: a running or aborted scan has holes, and they are left out."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if setup.mode == "complex" and not np.iscomplexobj(y):
        raise ValueError("complex fit needs complex data -- this curve is real; "
                         "use 'one channel'")
    if setup.mode == "real" and np.iscomplexobj(y):
        raise ValueError("'one channel' fits real data; pass Re, Im or |z|")
    ok = np.isfinite(x) & np.isfinite(np.abs(y))
    if setup.xmin is not None:
        ok &= x >= setup.xmin
    if setup.xmax is not None:
        ok &= x <= setup.xmax
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    n_free = len(param_names(setup)) * (1 if setup.mode == "real" else 0.5)
    if x.size < max(5, n_free + 2):
        raise ValueError(f"only {x.size} measured points in the fit range -- too few")
    return Data(x=x, y=y, xc=0.5 * (x[0] + x[-1]))


def evaluate(values: dict, x, setup: Setup, hand: int, xc: float) -> np.ndarray:
    """The model at x: complex in complex mode, real (= Re S) in real mode."""
    x = np.asarray(x, dtype=float)
    s = np.zeros(x.shape, dtype=complex)
    for k in range(1, setup.n_peaks + 1):
        s += (values[f"p{k}_amp"] * np.exp(1j * np.deg2rad(values[f"p{k}_phase"]))
              * chi(x, values[f"p{k}_center"], values[f"p{k}_hwhm"], hand))
    if setup.mode == "complex":
        s += values["bg_re"] + 1j * values["bg_im"]
        if setup.baseline == "linear":
            s += (values["slope_re"] + 1j * values["slope_im"]) * (x - xc)
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


def _guess_peak(x, z, hand: int) -> dict[str, float]:
    """Largest bump of |z|: its position, its width at 1/sqrt(2) of the top
    (|chi| = 1/sqrt(2) at x0 +- dx), its height and its phase."""
    mag = _smooth(np.abs(z))
    i = int(np.argmax(mag))
    top = mag[i]
    lo = i
    while lo > 0 and mag[lo] > top / np.sqrt(2):
        lo -= 1
    hi = i
    while hi < x.size - 1 and mag[hi] > top / np.sqrt(2):
        hi += 1
    step = np.median(np.diff(x)) if x.size > 1 else 1.0
    w = max(0.5 * (x[hi] - x[lo]), step)
    # on resonance chi = +i (hand +1) or -i (hand -1)
    phase = np.rad2deg(np.angle(z[i]) - hand * np.pi / 2)
    return {"center": float(x[i]), "hwhm": float(w), "amp": float(top),
            "phase": float(_wrap(phase))}


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
    n = d.x.size
    m = max(3, n // 10)
    edge = np.r_[np.arange(m), np.arange(n - m, n)]
    cols = [np.ones(n)] + ([d.x - d.xc] if setup.baseline == "linear" else [])
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A[edge], d.y[edge], rcond=None)
    bg = A @ coef
    specs: dict[str, Spec] = {}
    if setup.mode == "complex":
        specs["bg_re"], specs["bg_im"] = Spec(float(coef[0].real)), Spec(float(coef[0].imag))
        if setup.baseline == "linear":
            specs["slope_re"] = Spec(float(coef[1].real))
            specs["slope_im"] = Spec(float(coef[1].imag))
    else:
        specs["bg"] = Spec(float(coef[0].real))
        if setup.baseline == "linear":
            specs["slope"] = Spec(float(coef[1].real))

    r = d.y - bg
    z = r if setup.mode == "complex" else _analytic(r)
    span = d.x[-1] - d.x[0]
    for k in range(1, setup.n_peaks + 1):
        names = peak_names(k)
        if keep and all(nm in keep for nm in names):
            vals = {p: keep[f"p{k}_{p}"].value for p in PEAK_PARAMS}
            specs.update({nm: keep[nm] for nm in names})
        else:
            vals = _guess_peak(d.x, z, hand)
            specs[f"p{k}_center"] = Spec(vals["center"], min=d.x[0] - 0.25 * span,
                                         max=d.x[-1] + 0.25 * span)
            # narrower than the point spacing cannot be resolved; letting the
            # width go to 0 turns a failed fit into a spike on one point
            step = float(np.median(np.diff(d.x)))
            specs[f"p{k}_hwhm"] = Spec(max(vals["hwhm"], step), min=0.25 * step, max=span)
            specs[f"p{k}_amp"] = Spec(vals["amp"], min=0.0)
            specs[f"p{k}_phase"] = Spec(vals["phase"])
        # the next peak is looked for in what this one leaves
        z = z - vals["amp"] * np.exp(1j * np.deg2rad(vals["phase"])) * chi(
            d.x, vals["center"], vals["hwhm"], hand)
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

    def evaluate(self, x) -> np.ndarray:
        return evaluate(self.values, x, self.setup, self.hand, self.xc)

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
        diff = evaluate(p.valuesdict(), d.x, setup, hand, d.xc) - d.y
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
                  xrange=(float(d.x[0]), float(d.x[-1])))


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
        for h in ([setup.hand] if setup.hand else [1, -1]):
            if not start:
                runs.append(_run(d, setup, h, guess(d.x, d.y, setup, hand=h)))
                continue
            runs.append(_run(d, setup, h, start))
            if not setup.hand:
                # the operator's start was made for ONE hand, and nobody knows
                # which: for the mirror image, turn every peak by 180 deg, which
                # keeps the value on resonance (chi -> conj(chi): +i -> -i)
                runs.append(_run(d, setup, h, _turned(start, setup)))
    else:
        starts = [start] if start else []
        if not start:
            starts.append(guess(d.x, d.y, setup, hand=1))
            mirror = guess(d.x, d.y, setup, hand=-1)
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
    return d.x, d.y - res.evaluate(d.x)


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
            unit = param_unit(f"p{k}_{p}", x_unit, y_unit)
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
    if res.setup.mode == "complex":
        for part, fn, col in (("Re", np.real, "#1f77b4"), ("Im", np.imag, "#d62728")):
            ax.plot(curve.x, fn(y), "o", ms=2.5, color=col, alpha=0.6, label=f"{part} data")
            ax.plot(xx, fn(model), "-", color=col, lw=1.5, label=f"{part} fit")
            axr.plot(rx, fn(rr), "o", ms=2, color=col)
    else:
        ax.plot(curve.x, y, "o", ms=2.5, color="#1f77b4", alpha=0.6, label="data")
        ax.plot(xx, model, "-", color="#d62728", lw=1.5, label="fit")
        axr.plot(rx, rr, "o", ms=2, color="#1f77b4")
    axr.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel(axis_title(curve.y_name if res.setup.mode == "real" else "S", curve.y_unit))
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


def fmt_pm(v: float, e: float | None) -> str:
    """'12.34(5)'-style is compact but opaque; '12.345 ± 0.052' reads anywhere.
    Two significant digits of the error set the digits of the value."""
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.6g}"
    digits = max(0, 1 - int(np.floor(np.log10(e))))
    return f"{v:.{digits}f} ± {e:.{digits}f}"
