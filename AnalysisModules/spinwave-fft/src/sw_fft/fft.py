"""fft.py -- spatial FFT of spin-wave lines, and the peaks in it. No Qt.

A line is a signal along a position axis x (TR-MOKE or BLS scanned along a
waveguide, a VNA/ lock-in map pos_x x rf_freq): v(x). Its spatial spectrum
shows the wavevector k of the spin wave that was excited at that frequency.

    F(k) = sum_j w_j (v_j - trend_j) exp(-i k x_j) / sum_j w_j

* COHERENT on complex data (a lock-in's X + iY): a wave exp(+i k0 x) -- moving
  towards +x with the convention exp(i(kx - wt)) -- peaks at +k0 only, a wave
  moving the other way at -k0. On a REAL signal (Re z, a Kerr trace) the two
  are mirror images and the sign of k says nothing.
* NORMALISED by the sum of the window, so a pure wave of amplitude A gives
  |F(k0)| = A whatever the window, the length or the padding -- spectra of
  lines of different lengths can be compared.
* Zero-padding interpolates the spectrum (smoother peaks, a better peak
  position from the refinement); it does NOT make two close peaks separable:
  the resolution is 2 pi / L, L = the length scanned.
* Uneven x (a scan with a hiccup) is interpolated onto an even grid; NaN holes
  (an aborted or running scan) are interpolated across, a line with fewer
  than MIN_POINTS real points gives an all-NaN spectrum.

Units: x in any length unit the file names (nm, um, mm, m); k comes out in
rad/um (k = 2 pi / lambda) or 1/um (1 / lambda), K_UNITS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOWS = ("none", "hann", "hamming", "blackman", "tukey")
DETRENDS = ("none", "mean", "linear")
K_UNITS = ("rad/um", "1/um")
PARTS = ("complex", "real", "imag", "abs")
MIN_POINTS = 4

#: a length unit -> micrometres
_LENGTH = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0,
           "nm": 1e-3, "pm": 1e-6}


def length_scale(unit: str) -> float | None:
    """Micrometres per unit of x; None when the unit is not a length."""
    return _LENGTH.get((unit or "").strip())


@dataclass
class Settings:
    window: str = "hann"
    tukey_alpha: float = 0.5          # fraction of the line that is tapered
    detrend: str = "mean"
    pad: int = 4                      # zero-padding: FFT length >= pad x points (power of 2)
    k_unit: str = "rad/um"
    part: str = "complex"             # what is transformed (complex = both quadratures)
    x_scale: float | None = None      # um per x unit; None = from the unit (1 if unknown)


def window(name: str, n: int, alpha: float = 0.5) -> np.ndarray:
    if n < 2 or name == "none":
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    if name == "tukey":
        a = float(np.clip(alpha, 0.0, 1.0))
        w = np.ones(n)
        if a <= 0:
            return w
        t = np.linspace(0.0, 1.0, n)
        edge = a / 2
        lo, hi = t < edge, t > 1 - edge
        w[lo] = 0.5 * (1 + np.cos(np.pi * (2 * t[lo] / a - 1)))
        w[hi] = 0.5 * (1 + np.cos(np.pi * (2 * t[hi] / a - 2 / a + 1)))
        return w
    raise ValueError(f"unknown window '{name}' (have: {', '.join(WINDOWS)})")


def _part(v: np.ndarray, part: str) -> np.ndarray:
    v = np.asarray(v)
    if part == "complex":
        return v.astype(complex)
    if part == "real":
        return np.real(v).astype(float)
    if part == "imag":
        return np.imag(v).astype(float)
    if part == "abs":
        return np.abs(v).astype(float)
    raise ValueError(f"unknown part '{part}' (have: {', '.join(PARTS)})")


def even_grid(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(x, v) sorted, holes and uneven steps interpolated onto an even grid over
    the measured span. None when fewer than MIN_POINTS real points."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v)
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() < MIN_POINTS:
        return None
    xs, vs = x[ok], v[ok]
    order = np.argsort(xs)
    xs, vs = xs[order], vs[order]
    keep = np.r_[True, np.diff(xs) > 0]            # a repeated position: the first
    xs, vs = xs[keep], vs[keep]
    if xs.size < MIN_POINTS:
        return None
    n = x.size if ok.all() else int(round((xs[-1] - xs[0]) / np.median(np.diff(xs)))) + 1
    grid = np.linspace(xs[0], xs[-1], max(n, MIN_POINTS))
    if xs.size == grid.size and np.allclose(xs, grid, rtol=0, atol=1e-9 * abs(grid[-1] - grid[0])):
        return grid, vs
    if np.iscomplexobj(vs):
        return grid, np.interp(grid, xs, vs.real) + 1j * np.interp(grid, xs, vs.imag)
    return grid, np.interp(grid, xs, vs)


def _detrend(x: np.ndarray, v: np.ndarray, how: str) -> np.ndarray:
    if how == "none":
        return v
    if how == "mean":
        return v - v.mean()
    if how == "linear":
        A = np.vstack([np.ones_like(x), x - x.mean()]).T
        if np.iscomplexobj(v):
            cr = np.linalg.lstsq(A, v.real, rcond=None)[0]
            ci = np.linalg.lstsq(A, v.imag, rcond=None)[0]
            return v - A @ (cr + 1j * ci)
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    raise ValueError(f"unknown detrend '{how}' (have: {', '.join(DETRENDS)})")


def n_fft(n: int, pad: int) -> int:
    return int(2 ** np.ceil(np.log2(max(n * max(int(pad), 1), 2))))


def k_axis(n: int, dx_um: float, k_unit: str) -> np.ndarray:
    """The (shifted) wavevector axis of an n-point FFT of step dx_um."""
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dx_um))     # cycles per um
    if k_unit == "rad/um":
        return 2 * np.pi * f
    if k_unit == "1/um":
        return f
    raise ValueError(f"unknown k unit '{k_unit}' (have: {', '.join(K_UNITS)})")


def x_scale_of(unit: str, s: Settings) -> float:
    if s.x_scale is not None:
        return float(s.x_scale)
    sc = length_scale(unit)
    return 1.0 if sc is None else sc


def line_spectrum(x, v, s: Settings, x_unit: str = "um", nfft: int | None = None):
    """(k, F): the complex spectrum of one line, k in s.k_unit, F normalised to
    the amplitude of a pure wave. Both all-NaN when the line has no data."""
    x = np.asarray(x, dtype=float) * x_scale_of(x_unit, s)
    v = _part(v, s.part)
    g = even_grid(x, v)
    if g is None:
        n = nfft or n_fft(max(np.asarray(x).size, MIN_POINTS), s.pad)
        return np.full(n, np.nan), np.full(n, np.nan + 0j)
    xg, vg = g
    dx = (xg[-1] - xg[0]) / (xg.size - 1)
    vg = _detrend(xg, vg, s.detrend)
    w = window(s.window, xg.size, s.tukey_alpha)
    n = nfft or n_fft(xg.size, s.pad)
    # the phase refers to x = xg[0]: exp(-i k (x - x0)); |F| does not care
    F = np.fft.fftshift(np.fft.fft(vg * w, n)) / w.sum()
    return k_axis(n, dx, s.k_unit), F


@dataclass
class Spectrum:
    """The FFT of every line of an input: F[i, :] is line i, over k."""
    k: np.ndarray
    F: np.ndarray                     # (n_lines, n_k) complex
    k_unit: str
    settings: Settings
    resolution: float                 # 2 pi / L (or 1 / L): what can be told apart

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.F)


def spectra(x, rows, s: Settings, x_unit: str = "um") -> Spectrum:
    """Every line of a map on ONE k axis: the longest even grid decides dx and
    the FFT length; lines are sampled alike in a map, so this is the norm."""
    x = np.asarray(x, dtype=float)
    rows = np.atleast_2d(np.asarray(rows))
    sc = x_scale_of(x_unit, s)
    xs = np.sort(x[np.isfinite(x)]) * sc
    if xs.size < MIN_POINTS:
        raise ValueError("fewer than 4 positions: nothing to transform")
    span = xs[-1] - xs[0]
    if span <= 0:
        raise ValueError("all positions are the same: nothing to transform")
    nfft = n_fft(x.size, s.pad)
    ks, out = None, []
    for r in rows:
        k, F = line_spectrum(x, r, s, x_unit, nfft=nfft)
        if ks is None and np.isfinite(k).all():
            ks = k
        out.append(F)
    if ks is None:
        ks = k_axis(nfft, span / (x.size - 1), s.k_unit)
    full = 2 * np.pi if s.k_unit == "rad/um" else 1.0
    return Spectrum(k=ks, F=np.array(out), k_unit=s.k_unit, settings=s,
                    resolution=full / span)


# ─────────────────────────────────── peaks ────────────────────────────────────

@dataclass
class PeakSettings:
    n_peaks: int = 1
    side: str = "both"                # "both", "positive", "negative"
    k_min: float = 0.0                # |k| >= k_min (leave out the DC remainder)
    k_max: float = np.inf
    min_rel: float = 0.1              # a peak must reach this fraction of the line's highest


SIDES = ("both", "positive", "negative")


@dataclass
class Peak:
    line: int
    k: float                          # refined position, signed
    amplitude: float                  # |F| at the peak (refined)
    fwhm: float                       # width in k (NaN if not measurable)


def _refine(k: np.ndarray, m: np.ndarray, i: int) -> tuple[float, float]:
    """Sub-bin position and height: a parabola through log|F| at i-1, i, i+1
    (exact for a Gaussian peak, within a few % of a bin for the Hann window's)."""
    if i <= 0 or i >= m.size - 1 or not np.all(m[i - 1:i + 2] > 0):
        return float(k[i]), float(m[i])
    a, b, c = np.log(m[i - 1:i + 2])
    den = a - 2 * b + c
    if den >= 0:
        return float(k[i]), float(m[i])
    p = 0.5 * (a - c) / den
    return float(k[i] + p * (k[1] - k[0])), float(np.exp(b - 0.25 * (a - c) * p))


def _fwhm(k: np.ndarray, m: np.ndarray, i: int) -> float:
    half = m[i] / 2
    lo = i
    while lo > 0 and m[lo] > half:
        lo -= 1
    hi = i
    while hi < m.size - 1 and m[hi] > half:
        hi += 1
    if m[lo] > half or m[hi] > half:
        return float("nan")

    def cross(j0, j1):          # linear between the two samples around half
        return k[j0] + (half - m[j0]) * (k[j1] - k[j0]) / (m[j1] - m[j0])

    return float(cross(hi - 1, hi) - cross(lo + 1, lo))


def find_peaks(k: np.ndarray, mag: np.ndarray, ps: PeakSettings, line: int = 0) -> list[Peak]:
    """The strongest local maxima of one |F| line inside the allowed k range,
    strongest first."""
    k = np.asarray(k, dtype=float)
    m = np.asarray(mag, dtype=float)
    if not np.isfinite(m).any():
        return []
    allowed = np.isfinite(m) & (np.abs(k) >= ps.k_min) & (np.abs(k) <= ps.k_max)
    if ps.side == "positive":
        allowed &= k > 0
    elif ps.side == "negative":
        allowed &= k < 0
    if not allowed.any():
        return []
    mm = np.where(allowed, m, -np.inf)
    top = np.nanmax(mm)
    inner = np.arange(1, m.size - 1)
    is_max = (mm[inner] >= mm[inner - 1]) & (mm[inner] > mm[inner + 1]) & allowed[inner]
    cand = inner[is_max & (mm[inner] >= ps.min_rel * top)]
    cand = cand[np.argsort(-m[cand])][:max(int(ps.n_peaks), 0)]
    out = []
    for i in cand:
        kk, amp = _refine(k, m, i)
        out.append(Peak(line=line, k=kk, amplitude=amp, fwhm=_fwhm(k, m, i)))
    return out


def all_peaks(sp: Spectrum, ps: PeakSettings) -> list[Peak]:
    out = []
    mag = sp.magnitude
    for i in range(mag.shape[0]):
        out.extend(find_peaks(sp.k, mag[i], ps, line=i))
    return out

