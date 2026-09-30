"""The options for frequency sweeps: oscillator lineshape, electrical delay,
reference subtract/divide, derivative-divide.

Each one against a signal made with a KNOWN background -- and, where it
matters, the plain fit on the same data, to show the option is what fixes it.
The demo VNA file (tools/make_demo_data.py, file 7) has everything at once:
delay, standing-wave ripple, sloped loss, the exact oscillator lineshape.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lmfit")

from aaltoview import export as E  # noqa: E402
from aaltoview.data import load  # noqa: E402
from fmr_fit import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_data as DEMO  # noqa: E402

F = np.linspace(1.0, 20.0, 1901)


def _noise(n, sigma, seed=3):
    rng = np.random.default_rng(seed)
    return sigma * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)


def test_the_oscillator_lineshape_is_the_lorentzian_near_resonance():
    x = np.linspace(9.9, 10.1, 5)
    lor = M.chi(x, 10.0, 0.01, 1, "lorentzian")
    osc = M.chi(x, 10.0, 0.01, 1, "oscillator")
    assert osc[2] == pytest.approx(1j) and lor[2] == pytest.approx(1j)
    assert np.allclose(lor, osc, rtol=1e-2)            # off by ~ (x - x0) / x0


def test_a_broad_line_needs_the_oscillator():
    """f0 = 3 GHz, HWHM 0.4 GHz: the Lorentzian is off, the oscillator is not."""
    y = 0.5 + 0.2 * np.exp(0.7j) * M.chi(F, 3.0, 0.4, 1, "oscillator")
    osc = M.fit(F, y, M.Setup(lineshape="oscillator", baseline="constant", xmax=8.0))
    lor = M.fit(F, y, M.Setup(lineshape="lorentzian", baseline="constant", xmax=8.0))
    assert osc.values["p1_center"] == pytest.approx(3.0, abs=1e-5)
    assert osc.values["p1_hwhm"] == pytest.approx(0.4, abs=1e-5)
    assert abs(lor.values["p1_hwhm"] - 0.4) > 100 * abs(osc.values["p1_hwhm"] - 0.4)


def test_the_delay_option_unwinds_the_cable():
    tau = 3.2
    y = (np.exp(-2j * np.pi * tau * F) * (0.8 + 0.03 * np.exp(0.6j)
                                         * M.chi(F, 7.0, 0.17, 1)) + _noise(F.size, 1e-3))
    s = M.Setup(xmin=5.0, xmax=9.0, delay=True)
    r = M.fit(F, y, s)
    assert r.values["delay"] == pytest.approx(tau, abs=1e-3)
    assert r.values["p1_center"] == pytest.approx(7.0, abs=2e-3)
    assert r.values["p1_hwhm"] == pytest.approx(0.17, rel=0.03)
    assert M.param_unit("delay", "GHz", "", s) == "ns"
    plain = M.fit(F, y, M.Setup(xmin=5.0, xmax=9.0))
    assert abs(plain.values["p1_center"] - 7.0) > 0.05       # lost without it


@pytest.mark.parametrize("k", [3, 10])
def test_derivative_divide_is_exact_on_a_smooth_background(k):
    """Sloped, curved loss + delay, no noise: derivative-divide with the delay
    in the model gives the resonance back exactly, at any step k (the model
    goes through the same finite difference as the data)."""
    b = 0.8 * (1 - 0.012 * F + 0.0004 * F ** 2) * np.exp(-2j * np.pi * 3.2 * F)
    y = b * (1 + 0.03 * np.exp(0.6j) * M.chi(F, 7.0, 0.17, 1, "oscillator"))
    s = M.Setup(xmin=5.3, xmax=8.7, dd=k, delay=True, lineshape="oscillator")
    r = M.fit(F, y, s)
    assert r.values["p1_center"] == pytest.approx(7.0, abs=5e-4)
    assert r.values["p1_hwhm"] == pytest.approx(0.17, abs=5e-4)
    assert r.values["p1_amp"] == pytest.approx(0.03, rel=0.01)   # RELATIVE to the background
    # tau only through the rotation of the peak: a constant phase slope is the
    # same thing as c0, so the delay is loosely pinned -- and need not be more
    assert r.values["delay"] == pytest.approx(3.2, abs=0.1)
    assert M.param_unit("p1_amp", "GHz", "", s) == ""
    d = M.displayed(F, y, s)                  # what is drawn: the derivative-divided data
    assert d.x.size == F.size - 2 * k


def test_derivative_divide_with_noise_wants_a_step_like_the_linewidth():
    """Noise is divided by the step: at k = 10 (0.2 GHz, about the HWHM) the
    resonance comes out within its error bars; small steps amplify the noise
    (k = 3 gave twice the amplitude on these data)."""
    b = 0.8 * (1 - 0.012 * F + 0.0004 * F ** 2) * np.exp(-2j * np.pi * 3.2 * F)
    y = b * (1 + 0.03 * np.exp(0.6j) * M.chi(F, 7.0, 0.17, 1, "oscillator"))
    y = y + _noise(F.size, 3e-4)
    r = M.fit(F, y, M.Setup(xmin=5.3, xmax=8.7, dd=10, delay=True, lineshape="oscillator"))
    assert abs(r.values["p1_center"] - 7.0) < 4 * r.errors["p1_center"]
    assert abs(r.values["p1_hwhm"] - 0.17) < 4 * r.errors["p1_hwhm"]
    rx, rr = M.residuals(r, F, y)
    assert rx.size > 100 and np.isfinite(rr).all()


def test_derivative_divide_without_a_delay_on_a_slow_background():
    b = 0.8 * (1 - 0.012 * F) * np.exp(0.3j)
    y = b * (1 + 0.03 * np.exp(0.6j) * M.chi(F, 7.0, 0.17, 1, "oscillator"))
    r = M.fit(F, y, M.Setup(xmin=5.3, xmax=8.7, dd=3, lineshape="oscillator"))
    assert r.values["p1_center"] == pytest.approx(7.0, abs=1e-3)
    assert r.values["p1_hwhm"] == pytest.approx(0.17, rel=0.01)


def test_reference_subtract_and_divide():
    ref = 0.8 * np.exp(-2j * np.pi * 3.2 * F) * (1 + 0.05 * np.sin(F))
    sig = 0.03 * np.exp(0.6j) * M.chi(F, 7.0, 0.17, 1)
    s = M.Setup(xmin=5.0, xmax=9.0, baseline="constant")
    for how, y in (("subtract", ref + sig), ("divide", ref * (1 + sig))):
        r = M.fit(F, M.reference(F, y, F, ref, how), s)
        assert r.values["p1_center"] == pytest.approx(7.0, abs=1e-6), how
        assert r.values["p1_hwhm"] == pytest.approx(0.17, rel=1e-5), how
    # a reference on another grid is interpolated; outside it: NaN, left out.
    # (Interpolating a phase that winds 0.4 rad per point is NOT accurate: record
    # the reference on the same frequency grid as the measurement.)
    slow = 0.8 * (1 + 0.05 * np.sin(F))
    out = M.reference(F, slow, F[::2][100:], slow[::2][100:], "divide")
    assert np.isnan(out[:200]).all() and np.allclose(out[200:1800], 1.0, atol=1e-4)
    with pytest.raises(ValueError, match="subtract or divide"):
        M.reference(F, ref, F, ref, "multiply")


def test_the_demo_vna_sweeps_with_a_reference():
    """Delay + ripple + loss, as a VNA really is: dividing by the 500 mT sweep
    (nothing resonates in the band there) and the oscillator lineshape give the
    film's resonance and linewidth within the fit's own error bars."""
    out = Path(__file__).resolve().parent / "_demo_tmp"
    DEMO.main([str(out)])
    try:
        path = next(out.glob("*/*_vna_freq_sweeps.nc"))
        ds = load(path).load()
        cs = E.curves_along(ds, E.Selection("s21", x="rf_freq"), "field", range(6), path)
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    ref = cs[-1]
    assert ref.held["field"][0] == 500.0
    for c in cs[:-1]:
        b = c.held["field"][0]
        f0, w = DEMO.kittel(b), DEMO.linewidth(b) / 2
        y = M.reference(c.x, c.z, ref.x, ref.z, "divide")
        r = M.fit(c.x, y, M.Setup(xmin=f0 - 10 * w, xmax=f0 + 10 * w, baseline="constant",
                                  lineshape="oscillator"))
        assert abs(r.values["p1_center"] - f0) < 4 * r.errors["p1_center"] + 1e-3, b
        assert abs(r.values["p1_hwhm"] - w) < 4 * r.errors["p1_hwhm"], b
