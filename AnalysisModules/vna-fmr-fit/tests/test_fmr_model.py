"""The VNA-FMR fit: does it give back the numbers that went in?

Synthetic curves with known resonance, width, amplitude and phase, both
handednesses, noise and holes; then the simulated measurements of
tools/make_demo_data.py, whose resonance fields and linewidths follow from the
Kittel formula -- so the fit is checked against physics, not against itself.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lmfit")

from fmr_fit import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TRUE = dict(p1_center=97.3, p1_hwhm=4.2, p1_amp=2.0, p1_phase=35.0,
            bg_re=5.0, bg_im=-2.0, slope_re=0.003, slope_im=0.001)


def _signal(hand, sigma=0.05, seed=1, x=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 150, 301) if x is None else x
    y = M.evaluate(TRUE, x, M.Setup(), hand, 100.0)
    return x, y + sigma * (rng.normal(size=x.size) + 1j * rng.normal(size=x.size))


@pytest.mark.parametrize("hand", [1, -1])
def test_complex_fit_recovers_the_parameters_and_the_handedness(hand):
    x, y = _signal(hand)
    r = M.fit(x, y, M.Setup())
    assert r.hand == hand and r.success
    assert r.values["p1_center"] == pytest.approx(97.3, abs=0.15)
    assert r.values["p1_hwhm"] == pytest.approx(4.2, rel=0.03)
    assert r.values["p1_amp"] == pytest.approx(2.0, rel=0.03)
    assert r.values["p1_phase"] == pytest.approx(35.0, abs=2.0)
    # the error bars mean something: the truth is within ~3 sigma
    for k in ("p1_center", "p1_hwhm"):
        assert abs(r.values[k] - TRUE[k]) < 4 * r.errors[k]
    w, we = r.fwhm(1)
    assert w == pytest.approx(2 * r.values["p1_hwhm"]) and we == pytest.approx(2 * r.errors["p1_hwhm"])


@pytest.mark.parametrize("part", [np.real, np.imag])
def test_one_channel_gives_the_same_resonance(part):
    x, y = _signal(-1)
    r = M.fit(x, part(y), M.Setup(mode="real"))
    assert r.values["p1_center"] == pytest.approx(97.3, abs=0.3)
    assert r.values["p1_hwhm"] == pytest.approx(4.2, rel=0.05)


def test_magnitude_on_a_large_background_is_a_mixed_lorentzian():
    """Only when the background is much larger than the peak (here 20x): then
    |bg + s| = |bg| + Re(s e^{-i arg bg}) to first order."""
    x, y = _signal(1, sigma=0.01)
    y = y + 40.0
    r = M.fit(x, np.abs(y), M.Setup(mode="real"))
    assert r.values["p1_center"] == pytest.approx(97.3, abs=0.3)
    assert r.values["p1_hwhm"] == pytest.approx(4.2, rel=0.08)


def test_holes_and_the_fit_range_are_respected():
    x, y = _signal(1)
    y[40:60] = np.nan                                    # an aborted stretch
    r = M.fit(x, y, M.Setup(xmin=70, xmax=130))
    assert r.xrange[0] >= 70 and r.xrange[1] <= 130
    assert r.ndata == 2 * np.sum((x >= 70) & (x <= 130) & np.isfinite(y))
    assert r.values["p1_center"] == pytest.approx(97.3, abs=0.2)


def test_a_fixed_parameter_stays_where_it_was_put():
    x, y = _signal(1)
    setup = M.Setup(hand=1)
    start = M.guess(x, y, setup, hand=1)
    start["p1_hwhm"] = M.Spec(5.0, vary=False)
    r = M.fit(x, y, setup, start=start)
    assert r.values["p1_hwhm"] == 5.0 and r.errors["p1_hwhm"] is None
    assert not r.vary["p1_hwhm"]


def test_two_peaks_are_separated():
    x = np.linspace(0, 200, 801)
    two = dict(TRUE, p1_center=60.0, p1_hwhm=3.0, p2_center=140.0, p2_hwhm=6.0,
               p2_amp=0.8, p2_phase=-40.0)
    rng = np.random.default_rng(3)
    y = M.evaluate(two, x, M.Setup(n_peaks=2), 1, 100.0)
    y = y + 0.02 * (rng.normal(size=x.size) + 1j * rng.normal(size=x.size))
    r = M.fit(x, y, M.Setup(n_peaks=2))
    got = sorted([(r.values["p1_center"], r.values["p1_hwhm"]),
                  (r.values["p2_center"], r.values["p2_hwhm"])])
    assert got[0] == pytest.approx((60.0, 3.0), rel=0.03)
    assert got[1] == pytest.approx((140.0, 6.0), rel=0.03)


def test_adding_a_peak_keeps_the_tuned_one():
    x, y = _signal(1)
    first = M.guess(x, y, M.Setup(n_peaks=1), hand=1)
    first["p1_center"].value = 99.0                     # the operator nudged it
    both = M.guess(x, y, M.Setup(n_peaks=2), hand=1, keep=first)
    assert both["p1_center"].value == 99.0 and "p2_center" in both


def test_the_demo_field_sweeps_give_the_kittel_resonance_and_linewidth():
    """The simulated film (tools/make_demo_data.py): f = g sqrt(B (B + Ms)), and
    a field-swept FWHM of dB0 + 2 alpha f / g (the demo's frequency linewidth
    alpha g (2B + Ms) + dB0 df/dB, seen along the field)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import make_demo_data as D
    D.RNG = np.random.default_rng(7)
    fb = np.linspace(0, 200, 401)
    for f in (6.0, 8.0, 10.0):
        z = D.fmr(fb, [f])[:, 0] + D.noise(401, 0.3)
        b = (-D.MS + np.sqrt(D.MS ** 2 + 4 * (f / D.G * 1000) ** 2)) / 2
        hwhm = (D.DB0 + 2 * D.ALPHA * f / (D.G / 1000)) / 2
        r = M.fit(fb, z, M.Setup())
        assert r.values["p1_center"] == pytest.approx(b, abs=0.3), f
        assert r.values["p1_hwhm"] == pytest.approx(hwhm, rel=0.04), f


def test_the_results_table_and_file(tmp_path):
    from aaltoview.export import Curve, Selection
    rows = []
    for f, seed in ((6.0, 1), (8.0, 2)):
        x, y = _signal(1, seed=seed)
        c = Curve(x=x, y=np.abs(y), label=f"rf_freq = {f:g} GHz", x_name="field", x_unit="mT",
                  y_name="|s21|", y_unit="V", selection=Selection("s21", x="field"),
                  source=r"D:\data\120000_sweeps.nc", z=y, held={"rf_freq": (f, "GHz")})
        rows.append((c, M.fit(x, y, M.Setup())))
    cols = M.results_table(rows)
    names = [c.name for c in cols]
    assert names[0] == "rf_freq" and cols[0].kind == "X" and cols[0].values == [6.0, 8.0]
    i = names.index("p1_fwhm")
    assert cols[i].unit == "mT" and cols[i + 1].kind == "E"
    assert cols[names.index("p1_phase")].unit == "deg"
    p = M.write_results(tmp_path / "r.csv", cols)
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("rf_freq,") and lines[1].startswith("GHz,")
    assert lines[3].startswith("6.0,") and lines[3].endswith("120000_sweeps.nc")
    fig = M.figure_fit(rows[0][0], rows[0][1], rows[0][0].z)
    assert len(fig.axes) == 2


def test_a_range_drawn_at_6_ghz_follows_the_peak_to_10_ghz():
    """The demo film: 44 mT at 6 GHz, 158 mT at 12 GHz, and broader. A window
    of fixed field values would miss the 12 GHz line; one in linewidths around
    x0 lands on it, and grows with the width."""
    sys.path.insert(0, str(ROOT / "tools"))
    import make_demo_data as D
    D.RNG = np.random.default_rng(11)
    fb = np.linspace(0, 200, 401)
    z6 = D.fmr(fb, [6.0])[:, 0] + D.noise(401, 0.3)
    z10 = D.fmr(fb, [10.0])[:, 0] + D.noise(401, 0.3)
    setup = M.Setup()
    x0, w = M.locate(fb, z6, setup)
    lo, hi = M.range_in_widths(x0, w, 27.0, 65.0)          # the band drawn at 6 GHz
    assert lo < 0 < hi
    xmin, xmax = M.follow_range(fb, z10, setup, lo, hi)
    x10, w10 = M.locate(fb, z10, setup)
    assert xmin < x10 < xmax and xmin > 65.0          # nowhere near the 6 GHz band
    assert (xmax - xmin) == pytest.approx((65.0 - 27.0) * w10 / w, rel=1e-6)
    r = M.fit(fb, z10, M.Setup(xmin=xmin, xmax=xmax))
    assert r.values["p1_center"] == pytest.approx(x10, abs=0.3)
