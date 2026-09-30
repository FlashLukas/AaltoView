"""The stripe dispersion against formulas derived independently, then fitted."""

import numpy as np
import pytest

from sw_fft import waveguide as W

PY = dict(gamma=28.0, Ms=1000.0, A=13.0, d=30.0, w=0.0, n=0.0, B=50.0, theta=90.0)
YIG = dict(gamma=28.0, Ms=176.0, A=3.7, d=100.0, w=0.0, n=0.0, B=20.0, theta=90.0)


def test_k_zero_is_kittel():
    f = W.frequency([0.0], PY)[0]
    assert f == pytest.approx(28.0 * np.sqrt(0.05 * 1.05))


def test_the_exchange_length_is_the_textbook_one():
    # permalloy 5.7 nm, YIG 17 nm
    assert W._derived(PY, "unpinned")["l_ex"][0] == pytest.approx(5.70, abs=0.02)
    assert W._derived(YIG, "unpinned")["l_ex"][0] == pytest.approx(17.3, abs=0.1)


def test_damon_eshbach_and_backward_volume_start_as_the_textbook_slopes():
    """f^2 = f0^2 + (g Ms)^2 k d / 2 (DE), f^2 = g^2 B (B + Ms (1 - k d / 2)) (BV)."""
    k = 0.2                                             # rad/um: kd = 0.006
    g, B, Ms, d = 28.0, 0.05, 1.0, 30e-9
    kd = k * 1e6 * d
    f0 = g * np.sqrt(B * (B + Ms))
    de = np.sqrt(f0 ** 2 + (g * Ms) ** 2 * kd / 2)
    bv = g * np.sqrt(B * (B + Ms * (1 - kd / 2)))
    v = dict(PY, A=0.0)                                 # dipolar only
    assert W.frequency([k], v)[0] == pytest.approx(de, rel=2e-3)
    assert W.frequency([k], dict(v, theta=0.0))[0] == pytest.approx(bv, rel=2e-3)


def test_damon_eshbach_follows_the_exact_surface_wave_at_small_kd():
    """Damon & Eshbach (1961): f^2 = g^2 (B (B + Ms) + Ms^2 (1 - exp(-2kd)) / 4).

    Kalinikos-Slavin's lowest mode is a thin-film expansion: its dipolar term
    Ms^2 P (1 - P) equals the exact Ms^2 (1 - exp(-2kd)) / 4 to first order in
    kd and exceeds it by Ms^2 (kd)^2 / 12 at second order -- so the agreement
    must be that good, and no better."""
    v = dict(YIG, A=0.0)
    for k in (0.5, 2.0, 5.0):                            # kd = 0.05 ... 0.5
        kd = k * 1e6 * v["d"] * 1e-9
        B, Ms = v["B"] * 1e-3, v["Ms"] * 1e-3
        exact2 = B * (B + Ms) + Ms ** 2 * (1 - np.exp(-2 * kd)) / 4
        second = Ms ** 2 * kd ** 2 / 12                  # in f^2 / g^2
        got2 = (W.frequency([k], v)[0] / 28.0) ** 2
        assert got2 > exact2
        if kd <= 0.05:
            assert got2 - exact2 == pytest.approx(second, rel=0.1), kd
        else:                                            # higher orders pull it down
            assert got2 - exact2 < second


def test_guslienko_effective_width():
    # p = d / w = 0.015: D = 2 pi / (p (1 + 2 ln(1/p))) = 44.565, w_eff = w D / (D - 2)
    assert W.guslienko_width(2.0, 30.0) == pytest.approx(2.0 * 44.5652 / 42.5652, rel=1e-4)
    assert W.guslienko_width(100.0, 1.0) == pytest.approx(100.0, rel=1e-4)   # wide & thin
    assert W.guslienko_width(0.0, 30.0) == 0.0
    v = dict(PY, w=2.0, n=1.0)
    assert W.width_wavevector(v) == pytest.approx(np.pi / W.guslienko_width(2.0, 30.0))
    assert W.width_wavevector(v, "unpinned") == pytest.approx(np.pi / 2.0)
    # the width mode lifts the band: at kx = 0 it is the film at k = ky
    ky = W.width_wavevector(v)
    film_bv = W.frequency([ky], dict(PY, theta=90.0 - 90.0))[0]     # k along M: BV-like
    assert W.frequency([0.0], v)[0] == pytest.approx(film_bv, rel=1e-9)


def test_the_fit_gives_back_what_was_put_in():
    rng = np.random.default_rng(3)
    truth = dict(PY, w=2.0, n=1.0, B=40.0)
    k = np.linspace(0.5, 8.0, 25)
    f = W.frequency(k, truth) + rng.normal(0, 0.005, k.size)
    pts = [W.Point(k=kk, f=ff) for kk, ff in zip(k, f)]
    specs = W.default_specs()
    for n, v in truth.items():
        specs[n].value = v
    specs["Ms"] = W.Spec(800.0, True, 1, 3000)
    specs["B"] = W.Spec(60.0, True, 0, 1000)
    res = W.fit(pts, specs)
    assert res.success
    assert res.values["Ms"] == pytest.approx(1000.0, rel=5e-3)
    assert res.values["B"] == pytest.approx(40.0, rel=5e-3)
    assert res.errors["Ms"] < 10 and res.rms < 0.01
    # per-point fields: B is not a free parameter then
    for p in pts:
        p.B = 40.0
    with pytest.raises(ValueError, match="own field"):
        W.fit(pts, specs)
    specs["B"].vary = False
    assert W.fit(pts, specs).values["Ms"] == pytest.approx(1000.0, rel=5e-3)
