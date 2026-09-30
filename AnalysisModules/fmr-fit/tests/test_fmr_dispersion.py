"""The magnetic model (dispersion.py) against formulas derived independently.

dispersion.py finds the equilibrium numerically on the sphere and takes the
curvature of the energy in a local frame. Here the same physics is written the
textbook way -- the in-plane Kittel formula with every anisotropy term, the
equilibrium angle by brute force over a fine grid -- and the two must agree.
Then the fits must give back parameters put into synthetic data.
"""

import numpy as np
import pytest

pytest.importorskip("lmfit")

from fmr_fit import dispersion as D  # noqa: E402

FILM = dict(gamma=29.0, Meff=1200.0, Bu=15.0, phi_u=30.0, B4=10.0, phi_4=5.0,
            B6=4.0, phi_6=-10.0)


def inplane_textbook(H, phiH, p, hex_=0.0):
    """f and the equilibrium angle of an in-plane magnetised film, field in-plane."""
    phi = np.deg2rad(np.linspace(-180, 180, 360001))
    dh = phi - np.deg2rad(phiH)
    du, d4, d6 = (phi - np.deg2rad(p[k]) for k in ("phi_u", "phi_4", "phi_6"))
    E = (-H * np.cos(dh) - p["Bu"] / 2 * np.cos(du) ** 2
         - p["B4"] / 16 * (3 + np.cos(4 * d4)) - p["B6"] / 36 * np.cos(6 * d6))
    i = np.argmin(E)
    dh, du, d4, d6 = dh[i], du[i], d4[i], d6[i]
    e_pp = (H * np.cos(dh) + p["Bu"] * np.cos(2 * du) + p["B4"] * np.cos(4 * d4)
            + p["B6"] * np.cos(6 * d6))
    e_tt = (H * np.cos(dh) + p["Meff"] + p["Bu"] * np.cos(du) ** 2
            + p["B4"] / 4 * (3 + np.cos(4 * d4)) + p["B6"] / 6 * np.cos(6 * d6))
    f = p["gamma"] * 1e-3 * np.sqrt((e_pp + hex_) * (e_tt + hex_))
    return f, np.rad2deg(phi[i])


@pytest.mark.parametrize("phiH", [0.0, 23.0, 75.0, 140.0, -100.0])
@pytest.mark.parametrize("H", [25.0, 80.0, 300.0])
def test_inplane_matches_the_textbook_formula(H, phiH):
    f_num, _, m = D.modes(np.array([H]), 90.0, phiH, FILM)
    f_ref, phi_ref = inplane_textbook(H, phiH, FILM)
    assert f_num[0] == pytest.approx(f_ref, rel=2e-5)
    assert np.rad2deg(np.arctan2(m[0, 1], m[0, 0])) == pytest.approx(phi_ref, abs=0.02)
    assert abs(m[0, 2]) < 1e-6                     # stays in the plane


def test_perpendicular_field_above_saturation():
    """theta_H = 0, H > Meff: f = gamma (H - Meff), whatever the in-plane terms."""
    iso = dict(FILM, Bu=0.0, B4=0.0, B6=0.0)
    H = np.array([1300.0, 1500.0, 2000.0])
    f = D.frequency(H, 0.0, 0.0, iso)
    assert f == pytest.approx(iso["gamma"] * 1e-3 * (H - iso["Meff"]), rel=1e-5)


def test_a_standing_spin_wave_adds_its_exchange_field_to_both_stiffnesses():
    iso = dict(FILM, Bu=0.0, B4=0.0, B6=0.0)
    H = np.array([50.0, 150.0])
    f = D.frequency(H, 90.0, 0.0, iso, hex_=180.0)
    ref = iso["gamma"] * 1e-3 * np.sqrt((H + 180) * (H + 180 + iso["Meff"]))
    assert f == pytest.approx(ref, rel=1e-5)


def test_inplane_field_swept_linewidth_is_dH0_plus_2_alpha_f_over_gamma():
    iso = dict(FILM, Bu=0.0, B4=0.0, B6=0.0)
    pts = [D.Point(H=h, f=D.frequency(np.array([h]), 90, 0, iso)[0], theta=90, phi=0,
                   swept="H") for h in (40.0, 90.0, 160.0)]
    w = D.linewidth(pts, iso, alpha=0.008, dH0=0.7)
    f = np.array([p.f for p in pts])
    assert w == pytest.approx(0.7 + 2 * 0.008 * f / (iso["gamma"] * 1e-3), rel=1e-4)


def test_resonance_field_inverts_the_frequency():
    f = np.array([4.0, 8.0, 12.0])
    for phiH in (0.0, 60.0):
        H = D.resonance_field(f, 90.0, phiH, FILM)
        assert D.frequency(H, 90.0, phiH, FILM) == pytest.approx(f, rel=1e-5)


def _angle_points(p, f=10.0, noise=0.05, seed=4):
    rng = np.random.default_rng(seed)
    phis = np.arange(0.0, 360.0, 10.0)
    Hs = D.resonance_field(np.full(phis.size, f), 90.0, phis, p)
    return [D.Point(H=H + noise * rng.normal(), f=f, theta=90.0, phi=phiH, swept="H",
                    err=noise) for H, phiH in zip(Hs, phis)]


def test_field_sweeps_at_angles_give_back_the_anisotropy():
    """Resonance field vs in-plane angle at 10 GHz, plus a few frequencies at
    one angle (without those, gamma and Meff are not separable)."""
    pts = _angle_points(FILM)
    for f in (5.0, 15.0, 20.0):
        H = D.resonance_field(np.array([f]), 90.0, 0.0, FILM)[0]
        pts.append(D.Point(H=H, f=f, theta=90.0, phi=0.0, swept="H", err=0.05))
    s = D.Settings(uniaxial=True, fourfold=True, sixfold=True)
    r = D.fit_positions(pts, s)
    v = r.values
    assert v["gamma"] == pytest.approx(29.0, rel=0.01)
    assert v["Meff"] == pytest.approx(1200.0, rel=0.02)
    assert v["Bu"] == pytest.approx(15.0, abs=0.3) and v["phi_u"] == pytest.approx(30.0, abs=1.0)
    assert v["B4"] == pytest.approx(10.0, abs=0.3) and v["phi_4"] == pytest.approx(5.0, abs=1.0)
    assert v["B6"] == pytest.approx(4.0, abs=0.3) and v["phi_6"] == pytest.approx(-10.0, abs=1.0)
    assert v["g"] == pytest.approx(29.0 / D.GAMMA_E, rel=0.01)
    assert r.errors["Bu"] is not None and r.errors["phi_u"] is not None


def test_frequency_sweeps_vs_angle_give_back_the_anisotropy():
    rng = np.random.default_rng(5)
    pts = []
    for H in (30.0, 100.0):
        for phiH in np.arange(0.0, 360.0, 15.0):
            f = D.frequency(np.array([H]), 90.0, phiH, FILM)[0]
            pts.append(D.Point(H=H, f=f + 0.002 * rng.normal(), theta=90.0, phi=phiH,
                               swept="f", err=0.002))
    r = D.fit_positions(pts, D.Settings(uniaxial=True, fourfold=True, sixfold=True))
    assert r.values["Bu"] == pytest.approx(15.0, abs=0.5)
    assert r.values["phi_4"] == pytest.approx(5.0, abs=1.0)


def test_kittel_plus_pssw_gives_the_exchange_stiffness():
    """Uniform + first PSSW vs frequency; A from Hex = 2 A (pi/d)^2 / Ms."""
    iso = dict(FILM, Bu=0.0, B4=0.0, B6=0.0)
    s = D.Settings(pssw="A", d_nm=60.0, Ms_mT=1000.0)     # PSSW below 20 GHz
    A_true = 13.0
    hex1 = A_true * D._a_coefficient(1, s)
    pts = []
    for f in np.arange(4.0, 20.0, 2.0):
        pts.append(D.Point(H=D.resonance_field(np.array([f]), 90, 0, iso)[0], f=f,
                           theta=90, phi=0, swept="H", err=0.05))
        h1 = D.resonance_field(np.array([f]), 90, 0, iso, hex_=hex1)[0]
        if np.isfinite(h1):
            pts.append(D.Point(H=h1, f=f, theta=90, phi=0, swept="H", err=0.05, role=1))
    r = D.fit_positions(pts, s)
    assert r.values["A"] == pytest.approx(A_true, rel=0.01)
    assert r.values["Hex1"] == pytest.approx(hex1, rel=0.01)
    free = D.fit_positions(pts, D.Settings(pssw="free", d_nm=60.0, Ms_mT=1000.0))
    assert free.values["A1"] == pytest.approx(A_true, rel=0.01)


def test_damping_from_the_linewidths():
    iso = dict(FILM, Bu=0.0, B4=0.0, B6=0.0)
    pts = []
    for f in np.arange(4.0, 20.0, 2.0):
        H = D.resonance_field(np.array([f]), 90, 0, iso)[0]
        pts.append(D.Point(H=H, f=f, theta=90, phi=0, swept="H", err=0.05,
                           fwhm=0.9 + 2 * 0.0075 * f / (iso["gamma"] * 1e-3),
                           fwhm_err=0.02))
    r = D.fit_positions(pts, D.Settings())
    dmp = D.fit_damping(pts, r)
    assert dmp.alpha == pytest.approx(0.0075, rel=0.01)
    assert dmp.dH0 == pytest.approx(0.9, abs=0.02)


def test_units_are_recognised():
    assert D.unit_kind("mT") == ("field", 1.0)
    assert D.unit_kind("Oe") == ("field", 0.1)
    assert D.unit_kind("MHz") == ("freq", 1e-3)
    assert D.unit_kind("rad")[0] == "angle"
    assert D.unit_kind("V") == (None, 1.0)
