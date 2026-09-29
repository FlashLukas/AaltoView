"""The whole chain on the simulated anisotropic film (tools/make_demo_data.py):
curves cut from the files -> one complex-Lorentzian fit per curve -> dispersion
points -> the anisotropy, Meff, gamma and damping the files were made with.

The demo film is computed with the textbook in-plane formula and a brute-force
equilibrium, independently of fmr_fit, so agreement here means the fitting
chain gets the physics right -- including the lineshape fits' own small bias
(a field sweep of a resonance whose frequency is not linear in field).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lmfit")

from aaltoview import export as E  # noqa: E402
from aaltoview.data import load  # noqa: E402
from fmr_fit import dispersion as D  # noqa: E402
from fmr_fit import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_data as DEMO  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    return out


def _fitted(path, x):
    ds = load(path).load()
    curves = E.curves_along(ds, E.Selection("s21", x=x), "phi_H", range(ds.sizes["phi_H"]),
                            path)
    return [(c, M.fit(c.x, c.z, M.Setup())) for c in curves]


@pytest.fixture(scope="module")
def field_rows(demo):
    return _fitted(next(demo.glob("*/*_angle_field_sweeps.nc")), "field")


@pytest.fixture(scope="module")
def freq_rows(demo):
    return _fitted(next(demo.glob("*/*_angle_freq_sweeps.nc")), "rf_freq")


def test_the_coordinates_are_recognised_from_the_units(field_rows):
    src = D.guess_sources(field_rows)
    assert src["H"] == ("dim", "field", None) and src["f"] == ("dim", "rf_freq", None)
    assert src["phi"] == ("dim", "phi_H", 0.0) and src["theta"] == ("const", 90.0)
    pts, problems = D.points_from_fits(field_rows, src)
    assert not problems and len(pts) == 36
    p = pts[9]                                   # phi_H = 90 deg
    assert (p.swept, p.f, p.phi, p.theta) == ("H", 10.0, 90.0, 90.0)


def test_field_sweeps_at_angles_give_the_anisotropy(field_rows):
    """One frequency cannot separate gamma from Meff: gamma held at the film's."""
    pts, _ = D.points_from_fits(field_rows, D.guess_sources(field_rows))
    s = D.Settings(uniaxial=True, fourfold=True, sixfold=True)
    specs = D.default_specs(pts, s)
    specs["gamma"] = M.Spec(DEMO.G, vary=False)
    r = D.fit_positions(pts, s, specs)
    a, v = DEMO.ANISO, r.values
    assert v["Meff"] == pytest.approx(a["Meff"], rel=0.01)
    assert v["Bu"] == pytest.approx(a["Bu"], abs=0.5)
    assert v["phi_u"] == pytest.approx(a["phi_u"], abs=1.5) or \
        abs(abs(v["phi_u"]) - 90) < 1.5                  # +90 and -90 are the same axis
    assert v["B4"] == pytest.approx(a["B4"], abs=0.5)
    assert v["phi_4"] == pytest.approx(a["phi_4"], abs=1.0)
    assert v["B6"] == pytest.approx(a["B6"], abs=0.4)
    assert v["phi_6"] == pytest.approx(a["phi_6"], abs=2.0)
    d = D.fit_damping(pts, r)
    assert d.alpha == pytest.approx(a["alpha"], rel=0.1)
    assert d.dH0 == pytest.approx(a["dB0"], abs=0.3)


def test_field_and_frequency_sweeps_together_separate_gamma(field_rows, freq_rows):
    rows = field_rows + freq_rows
    pts, problems = D.points_from_fits(rows, D.guess_sources(rows))
    assert not problems
    assert {p.swept for p in pts} == {"H", "f"}
    s = D.Settings(uniaxial=True, fourfold=True, sixfold=True)
    # from poor starts too: fitting everything at once went to a false minimum,
    # and gamma free from the start ran off along the gamma-Meff valley (2026-09-29)
    for g0, m0 in ((25.0, 1000.0), (31.0, 2000.0)):
        specs = D.default_specs(pts, s)
        specs["gamma"], specs["Meff"] = M.Spec(g0, min=10, max=60), M.Spec(m0)
        r = D.fit_positions(pts, s, specs)
        assert r.values["gamma"] == pytest.approx(DEMO.G, rel=0.01), (g0, m0)
        assert r.values["Meff"] == pytest.approx(DEMO.ANISO["Meff"], rel=0.02), (g0, m0)
        assert r.values["B4"] == pytest.approx(DEMO.ANISO["B4"], abs=0.6), (g0, m0)


def test_the_plots_and_the_results_table(field_rows, tmp_path):
    pts, _ = D.points_from_fits(field_rows, D.guess_sources(field_rows))
    s = D.Settings(uniaxial=True, fourfold=True, sixfold=True)
    specs = D.default_specs(pts, s)
    specs["gamma"] = M.Spec(DEMO.G, vary=False)
    r = D.fit_positions(pts, s, specs)
    dmp = D.fit_damping(pts, r)
    pos = D.position_plot(pts, r)
    assert (pos.x, pos.y) == ("phi", "H")          # resonance field against the angle
    assert len(pos.data) == 1 and len(pos.data[0].x) == 36
    line = pos.lines[0]
    # the model line goes through the points
    at = np.interp(pos.data[0].x, line.x, line.y)
    assert np.nanmax(np.abs(at - pos.data[0].y)) < 1.0
    wid = D.width_plot(pts, r, dmp)
    assert wid.y == "fwhm" and np.isfinite(wid.lines[0].y).all()
    rows = D.result_rows(r, dmp)
    names = [n for n, *_ in rows]
    assert names[:3] == ["gamma", "g", "Meff"] and "phi_6" in names and names[-1] == "dH0"
    fig = D.figure_dispersion(pos, wid)
    assert len(fig.axes) == 2


def test_a_file_without_the_angle_dimension_joins_with_the_fallback(field_rows, demo):
    """A field sweep measured at ONE fixed angle, in a file with no phi_H
    dimension, next to an angle series: its points used to be left out
    ("no value for phi_H"). Now they take the fallback value (2026-09-29)."""
    ds = load(next(demo.glob("*/*_field_sweeps.nc"))).load()     # rf_freq x field, no angle
    plain = [(c, M.fit(c.x, c.z, M.Setup()))
             for c in E.curves_along(ds, E.Selection("lockin", x="field"), "rf_freq", [1, 2])]
    rows = field_rows[:3] + plain
    src = D.guess_sources(rows)
    notes = []
    pts, problems = D.points_from_fits(rows, src, notes=notes)
    assert not problems and len(pts) == 5
    assert [p.phi for p in pts[3:]] == [0.0, 0.0]
    assert notes and "phi_H = 0 deg for 2 curves" in notes[0]
    # without a fallback they are left out, and the reason says what to do
    src["phi"] = ("dim", "phi_H", None)
    pts, problems = D.points_from_fits(rows, src)
    assert len(pts) == 3 and "give a value in Coordinates" in problems[0]
