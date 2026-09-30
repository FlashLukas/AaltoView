"""200 nm YIG: the uniform mode and four standing spin waves, fitted together.

tools/make_demo_data.py file 8, computed with the textbook formulas: Kittel
with H_ex,n = 2 A (n pi / d)^2 / Ms added for PSSW n. The chain -- five peaks
per field sweep, roles by order, one dispersion fit through A -- must give
back A = 3.7 pJ/m, Ms, gamma and the damping.
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
def yig_rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    path = next(out.glob("*/*_yig_200nm_field_sweeps.nc"))
    ds = load(path).load()
    curves = E.curves_along(ds, E.Selection("lockin", x="field"), "rf_freq",
                            range(ds.sizes["rf_freq"]), path)
    return [(c, M.fit(c.x, c.z, M.Setup(n_peaks=5))) for c in curves]


def test_the_exchange_fields_are_the_textbook_ones():
    assert [DEMO.yig_hex(n) for n in (1, 2, 3, 4)] == pytest.approx(
        [13.03, 52.13, 117.3, 208.5], abs=0.1)
    s = D.Settings(pssw="A", d_nm=200.0, Ms_mT=176.0)
    assert 3.7 * D._a_coefficient(1, s) == pytest.approx(DEMO.yig_hex(1), rel=1e-9)


def _kittel_field(f: float, n: int) -> float:
    """Resonance field (mT) of YIG mode n at f: f = g sqrt((B+Hex)(B+Hex+Ms))."""
    ms, b = DEMO.YIG["Ms"], f / (DEMO.G / 1000)
    return (-ms + np.sqrt(ms * ms + 4 * b * b)) / 2 - (DEMO.yig_hex(n) if n else 0.0)


def test_five_peaks_per_sweep_are_found(yig_rows):
    for c, r in yig_rows:
        f = c.held["rf_freq"][0]
        assert not M.suspicious(r), (f, M.suspicious(r))
        found = sorted(r.values[f"p{k}_center"] for k in range(1, 6))
        want = sorted(_kittel_field(f, n) for n in range(5))
        assert np.allclose(found, want, atol=0.2), (f, found, want)


def test_roles_by_order_and_the_exchange_stiffness(yig_rows):
    src = D.guess_sources(yig_rows)
    roles = D.pssw_roles(yig_rows, src)
    pts, problems = D.points_from_fits(yig_rows, src, roles)
    assert not problems
    assert sorted({p.role for p in pts}) == [0, 1, 2, 3, 4]
    # through A: one number for all four orders
    s = D.Settings(pssw="A", d_nm=200.0, Ms_mT=176.0)
    r = D.fit_positions(pts, s)
    assert r.values["A"] == pytest.approx(DEMO.YIG["A"], rel=0.02)
    assert r.values["Meff"] == pytest.approx(DEMO.YIG["Ms"], rel=0.02)
    assert r.values["gamma"] == pytest.approx(DEMO.G, rel=0.01)
    # per mode: each exchange field on its own -- the n^2 law is then a check
    free = D.fit_positions(pts, D.Settings(pssw="free", d_nm=200.0, Ms_mT=176.0))
    for n in (1, 2, 3, 4):
        assert free.values[f"Hex{n}"] == pytest.approx(DEMO.yig_hex(n), rel=0.03), n
        assert free.values[f"A{n}"] == pytest.approx(DEMO.YIG["A"], rel=0.05), n
    d = D.fit_damping(pts, r)
    assert abs(d.alpha - DEMO.YIG["alpha"]) < 4 * d.alpha_err
    assert abs(d.dH0 - DEMO.YIG["dB0"]) < 4 * d.dH0_err


def test_the_window_path_from_one_to_five_peaks(tmp_path):
    """The window keeps peak 1 when the peak count goes from 1 to 5; peak 2
    was put on the same line again, "by order" then numbered every PSSW one
    off and A came out 2.7 instead of 3.7 (screenshot, 2026-09-29)."""
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    DEMO.main([str(tmp_path)])
    path = next(tmp_path.glob("*/*_yig_200nm_field_sweeps.nc"))
    ds = load(path).load()
    curves = E.curves_along(ds, E.Selection("lockin", x="field"), "rf_freq", range(8), path)
    win = FitWindow()
    try:
        win.add_curves(curves)
        win.peaks.setValue(5)
        win.fit_all()
        for e in win.entries:
            f = e.curve.held["rf_freq"][0]
            got = sorted(e.result.values[f"p{k}_center"] for k in range(1, 6))
            assert np.allclose(got, sorted(_kittel_field(f, n) for n in range(5)), atol=0.2), f
        win.tabs.setCurrentWidget(win.dispersion)
        tab = win.dispersion
        tab.assign_pssw()
        tab.pssw.setCurrentIndex(tab.pssw.findData("A"))
        tab.d_edit.setText("200")
        tab.ms_edit.setText("176")
        tab._model_changed()
        tab.fit()
        assert tab.dres.values["A"] == pytest.approx(DEMO.YIG["A"], rel=0.02)
    finally:
        win.close()


def test_every_curve_of_the_window_path_has_error_bars(tmp_path):
    """The curve on screen during Fit all is fitted from its start; that start
    mixed a peak guessed for one hand with peaks guessed for the other, every
    fit from it failed (chi2 221 vs 27) and it had no error bars."""
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    DEMO.main([str(tmp_path)])
    path = next(tmp_path.glob("*/*_yig_200nm_field_sweeps.nc"))
    ds = load(path).load()
    curves = E.curves_along(ds, E.Selection("lockin", x="field"), "rf_freq", range(3), path)
    win = FitWindow()
    try:
        win.add_curves(curves)
        win.peaks.setValue(5)
        win.fit_all()
        for e in win.entries:
            r = e.result
            assert all(r.errors[f"p{k}_center"] is not None for k in range(1, 6)), e.curve.label
            assert r.redchi < 2 * min(o.result.redchi for o in win.entries), e.curve.label
    finally:
        win.close()


def test_fitting_again_from_the_result_is_quick(yig_rows, monkeypatch):
    """Counted in model evaluations, not seconds (machines differ): from its
    own result a 5-peak YIG fit also tried the wrong hand, twice into lmfit's
    50 000-evaluation limit -- 65 s for a curve that fits in 0.15 s (2026-09-30)."""
    import lmfit
    c, r = yig_rows[3]
    used = []
    orig = lmfit.minimize

    def counting(*a, **k):
        out = orig(*a, **k)
        used.append(out.nfev)
        return out

    monkeypatch.setattr(lmfit, "minimize", counting)
    again = M.fit(c.x, c.z, M.Setup(n_peaks=5), start=r.specs())
    assert sum(used) < 3000, used
    assert again.chisqr == pytest.approx(r.chisqr, rel=1e-3)
