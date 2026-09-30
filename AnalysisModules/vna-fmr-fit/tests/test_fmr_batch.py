"""Fit all over many curves keeps the window alive and can be stopped.

174 angle sweeps x 0.24 s blocked the window for 42 s ("Not Responding",
2026-09-30): now one curve at a time, rows updated, events handled, the
button reads Stop. Simulated sweeps with known centres (put IN, checked OUT).
"""

import os

import numpy as np
import pytest

pytest.importorskip("lmfit")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from aaltoview import export as E  # noqa: E402
from fmr_fit import model as M  # noqa: E402

CENTRES = np.linspace(4.2, 5.0, 12)


def _curves():
    rng = np.random.default_rng(3)
    f = np.linspace(3.6, 5.6, 2001)
    out = []
    for k, f0 in enumerate(CENTRES):
        z = 1 + 0.1 * np.exp(-1.3j) * 0.01 / (f0 - f - 0.01j)
        z = z + 2e-4 * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
        out.append(E.Curve(x=f, y=np.abs(z), label=f"phi_H = {2 * k} deg", x_name="rf_freq",
                           x_unit="GHz", y_name="|s21|", y_unit="",
                           selection=E.Selection("s21", x="rf_freq"), z=z,
                           held={"phi_H": (2.0 * k, "deg"), "field": (100.0, "mT")}))
    return out


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow()
    yield w
    w.close()


def test_fit_all_fits_every_curve_one_by_one(win):
    win.add_curves(_curves())
    seen = []
    step = win._batch_step
    win._batch_step = lambda k, n, e, what="fitting": (seen.append((k, n, win.fit_all_btn.text(),
                                                                  win.curve_list.isEnabled())),
                                                      step(k, n, e, what))[1]
    win.fit_all()
    assert [s[0] for s in seen] == list(range(1, 13)) and seen[0][2] == "Stop"
    assert not seen[0][3]                               # the list is locked meanwhile
    for e, f0 in zip(win.entries, CENTRES):
        assert e.result.values["p1_center"] == pytest.approx(f0, abs=2e-4)
    assert win.fit_all_btn.text() == "Fit all" and win.curve_list.isEnabled()
    assert win.fit_btn.isEnabled()


def test_stop_ends_fit_all_after_the_curve_in_hand(win):
    win.add_curves(_curves())
    fit = M.fit
    calls = []

    def fit_then_press_stop(*a, **kw):
        calls.append(1)
        if len(calls) == 3:
            win.fit_all()                               # the button, now reading Stop
        return fit(*a, **kw)

    M.fit, saved = fit_then_press_stop, M.fit
    try:
        win.fit_all()
    finally:
        M.fit = saved
    fitted = [e for e in win.entries if e.result is not None]
    assert len(fitted) == 3                             # the third finishes, then it stops
    assert "STOPPED" in win.status.text()
    assert win.fit_all_btn.text() == "Fit all" and win.guess_btn.isEnabled()
