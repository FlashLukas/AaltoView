"""Re on the left axis, Im on the right, each scaled to itself (asked for
2026-10-08: a VNA's Re sits near 1 and Im near 0; on one shared scale the
resonance was a few pixels). Simulated sweep, known numbers."""

import os

import numpy as np
import pytest

pytest.importorskip("lmfit")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from aaltoview import export as E  # noqa: E402


def _curve(complex_=True):
    f = np.linspace(3.6, 5.6, 2001)
    z = 1 + 0.05 * np.exp(-1.3j) * 0.01 / (4.7 - f - 0.01j)
    return E.Curve(x=f, y=np.abs(z), label="phi_H = 0 deg", x_name="rf_freq", x_unit="GHz",
                   y_name="|s21|", y_unit="", selection=E.Selection("s21", x="rf_freq"),
                   z=z if complex_ else None, held={"field": (100.0, "mT")})


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    w.close()


def _ranges(win):
    QtWidgets.QApplication.processEvents()
    return win.plot.vb.viewRange()[1], win.vb2.viewRange()[1]


def test_im_gets_its_own_axis_and_scale(win):
    win.add_curves([_curve()])
    win.fit()
    on_right = [it for plot, it in win._items if plot is win.vb2]
    assert len(on_right) == 2                       # Im data + Im fit
    assert win.plot.getAxis("right").isVisible()
    (re_lo, re_hi), (im_lo, im_hi) = _ranges(win)
    assert re_lo < 1.0 < re_hi and re_hi - re_lo < 0.2      # Re about 1, its own scale
    assert im_hi - im_lo < 0.2 and abs((im_lo + im_hi) / 2) < 0.05   # Im about 0
    names = [lab.text for _, lab in win.legend.items]
    assert "Im data" in names and "Im fit" in names
    # the fit is not changed by the drawing: the centre put in comes out
    assert win.current().result.values["p1_center"] == pytest.approx(4.7, abs=1e-4)


def test_one_axis_when_unticked_or_real(win):
    win.add_curves([_curve()])
    win.two_axes.setChecked(False)
    assert not [it for plot, it in win._items if plot is win.vb2]
    assert not win.plot.getAxis("right").isVisible()
    win.add_curves([_curve(complex_=False)])
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(1))
    win.two_axes.setChecked(True)
    assert not [it for plot, it in win._items if plot is win.vb2]   # one part: one axis
    assert not win.two_axes.isEnabled()
