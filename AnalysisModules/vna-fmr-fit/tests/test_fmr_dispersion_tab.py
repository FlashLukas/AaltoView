"""The Dispersion tab, offscreen: resonances in, roles, fit, results out."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lmfit")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from aaltoview import export as E  # noqa: E402
from aaltoview.data import load  # noqa: E402
from fmr_fit import dispersion as D  # noqa: E402
from fmr_fit import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_data as DEMO  # noqa: E402


@pytest.fixture(scope="module")
def angle_curves(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    path = next(out.glob("*/*_angle_field_sweeps.nc"))
    ds = load(path).load()
    return E.curves_along(ds, E.Selection("s21", x="field"), "phi_H", range(36), path)


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow()
    yield w
    w.close()


def test_resonances_become_points_and_the_anisotropy_is_fitted(win, angle_curves):
    win.add_curves(angle_curves)
    win.fit_all()
    win.tabs.setCurrentWidget(win.dispersion)             # refreshes the points
    tab = win.dispersion
    assert len(tab.points) == 36 and tab.ptable.rowCount() == 36
    # one frequency: gamma cannot be separated from Meff, so it starts fixed
    assert not tab.specs["gamma"].vary and "γ fixed" in tab.status.text()
    assert tab.coord_rows["phi"][0].currentText() == "phi_H"
    for box in (tab.uni, tab.four, tab.six):
        box.setChecked(True)
    tab.fit()
    v = tab.dres.values
    assert v["Bu"] == pytest.approx(DEMO.ANISO["Bu"], abs=0.6)
    assert v["B4"] == pytest.approx(DEMO.ANISO["B4"], abs=0.6)
    assert tab.damp.alpha == pytest.approx(DEMO.ANISO["alpha"], rel=0.15)
    assert tab.results.rowCount() == len(D.result_rows(tab.dres, tab.damp))
    cols = tab.columns()
    assert cols[0].values[:3] == ["gamma", "g", "Meff"]
    assert len(tab.figure().axes) == 2


def test_a_point_can_be_left_out_and_a_peak_flagged_as_pssw(win, angle_curves):
    win.add_curves(angle_curves[:6])
    win.fit_all()
    win.tabs.setCurrentWidget(win.dispersion)
    tab = win.dispersion
    tab.ptable.item(0, 0).setCheckState(QtCore.Qt.Unchecked)
    assert not tab.points[0].use
    combo = tab.ptable.cellWidget(1, 2)
    combo.setCurrentIndex(combo.findData(1))               # PSSW n = 1
    assert tab.points[1].role == 1
    assert "Hex1" in tab._names()
    # the choices survive a refresh (coming back to the tab after more fits)
    tab.refresh()
    assert not tab.points[0].use and tab.points[1].role == 1
