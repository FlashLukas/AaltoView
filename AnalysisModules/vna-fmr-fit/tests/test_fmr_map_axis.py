"""A map sent with FIELD on X is still fitted in FREQUENCY (asked for
2026-10-09): the fit cuts it along its frequency axis, one sweep per field.
Simulated Kittel line; the frequency put in comes out."""

import os

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("lmfit")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from aaltoview import export as E  # noqa: E402

FIELDS = np.array([40.0, 60.0, 80.0, 100.0])
FREQ = np.linspace(2.0, 8.0, 1201)


def kittel(b):
    return 28.0e-3 * np.sqrt(b * (b + 800.0))        # GHz, b in mT


def _ds():
    z = 1 + 0.05 * np.exp(-0.7j) * 0.02 / (kittel(FIELDS)[:, None] - FREQ[None, :] - 0.02j)
    pair = {"complex_pair": "s21"}
    return xr.Dataset({"s21_real": (("field", "rf_freq"), z.real, {**pair, "complex_part": "real"}),
                       "s21_imag": (("field", "rf_freq"), z.imag, {**pair, "complex_part": "imag"})},
                      coords={"field": ("field", FIELDS, {"units": "mT"}),
                              "rf_freq": ("rf_freq", FREQ, {"units": "GHz"})},
                      attrs={"dims": "field,rf_freq"})


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow()
    yield w
    w.close()


def test_field_on_x_is_cut_into_frequency_sweeps(win):
    ds = _ds()
    m = E.make_map_data(ds, E.Selection("s21", x="field", y="rf_freq"))   # X = field
    win.add_maps([m])
    assert len(win.entries) == len(FIELDS)                                 # one per field
    for e, b in zip(win.entries, FIELDS):
        c = e.curve
        assert c.x_name == "rf_freq" and c.x.size == FREQ.size
        assert c.held["field"] == (b, "mT")
    assert "along rf_freq (the map's Y axis" in win.status.text()
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(2))           # 80 mT
    win.fit()
    assert win.current().result.values["p1_center"] == pytest.approx(kittel(80.0), abs=1e-3)


def test_frequency_on_x_is_cut_as_before_and_both_give_the_same_sweeps(win):
    ds = _ds()
    a = E.map_to_curves(E.transpose_map(E.make_map_data(ds, E.Selection("s21", x="field",
                                                                          y="rf_freq"))))
    b = E.map_to_curves(E.make_map_data(ds, E.Selection("s21", x="rf_freq", y="field")))
    for ca, cb in zip(a, b):
        np.testing.assert_allclose(ca.z, cb.z)
        assert ca.held == cb.held and ca.label == cb.label


def test_a_reference_turns_with_the_map():
    ds = _ds()
    style = E.MapStyle(ref="column", ref_i=-1)        # X = field: the column at 100 mT
    m = E.make_map_data(ds, E.Selection("s21", x="field", y="rf_freq"), style)
    t = E.transpose_map(m)
    assert t.ref["mode"] == "row"                     # turned: the ROW at 100 mT
    np.testing.assert_allclose(t.z[-1], 1.0)          # 100 mT divided by itself
