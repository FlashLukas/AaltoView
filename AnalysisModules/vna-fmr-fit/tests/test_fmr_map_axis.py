"""A map sent to the VNA-FMR fit: the map's Y axis becomes the fit's x axis
(his rule, 2026-10-09) -- one sweep per X value, along Y. Many sweeps: the
fit asks first and offers fewer. Simulated Kittel map: what goes in comes out."""

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


def test_y_axis_of_the_map_is_the_fits_x_axis(win):
    """X = field, Y = rf_freq: frequency sweeps, one per field."""
    m = E.make_map_data(_ds(), E.Selection("s21", x="field", y="rf_freq"))
    win.add_maps([m])
    assert len(win.entries) == len(FIELDS)
    for e, b in zip(win.entries, FIELDS):
        assert e.curve.x_name == "rf_freq" and e.curve.x.size == FREQ.size
        assert e.curve.held["field"] == (b, "mT")
    assert "sweeps along rf_freq (the map's Y), one per field" in win.status.text()
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(2))           # 80 mT
    win.fit()
    assert win.current().result.values["p1_center"] == pytest.approx(kittel(80.0), abs=1e-3)


def test_many_sweeps_ask_first_and_take_fewer(win):
    """X = rf_freq, Y = field: 1201 field sweeps -- asked, 10 between 3 and 5 GHz taken."""
    m = E.make_map_data(_ds(), E.Selection("s21", x="rf_freq", y="field"))
    asked = []

    def ask(t):
        asked.append(t.y.size)
        return E.pick_rows(t.y, 10, 3.0, 5.0)

    win.ask_fewer = ask
    win.add_maps([m])
    assert asked == [FREQ.size]
    assert len(win.entries) == 10
    fs = [e.curve.held["rf_freq"][0] for e in win.entries]
    assert min(fs) >= 3.0 and max(fs) <= 5.0
    assert all(e.curve.x_name == "field" and e.curve.x.size == FIELDS.size for e in win.entries)
    # the values are the map's: the column at that frequency
    k = int(np.argmin(np.abs(FREQ - fs[0])))
    np.testing.assert_allclose(win.entries[0].curve.z, m.z[:, k])


def test_cancel_takes_nothing(win):
    m = E.make_map_data(_ds(), E.Selection("s21", x="rf_freq", y="field"))
    win.ask_fewer = lambda t: None
    win.add_maps([m])
    assert win.entries == [] and "cancelled" in win.status.text()


def test_the_dialog_returns_what_it_shows(win):
    from fmr_fit.app import FewerSweeps
    t = E.transpose_map(E.make_map_data(_ds(), E.Selection("s21", x="rf_freq", y="field")))
    d = FewerSweeps(win, t)
    d.lo.setText("3"); d.hi.setText("5"); d.count.setValue(10)
    np.testing.assert_array_equal(d.rows(), E.pick_rows(t.y, 10, 3.0, 5.0))
    assert "10 sweeps" in d.info.text()
    d.all.click()
    assert d.rows().size == FREQ.size


def test_a_reference_turns_with_the_map():
    m = E.make_map_data(_ds(), E.Selection("s21", x="field", y="rf_freq"),
                        E.MapStyle(ref="column", ref_i=-1))       # the column at 100 mT
    t = E.transpose_map(m)
    assert t.ref["mode"] == "row"
    np.testing.assert_allclose(t.z[-1], 1.0)                      # 100 mT / itself
