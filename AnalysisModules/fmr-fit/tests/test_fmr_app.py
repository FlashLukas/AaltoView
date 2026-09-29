"""The FMR fit window, offscreen: curves in, fit, results out."""

import os

import numpy as np
import pytest

pytest.importorskip("lmfit")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from aaltoview.export import Curve, Selection  # noqa: E402
from fmr_fit import model as M  # noqa: E402


def _curve(center, hand=1, complex_=True, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 200, 401)
    vals = dict(p1_center=center, p1_hwhm=3.0, p1_amp=1.0, p1_phase=20.0,
                bg_re=2.0, bg_im=0.5, slope_re=0.0, slope_im=0.0)
    z = M.evaluate(vals, x, M.Setup(), hand, 100.0)
    z = z + 0.02 * (rng.normal(size=x.size) + 1j * rng.normal(size=x.size))
    return Curve(x=x, y=np.abs(z), label=f"at {center}", x_name="field", x_unit="mT",
                 y_name="|s21|", y_unit="V", selection=Selection("s21", x="field"),
                 source=None, z=z if complex_ else None, held={"rf_freq": (center / 10, "GHz")})


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow()
    yield w
    w.close()


def test_curves_arrive_with_a_guess_on_screen(win):
    win.add_curves([_curve(80.0), _curve(120.0, seed=1)])
    assert win.curve_list.topLevelItemCount() == 2
    e = win.current()
    assert e.setup.mode == "complex" and e.start is not None
    assert win.table.rowCount() == len(M.param_names(e.setup))
    assert win.fit_btn.isEnabled()


def test_fit_then_fit_all_fill_the_results(win):
    win.add_curves([_curve(80.0), _curve(120.0, hand=-1, seed=1)])
    win.fit()
    e = win.current()
    assert e.result.values["p1_center"] == pytest.approx(80.0, abs=0.2)
    assert "x0 = " in win.curve_list.topLevelItem(0).text(1)
    win.fit_all()
    assert all(e.result is not None for e in win.entries)
    assert win.entries[1].result.hand == -1
    assert win.results.rowCount() == 2
    assert win.results.horizontalHeaderItem(0).text().startswith("rf_freq")


def test_a_fixed_value_typed_in_the_table_is_used(win):
    win.add_curves([_curve(80.0)])
    names = M.param_names(win.current().setup)
    r = names.index("p1_hwhm")
    win.table.item(r, 1).setText("4.5")
    win.table.item(r, 4).setCheckState(QtCore.Qt.Checked)      # Fixed
    win.fit()
    assert win.current().result.values["p1_hwhm"] == 4.5


def test_a_real_curve_is_fitted_as_one_channel(win):
    win.add_curves([_curve(80.0, complex_=False)])
    assert win.current().setup.mode == "real"
    win.fit()
    assert win.current().result.values["p1_center"] == pytest.approx(80.0, abs=1.0)


def test_more_peaks_add_rows_and_keep_the_first(win):
    win.add_curves([_curve(80.0)])
    before = win.current().start["p1_center"].value
    win.peaks.setValue(2)
    assert "p2_center" in win.current().start
    assert win.current().start["p1_center"].value == before


def test_the_results_leave_as_a_file(win, tmp_path):
    win.add_curves([_curve(80.0)])
    with pytest.raises(ValueError, match="nothing fitted"):
        win.columns()
    win.fit()
    p = M.write_results(tmp_path / "r.dat", win.columns())
    assert p.read_text(encoding="utf-8").splitlines()[0].startswith("rf_freq\t")


def test_fit_all_from_the_start_on_screen_finds_the_other_hand(win):
    """The start guessed on screen assumes hand +1; with hand on auto the -1
    fit started from the same phase and collapsed into a zero-width spike
    (6 GHz demo sweep, found in the first screenshot, 2026-09-29)."""
    win.add_curves([_curve(60.0, hand=-1), _curve(120.0, hand=-1, seed=1)])
    win.fit_all()                        # the first one fits from the on-screen start
    for e in win.entries:
        r = e.result
        assert r.hand == -1 and not M.suspicious(r)
        assert r.values["p1_hwhm"] == pytest.approx(3.0, rel=0.05)


def _drag(win, lo, hi):
    win.region.setRegion((lo, hi))
    win._range_dragged()


def test_a_dragged_range_follows_the_peak_to_the_other_curves(win):
    """Before 2026-09-29 Fit all gave EVERY curve the field values of the band
    drawn on the current one -- a peak elsewhere fell outside its own range."""
    win.add_curves([_curve(50.0), _curve(150.0, seed=1), _curve(100.0, seed=2)])
    _drag(win, 35.0, 70.0)                    # around 50 mT: x0 -5 ... +6.7 HWHM
    win.fit_all()
    first, second, third = win.entries
    assert first.result.xrange == pytest.approx((35.0, 70.0), abs=0.5)
    lo, hi = second.result.xrange
    assert lo < 150.0 < hi and hi - lo == pytest.approx(35.0, abs=2.0)
    assert second.result.values["p1_center"] == pytest.approx(150.0, abs=0.3)
    # a curve shown next gets the carried band on screen too
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(2))
    lo, hi = win.region.getRegion()
    assert lo < 100.0 < hi
    assert "following the peak" in win.status.text()


def test_own_bands_stay_and_the_rule_can_be_changed(win):
    win.add_curves([_curve(50.0), _curve(150.0, seed=1)])
    _drag(win, 35.0, 70.0)
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(1))
    _drag(win, 120.0, 190.0)                  # the second curve gets its own band
    win.rule_combo.setCurrentIndex(win.rule_combo.findData("same"))
    win.fit_all()
    assert win.entries[0].result.xrange == pytest.approx((35.0, 70.0), abs=0.5)
    assert win.entries[1].result.xrange == pytest.approx((120.0, 190.0), abs=0.5)
    win.full_range()                          # on the second: back to everything
    win.fit_all()
    assert win.entries[1].result.xrange == pytest.approx((0.0, 200.0))


def test_fit_all_fits_only_the_selected_curves(win):
    """Several selected -> only those (asked for 2026-09-29, with 141 map rows)."""
    win.add_curves([_curve(50.0), _curve(100.0, seed=1), _curve(150.0, seed=2)])
    win.curve_list.clearSelection()
    win.curve_list.topLevelItem(1).setSelected(True)
    win.curve_list.topLevelItem(2).setSelected(True)
    assert win.fit_all_btn.text() == "Fit selected (2)"
    win.fit_all()
    assert [e.result is not None for e in win.entries] == [False, True, True]
    assert "fitted 2 of 2 selected" in win.status.text()
    win.curve_list.clearSelection()
    assert win.fit_all_btn.text() == "Fit all"
