"""The VNA-FMR fit window, offscreen: curves in, fit, results out."""

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


def _vna_curves(tmp_path):
    import sys
    from pathlib import Path
    from aaltoview import export as E
    from aaltoview.data import load
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import make_demo_data as DEMO
    DEMO.main([str(tmp_path)])
    path = next(tmp_path.glob("*/*_vna_freq_sweeps.nc"))
    ds = load(path).load()
    return DEMO, E.curves_along(ds, E.Selection("s21", x="rf_freq"), "field", range(6), path)


def test_a_reference_and_the_oscillator_on_vna_sweeps(win, tmp_path):
    DEMO, curves = _vna_curves(tmp_path)
    win.add_curves(curves)                                   # the last one: 500 mT
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(2))    # 60 mT
    win.ref_combo.setCurrentIndex(win.ref_combo.findText(curves[-1].label))
    win.lineshape.setCurrentIndex(win.lineshape.findData("oscillator"))
    win.baseline.setCurrentText("constant")
    e = win.current()
    assert e.reference[0] is win.entries[-1] and e.reference[1] == "divide"
    f0, w = DEMO.kittel(60.0), DEMO.linewidth(60.0) / 2
    win.region.setRegion((f0 - 10 * w, f0 + 10 * w))
    win._range_dragged()
    win.fit_all()
    assert win.entries[-1].result is None          # the reference is not fitted
    r = e.result
    assert abs(r.values["p1_center"] - f0) < 4 * r.errors["p1_center"] + 1e-3
    assert abs(r.values["p1_hwhm"] - w) < 4 * r.errors["p1_hwhm"]
    for other in win.entries[:4]:                  # the range followed each peak
        assert other.result is not None and other.reference is e.reference


def test_derivative_divide_and_delay_from_the_window(win, tmp_path):
    DEMO, curves = _vna_curves(tmp_path)
    win.add_curves(curves[2:3])
    win.lineshape.setCurrentIndex(win.lineshape.findData("oscillator"))
    win.delay.setChecked(True)
    win.dd.setChecked(True)
    win.dd_k.setValue(10)
    e = win.current()
    assert e.setup.dd == 10 and e.setup.delay and "delay" in e.start
    f0, w = DEMO.kittel(60.0), DEMO.linewidth(60.0) / 2
    win.region.setRegion((f0 - 10 * w, f0 + 10 * w))
    win._range_dragged()
    win.fit()
    assert e.result.values["p1_center"] == pytest.approx(f0, abs=0.05)
    names = [win.table.item(r, 0).text() for r in range(win.table.rowCount())]
    assert "electrical delay τ" in names
    units = {win.table.item(r, 0).text(): win.table.item(r, 3).text()
             for r in range(win.table.rowCount())}
    assert units["electrical delay τ"] == "ns" and units["peak 1: amplitude A"] == ""
    # a real channel: no delay, no derivative-divide
    win.mode.setCurrentIndex(win.mode.findData("real"))
    assert not win.dd.isEnabled() and win.current().setup.dd == 0


def test_the_settings_go_to_the_next_curve(win):
    """Asked for 2026-09-29: set up and fit one sweep, go to the next -- same
    settings. A curve already set up or fitted keeps its own."""
    win.add_curves([_curve(60.0), _curve(100.0, seed=1), _curve(140.0, seed=2)])
    win.peaks.setValue(2)
    win.baseline.setCurrentText("constant")
    win.fit()
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(1))
    e1 = win.entries[1]
    assert e1.setup.n_peaks == 2 and e1.setup.baseline == "constant"
    assert win.peaks.value() == 2                      # the controls show it
    # the second curve set differently stays so; the third takes the newest
    win.peaks.setValue(1)
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(0))
    assert win.entries[0].setup.n_peaks == 2           # fitted: kept
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(2))
    assert win.entries[2].setup.n_peaks == 1


def test_a_frequency_sweep_gets_the_delay_and_is_drawn_without_it(win, tmp_path):
    """Raw, a 3.2 ns delay spins Re and Im through ~60 turns and the sweep
    looked like noise (screenshot, 2026-09-29): a frequency sweep whose phase
    winds gets the delay option and the oscillator lineshape, and is drawn
    with the delay taken out."""
    from fmr_fit import model as M
    DEMO, curves = _vna_curves(tmp_path)
    win.add_curves(curves[2:3])                        # 60 mT
    e = win.current()
    assert e.setup.lineshape == "oscillator" and e.setup.delay
    assert "electrical delay on" in win.status.text()
    assert "delay" in e.start and e.start["delay"].value == pytest.approx(3.2, abs=0.05)
    drawn = next(it for _, it in win._items if it.name() == "Re data")
    x, re = drawn.getData()
    im = next(it for _, it in win._items if it.name() == "Im data").getData()[1]
    assert M.phase_turns(x, re + 1j * im) < 2          # was ~60 raw
    assert "e^{+i2πτ" in win.plot.getAxis("left").labelText


def test_long_sweeps_draw_every_kth_symbol_and_zooming_brings_them_back(win):
    """At most MAX_SYMBOLS symbols across the plot (asked for 2026-09-30)."""
    from fmr_fit.app import MAX_SYMBOLS, thin_step
    x = np.linspace(0, 200, 10001)
    z = M.evaluate(dict(p1_center=100.0, p1_hwhm=0.5, p1_amp=1.0, p1_phase=0.0,
                        bg_re=0.0, bg_im=0.0, slope_re=0.0, slope_im=0.0),
                   x, M.Setup(), 1, 100.0)
    c = Curve(x=x, y=np.abs(z), label="long", x_name="field", x_unit="mT", y_name="s",
              y_unit="V", selection=Selection("s", x="field"), z=z)
    win.add_curves([c])
    data = [it for _, it in win._items if getattr(it, "thinned", False)]
    assert data and all(it.opts["symbol"] == "o" for it in data)   # still symbols
    win.plot.setXRange(0, 200, padding=0)
    k = data[0].opts["downsample"]
    assert k == thin_step(x, 0, 200) and 10001 / k <= MAX_SYMBOLS + 1
    win.plot.setXRange(99, 101, padding=0)                          # zoom in: all
    assert data[0].opts["downsample"] == 1
    assert thin_step(x[:100], 0, 200) == 1                          # short: all
