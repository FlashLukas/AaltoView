"""The window, the way it is used: a map in, FFT, peaks, the field typed in,
the dispersion fitted, everything exported."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from aaltoview import export as E  # noqa: E402
from aaltoview.data import load  # noqa: E402
from aaltoview.view import Slice  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_data as DEMO  # noqa: E402


@pytest.fixture(scope="module")
def stripe_map(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    path = next(out.glob("*/*_py_stripe_trmoke.nc"))
    ds = load(path).load()
    return ds, path, E.make_map_data(ds, E.Selection("lockin", x="pos_x", y="rf_freq",
                                                     slices={"pos_y": Slice("at", 0)}),
                                     source=path)


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from sw_fft.app import FFTWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FFTWindow()
    yield w
    w.close()


def test_map_in_fft_peaks_fit_out(win, stripe_map, tmp_path):
    ds, path, m = stripe_map
    win.add_maps([m])
    assert win.spectrum is not None and win.spectrum.F.shape[0] == 81
    assert "resolution" in win.status.text()
    win.side.setCurrentIndex(win.side.findData("positive"))
    win.kmin.setText("0.5")
    win.find_peaks()
    assert len(win.peaks) == 81 and len(win.points) == 81
    # f from the lines' rf_freq (MHz -> GHz); the field is not an axis: typed
    assert win.f_src.currentData() == "y" and win.b_src.currentData() == "constant"
    assert win.points[0].f == pytest.approx(5.0)
    win.b_const.setText("20")
    win._sources_changed()
    assert win.points[0].B == pytest.approx(20.0)
    for n in ("A", "d", "w"):
        win.specs[n].value = DEMO.STRIPE[n]
    win.specs["Ms"].value, win.specs["Ms"].vary = 700.0, True
    win.fit()
    assert win.result is not None
    assert win.result.values["Ms"] == pytest.approx(DEMO.STRIPE["Ms"], rel=5e-3)
    assert "fitted" in win.status.text()
    assert win.model_items                              # the fit drawn on the FFT map
    # 1/um: the same peaks, divided by 2 pi -- and the points stay in rad/um
    win.k_unit.setCurrentIndex(win.k_unit.findData("1/um"))
    assert win.peaks[0].k == pytest.approx(win.points[0].k / (2 * np.pi), rel=1e-6)
    assert win.points[0].k == pytest.approx(DEMO.stripe_k(5.0), abs=0.03)
    # out
    mp = win.fft_map()
    E.write_map(tmp_path / "fft.csv", mp)
    text = (tmp_path / "fft.csv").read_text(encoding="utf-8-sig")
    assert text.startswith("rf_freq (MHz) \\ 1/λ (1/µm)")
    from sw_fft import results as R
    R.write_columns(tmp_path / "peaks.csv", win.peak_columns())
    R.write_columns(tmp_path / "res.dat", win.result_columns())
    head = (tmp_path / "peaks.csv").read_text(encoding="utf-8-sig").splitlines()
    assert head[0].startswith("rf_freq,1/λ,|1/λ|,wavelength") and len(head) == 3 + 81
    assert "Ms" in (tmp_path / "res.dat").read_text(encoding="utf-8")
    E.save_figure(win.figure(), tmp_path / "fft.png")
    from sw_fft.results import figure_dispersion
    E.save_figure(figure_dispersion(win.points, win.result), tmp_path / "disp.png")
    assert (tmp_path / "fft.png").stat().st_size > 10000


def test_curves_are_stacked_into_one_input(win, stripe_map):
    ds, path, m = stripe_map
    curves = E.map_to_curves(m)[:5]
    win.add_curves(curves)
    assert len(win.inputs) == 1 and win.inputs[0].rows.shape == (5, 201)
    assert win.inputs[0].y_name == "rf_freq"
    win.find_peaks()
    assert {p.line for p in win.peaks} == set(range(5))


def test_a_real_input_offers_no_imaginary_part(win, stripe_map):
    ds, path, m = stripe_map
    real = E.make_map_data(ds, E.Selection("lockin", x="pos_x", y="rf_freq", part="real",
                                           slices={"pos_y": Slice("at", 0)}))
    real.z = None
    win.add_maps([real])
    assert win.part.currentData() == "real"
    assert not win.part.model().item(win.part.findData("imag")).isEnabled()


def test_width_modes_can_be_switched_off(win):
    from PySide6 import QtCore
    from sw_fft import waveguide as W
    win.pinning.setCurrentIndex(win.pinning.findData("none"))
    for r, n in enumerate(W.PARAMS):
        enabled = bool(win.mtable.item(r, 4).flags() & QtCore.Qt.ItemIsEnabled)
        assert enabled == (n not in W.WIDTH_PARAMS), n
    win.pinning.setCurrentIndex(win.pinning.findData("guslienko"))
    assert win.mtable.item(list(W.PARAMS).index("w"), 4).flags() & QtCore.Qt.ItemIsEnabled


def test_tr_moke_unfold_in_the_window(win, stripe_map):
    """The stripe map as a TR-MOKE with an 80 MHz laser would record it: lines
    with a negative alias conjugated. Unfold (80 MHz) puts every peak at +k."""
    import copy
    from sw_fft import fft as F
    ds, path, m = stripe_map
    seen = copy.deepcopy(m)
    d = F.alias(seen.y / 1000.0, 80.0)                           # rf_freq in MHz
    seen.z = np.where((d < 0)[:, None], np.conj(m.z), m.z)
    win.add_maps([seen])
    win.kmin.setText("0.5")
    win.find_peaks()
    assert any(p.k < 0 for p in win.peaks)                       # the dashed V
    win.unfold.setCurrentIndex(win.unfold.findData(80.0))
    assert all(p.k > 0 for p in win.peaks)
    assert "unfolded for 80 MHz" in win.status.text()
    win.unfold_invert.setChecked(True)
    # every line that has a direction goes to -k; on a harmonic or half-way
    # (alias 0 or 40 MHz: 2 lines in 8 at 50 MHz steps) there is none to flip
    has_sign = (np.abs(d) > 0) & (np.abs(d) < 40)
    assert sorted(p.line for p in win.peaks if p.k < 0) == list(np.flatnonzero(has_sign))
    # no frequency per line: said, not guessed
    win.unfold_invert.setChecked(False)
    win.f_src.setCurrentIndex(win.f_src.findData("constant"))
    assert "unfold OFF" in win.status.text()
