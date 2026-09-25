"""The data viewer's window, offscreen: the things that are easy to break.

The arithmetic and the files are tested in test_export.py. Here: the
file list opens a file, the controls follow the data, a cut on the map lands in
1D plots, "Add selected" makes one curve per value, and the exports produce
files from what is on screen.
"""

import os

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402


def _cube() -> xr.Dataset:
    f = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 20.0])
    x = np.array([0.0, 1.0])
    vals = f[:, None, None] * 100 + y[None, :, None] + x[None, None, :]
    return xr.Dataset(
        {"kerr": (("freq", "y", "x"), vals, {"units": "mdeg"}),
         "refl": (("freq", "y", "x"), vals * 0 + 1.0, {"units": "V"})},
        coords={"freq": ("freq", f, {"units": "GHz"}),
                "y": ("y", y, {"units": "um"}),
                "x": ("x", x, {"units": "um"})},
        attrs={"name": "cube", "dims": "freq,y,x"})


@pytest.fixture
def viewer(tmp_path):
    from aaltoview.apps import viewer as VW
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    day = tmp_path / "data" / "2026-09-16"
    day.mkdir(parents=True)
    _cube().to_netcdf(day / "120000_cube.nc", engine="h5netcdf")
    w = VW.ViewerWidget(tmp_path / "data")
    w.browser.refresh()
    yield w
    w.close()


def _open_first(viewer):
    item = viewer.browser.tree.topLevelItem(0)
    viewer.browser._activated(item)


def test_the_folder_is_listed_and_a_double_click_opens_the_file(viewer):
    assert viewer.browser.tree.topLevelItemCount() == 1
    item = viewer.browser.tree.topLevelItem(0)
    assert item.text(1) == "2026-09-16 12:00:00"
    viewer.browser._show_header(item)
    assert "freq  (4)" in viewer.browser.info.toPlainText()
    _open_first(viewer)
    assert viewer.ds is not None and viewer.path.name == "120000_cube.nc"


def test_the_map_controls_are_built_from_the_file(viewer):
    _open_first(viewer)
    c = viewer.map.controls
    assert (c.x_combo.currentText(), c.y_combo.currentText()) == ("x", "y")
    assert [r.dim for r in c.rows] == ["freq"]
    c.rows[0].slider.setValue(2)
    assert viewer.map._map.z[0].tolist() == [300.0, 301.0]
    c.x_combo.setCurrentText("freq")                     # a different question
    assert [r.dim for r in c.rows] == ["x"]
    assert viewer.map._map.z.shape == (3, 4)


def test_switching_detector_keeps_the_chosen_axes(viewer):
    _open_first(viewer)
    c = viewer.map.controls
    c.x_combo.setCurrentText("freq")
    c.y_combo.setCurrentText("x")
    c.det_combo.setCurrentText("refl")
    assert (c.x_combo.currentText(), c.y_combo.currentText()) == ("freq", "x")


def test_a_cut_through_the_cursor_lands_in_1d_plots(viewer):
    _open_first(viewer)
    m = viewer.map
    m.controls.rows[0].slider.setValue(1)                 # freq = 2 GHz
    m._place_cursor(1, 2)                                 # x = 1, y = 20
    m._cut("row")
    m._cut("column")
    row, col = viewer.lines.curves
    assert row.y.tolist() == [220.0, 221.0]               # along x at y = 20
    assert col.y.tolist() == [201.0, 211.0, 221.0]        # along y at x = 1
    assert row.label == "freq = 2 GHz, y = 20 um"
    assert viewer.lines.table.topLevelItemCount() == 2


def test_add_selected_makes_one_curve_per_value(viewer):
    _open_first(viewer)
    lines = viewer.lines
    lines.controls.x_combo.setCurrentText("y")
    lines.along_combo.setCurrentText("freq")
    assert lines.values.count() == 4
    for i in (0, 3):
        lines.values.item(i).setSelected(True)
    lines.add_selected()
    assert [c.label for c in lines.curves] == ["freq = 1 GHz, x = 0 um",
                                               "freq = 4 GHz, x = 0 um"]
    # untick one: it leaves the plot and the export
    item = lines.table.topLevelItem(0)
    from PySide6 import QtCore
    item.setCheckState(0, QtCore.Qt.Unchecked)
    assert [c.visible for c in lines.curves] == [False, True]
    assert len(lines._items) == 1


def test_exports_write_what_is_on_screen(viewer, tmp_path):
    _open_first(viewer)
    viewer.map.controls.rows[0].mode.setCurrentText("mean all")
    p = viewer.map.write_data(tmp_path / "m.dat", "matrix")
    rows = p.read_text(encoding="utf-8").splitlines()
    assert rows[1].split("\t")[1:] == ["250.0", "251.0"]  # the mean over freq
    kw = viewer.map.notebook_kwargs()
    assert kw["m"].source.endswith("120000_cube.nc")

    viewer.lines.add_current()
    viewer.lines.norm_combo.setCurrentIndex(list(
        __import__("aaltoview.export", fromlist=["NORMS"]).NORMS).index("peak"))
    p = viewer.lines.write_data(tmp_path / "c.csv", "columns")
    last = p.read_text(encoding="utf-8").splitlines()[-1]
    assert last.split(",")[1] == "1.0"
    fig = viewer.lines.figure()
    assert fig.axes[0].get_lines()


def test_exporting_nothing_says_so_instead_of_raising(viewer):
    viewer.lines.copy_data()
    assert "no curves yet" in viewer.lines.status.text()


def test_a_one_dimensional_scan_opens_on_1d_plots(viewer, tmp_path):
    ds = xr.Dataset({"r": (("field",), np.arange(5.0))},
                    coords={"field": ("field", np.linspace(0, 40, 5), {"units": "mT"})})
    viewer.set_dataset(ds, tmp_path / "line.nc")
    assert viewer.tabs.currentWidget() is viewer.lines
    assert "one dimension" in viewer.map.status.text()


def test_dragging_the_colour_bar_stays_on_the_scale_of_the_data(tmp_path):
    """pyqtgraph rounds dragged levels to multiples of `rounding` (default 1):
    on a 0.004..0.014 V map one nudge of a handle threw the range out to whole
    volts (found on the rig, 2026-09-24)."""
    from aaltoview.apps import viewer as VW
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    day = tmp_path / "data" / "2026-09-24"
    day.mkdir(parents=True)
    ds = _cube()
    ds["kerr"] = ds["kerr"] * 1e-5 + 0.004            # ~0.005 .. 0.008
    ds.to_netcdf(day / "120000_small.nc", engine="h5netcdf")
    w = VW.ViewerWidget(tmp_path / "data")
    try:
        w.browser.refresh()
        _open_first(w)
        lo, hi = w.map.cbar.levels()
        w.map.cbar.region.setRegion((70, 191))        # nudge the lower handle (rest: 63)
        w.map.cbar._regionChanged()                   # release
        new_lo, new_hi = w.map.cbar.levels()
        assert lo < new_lo < hi and new_hi == pytest.approx(hi, rel=1e-3)
        assert not w.map.auto.isChecked()
    finally:
        w.close()


def test_the_viewer_has_its_own_window_icon():
    """It had none: the suite's apply_window_icon never reached this repo, so the
    viewer showed up as a generic Python window (found 2026-09-24)."""
    from aaltoview.apps import theme
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert theme.ICON_FILE.exists()
    # well-formed XML: a "--" inside an SVG comment makes Qt drop the whole
    # file SILENTLY -- the icon is simply null (it happened while writing this)
    import xml.etree.ElementTree as ET
    ET.parse(theme.ICON_FILE)
    theme.apply_window_icon(app)
    assert not app.windowIcon().isNull()
    assert not app.windowIcon().pixmap(32, 32).isNull()   # the SVG actually renders
