"""render_docs.py -- the README screenshots, offscreen and reproducible.

    python tools/render_docs.py            # writes docs/*.png

Generates the fake FMR measurements (tools/make_demo_data.py) into a temporary
folder, opens the viewer on them without showing a window, poses it, and grabs
the window. Re-run after any change to the GUI, so the pictures never go stale.

Three things the offscreen platform gets wrong unless told (learnt the hard way
in the AaltoFlow suite, docs/DEVELOPER_NOTES.md section 9):
* its font database is EMPTY on Windows -> every label is a box: QT_QPA_FONTDIR;
* app.setStyle("Fusion") re-polishes and drops a font pinned earlier -> pin it
  again after the widgets exist;
* timers and paints need the event loop -> pump it before grabbing.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
sys.path.insert(0, str(HERE))

from PySide6 import QtGui, QtWidgets  # noqa: E402

import make_demo_data  # noqa: E402
from aaltoview.apps import theme  # noqa: E402

SIZE = (1640, 980)


def pump(app, seconds: float):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def pin_font(app):
    base = QtGui.QFont("Segoe UI", 9)
    app.setFont(base)
    for w in app.allWidgets():
        f = w.font()
        f.setFamily(base.family())
        w.setFont(f)


def shot(app, name: str, theme_name: str, pose):
    """Build a fresh window in `theme_name`, pose it, save docs/<name>.png."""
    from aaltoview.apps import viewer as VW
    theme.set_theme(theme_name)
    VW.configure_pyqtgraph()
    theme.apply(app)
    win = VW.ViewerWindow(folder=DATA)
    win.resize(*SIZE)
    win.show()
    pump(app, 0.3)
    pin_font(app)
    win.viewer.browser.refresh()        # column widths measured with the real font
    pose(win.viewer)
    pin_font(app)
    win.resize(*SIZE)
    pump(app, 1.5)
    DOCS.mkdir(exist_ok=True)
    out = DOCS / f"{name}.png"
    win.grab().save(str(out))
    print(f"  {out.relative_to(HERE.parent)}")
    win.close()
    win.deleteLater()
    pump(app, 0.2)


def select_file(v, stem: str):
    tree = v.browser.tree
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        if Path(item.data(0, 0x0100)).stem == stem:
            tree.setCurrentItem(item)
            v.load_file(item.data(0, 0x0100))
            return
    raise SystemExit(f"no demo file {stem}")


def row(controls, dim):
    return next(r for r in controls.rows if r.dim == dim)


# ─────────────────────────────── the poses ────────────────────────────────────

def pose_map(v):
    """The 3-D cube: field x frequency at 2 um from the antenna, cursor on the line."""
    select_file(v, "101530_fmr_distance_cube")
    m = v.map
    m.controls.x_combo.setCurrentText("field")
    m.controls.y_combo.setCurrentText("rf_freq")
    row(m.controls, "distance").slider.setValue(1)
    m.refresh()
    m._place_cursor(40, 58)                 # 100 mT, 9.25 GHz: on the Kittel line
    m._cut("row")
    v.tabs.setCurrentWidget(m)


def pose_spinwave(v):
    """Re(signal) of the spin-wave image, red-blue and symmetric, 75 mT."""
    select_file(v, "094500_spinwave_image_8GHz")
    m = v.map
    m.controls.part_combo.setCurrentText("Re z")
    m.controls.x_combo.setCurrentText("pos_x")
    m.controls.y_combo.setCurrentText("pos_y")
    row(m.controls, "field").slider.setValue(1)
    m.cmap_combo.setCurrentText("red-blue")
    m.symmetric.setChecked(True)
    m.refresh()
    m._place_cursor(20, 40)
    v.tabs.setCurrentWidget(m)


def pose_curves(v):
    """Spectra at 100 mT for every distance, from the cube, plus the same spectrum
    from ANOTHER file on another frequency grid -- curves remember their file."""
    select_file(v, "101530_fmr_distance_cube")
    lines = v.lines
    lines.controls.x_combo.setCurrentText("rf_freq")
    row(lines.controls, "field").slider.setValue(40)            # 100 mT
    lines.along_combo.setCurrentText("distance")
    for i in range(lines.values.count()):
        lines.values.item(i).setSelected(True)
    lines.add_selected()
    # a second measurement on a different frequency grid, same axis
    select_file(v, "153012_fmr_field_freq")
    lines.controls.x_combo.setCurrentText("rf_freq")
    row(lines.controls, "field").slider.setValue(50)            # 100 mT
    lines.add_current()
    item = lines.table.topLevelItem(lines.table.topLevelItemCount() - 1)
    item.setText(0, "field = 100 mT (separate map)")            # labels are editable
    lines.show_preview.setChecked(False)
    lines.redraw()
    v.tabs.setCurrentWidget(lines)


def pose_normalised(v):
    """Field sweeps at four frequencies, peak-normalised and stacked (light theme)."""
    select_file(v, "131205_field_sweeps")
    lines = v.lines
    lines.controls.x_combo.setCurrentText("field")
    lines.along_combo.setCurrentText("rf_freq")
    for i in range(lines.values.count()):
        lines.values.item(i).setSelected(True)
    lines.add_selected()
    lines.norm_combo.setCurrentIndex(list(lines.norm_combo.itemData(i)
                                          for i in range(lines.norm_combo.count())).index("peak"))
    lines.offset_edit.setText("1.2")
    lines.show_preview.setChecked(False)
    lines.redraw()
    v.tabs.setCurrentWidget(lines)


def pose_map_rows(v):
    """The 2-D map with every frequency line normalised to its peak (light theme)."""
    select_file(v, "153012_fmr_field_freq")
    m = v.map
    m.controls.x_combo.setCurrentText("field")
    m.controls.y_combo.setCurrentText("rf_freq")
    m.cmap_combo.setCurrentText("viridis")
    m.norm_combo.setCurrentIndex(1)                             # each row
    m.refresh()
    v.tabs.setCurrentWidget(m)


def main() -> int:
    global DATA
    # A NEUTRAL folder: its path is on screen, and a published picture must not
    # carry the user name of the PC that rendered it (the old renders showed
    # C:\Users\<account>\AppData\...). C:\Users\Public names nobody.
    import shutil
    base = (Path(r"C:\Users\Public\Documents\AaltoView") if sys.platform == "win32"
            else Path(tempfile.gettempdir()) / "AaltoView")
    shutil.rmtree(base / "data", ignore_errors=True)
    DATA = base / "data"
    make_demo_data.main([str(DATA)])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shot(app, "map", "dark", pose_map)
    shot(app, "spinwave", "dark", pose_spinwave)
    shot(app, "curves", "dark", pose_curves)
    shot(app, "curves-normalised-light", "light", pose_normalised)
    shot(app, "map-rows-light", "light", pose_map_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
