"""case_study_fmr_pssw.py -- the worked example of docs/CASE_STUDY_FMR_PSSW.md.

    uv run --all-packages python tools/case_study_fmr_pssw.py     # docs/case-study/*.png

200 nm YIG measured the lab's way (tools/make_demo_data.py file 9: field set,
VNA S21 swept 2-18 GHz, a reference at 800 mT), taken through the windows
exactly as an operator would, headless:

  1. the raw map: the cables, not the sample;
  2. background subtraction: every line divided by the 800 mT reference;
  3. three sweeps fitted by hand (100, 200, 300 mT), uniform mode + PSSW 1-4;
  4. a rough dispersion from those 15 resonances;
  5. "Predict + fit the others": the dispersion places, bounds and labels the
     peaks of the other 9 sweeps -- and leaves out a mode that is not in a sweep;
  6. the final dispersion through ONE exchange stiffness A.

Every number the case study quotes is printed here, next to the truth the data
was made from, and saved to docs/case-study/numbers.txt; a screenshot of each
step goes to docs/case-study/. Re-run after a change to the module.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "case-study"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

import make_demo_data as DEMO  # noqa: E402
import render_docs as R  # noqa: E402
from aaltoview import export as E  # noqa: E402
from aaltoview.apps import theme  # noqa: E402
from aaltoview.data import load  # noqa: E402

BY_HAND = (100.0, 200.0, 300.0)       # mT: the three sweeps fitted by hand
LINES: list[str] = []


def say(text: str = ""):
    print(text)
    LINES.append(text)


def truth(b: float) -> dict[int, float]:
    """f0 (GHz) of the uniform mode (0) and PSSW n = 1-4 at field b."""
    return {n: float(DEMO.yig_mode(np.array([b]), n)[0][0]) for n in range(5)}


def grab(app, win, name: str):
    R.pin_font(app)
    win.resize(*R.SIZE)
    R.pump(app, 1.5)
    out = OUT / f"{name}.png"
    win.grab().save(str(out))
    print(f"  -> {out.relative_to(HERE.parent)}")


def zoom(app, win, lo: float, hi: float):
    """The lines are ~12 MHz wide in a 16 GHz sweep: show the part with peaks."""
    R.pump(app, 0.3)
    for plot in (win.plot, win.rplot):
        plot.getViewBox().setAutoVisible(y=True)
        plot.setXRange(lo, hi, padding=0)
        plot.enableAutoRange(axis="y")


def select(win, fields):
    win.curve_list.clearSelection()
    for i, e in enumerate(win.entries):
        if e.curve.held["field"][0] in fields:
            win.curve_list.topLevelItem(i).setSelected(True)


def show(win, field: float):
    for i, e in enumerate(win.entries):
        if e.curve.held["field"][0] == field:
            win.curve_list.setCurrentItem(win.curve_list.topLevelItem(i))
            return e
    raise KeyError(field)


def roles_of(tab, e) -> list[int]:
    return [tab.roles.get((id(e), k), -1) for k in range(1, e.setup.n_peaks + 1)]


def worst_error(tab, entries) -> float:
    worst = 0.0
    for e in entries:
        t = truth(e.curve.held["field"][0])
        for k, role in enumerate(roles_of(tab, e), 1):
            if role >= 0:
                worst = max(worst, abs(e.result.values[f"p{k}_center"] - t[role]))
    return worst


def dispersion_rows(tab, keys):
    v, err = tab.dres.values, tab.dres.errors
    for k in keys:
        if k in v:
            e = err.get(k)
            say(f"    {k:8s} = {v[k]:.6g}" + ("" if e is None else f" +- {e:.2g}"))
    if tab.damp is not None:
        say(f"    alpha    = {tab.damp.alpha:.3g} +- {tab.damp.alpha_err or 0:.2g}")


def pose_raw(v):
    R.select_file(v, "141500_yig_200nm_vna")
    m = v.map
    m.controls.x_combo.setCurrentText("rf_freq")
    m.controls.y_combo.setCurrentText("field")
    m.refresh()
    v.tabs.setCurrentWidget(m)


def main() -> int:
    base = (Path(r"C:\Users\Public\Documents\AaltoView") if sys.platform == "win32"
            else Path(tempfile.gettempdir()) / "AaltoView")
    shutil.rmtree(base / "case", ignore_errors=True)
    R.DATA = base / "case"
    DEMO.main([str(R.DATA)])
    OUT.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # 1-2. the map, raw and divided by the 800 mT line (the viewer)
    R.DOCS = OUT
    R.shot(app, "1-raw-map", "dark", pose_raw)
    R.shot(app, "2-reference-map", "dark", R.pose_map_reference)

    from aaltoview.apps import viewer as VW
    from fmr_fit import model as M
    from fmr_fit.app import FitWindow
    theme.set_theme("dark")
    VW.configure_pyqtgraph()
    theme.apply(app)
    path = next(R.DATA.glob("*/*_yig_200nm_vna.nc"))
    ds = load(path).load()
    curves = E.curves_along(ds, E.Selection("s21", x="rf_freq"), "field",
                            range(ds.sizes["field"]), path)
    fields = ", ".join(f"{c.held['field'][0]:g}" for c in curves)
    say(f"{path.name}: {len(curves)} sweeps, {curves[0].x.size} points each, "
        f"fields {fields} mT")
    say("truth: A = %.3g pJ/m, Ms = %g mT, gamma/2pi = %g GHz/T, alpha = %g"
        % (DEMO.YIG["A"], DEMO.YIG["Ms"], DEMO.G, DEMO.YIG["alpha"]))
    say("       Hex,n = " + ", ".join(f"{DEMO.yig_hex(n):.2f}" for n in range(1, 5)) + " mT")

    win = FitWindow()
    win.setWindowTitle("FMR fit")
    win.resize(*R.SIZE)
    win.show()
    R.pump(app, 0.3)
    win.add_curves(curves)
    # the settings are made on the curve on screen (25 mT) and carry over to
    # every curve not set up yet: the reference, then the number of peaks
    ref = curves[-1]
    win.ref_combo.setCurrentIndex(win.ref_combo.findText(ref.label))
    win.peaks.setValue(5)

    # 2b. what the background does to a fit: 200 mT, 5 peaks, WITHOUT the reference
    say()
    say("200 mT, 5 peaks, oscillator + delay, NO reference:")
    e200 = show(win, 200.0)
    win.ref_combo.setCurrentIndex(0)                      # "none"
    win.fit()                                             # this curve only
    t = sorted(truth(200.0).values())
    got = sorted(e200.result.values[f"p{k}_center"] for k in range(1, 6))
    say("    f0 found  " + "  ".join(f"{g:.4f}" for g in got))
    say("    f0 truth  " + "  ".join(f"{g:.4f}" for g in t))
    say(f"    worst miss {max(abs(a - b) for a, b in zip(got, t)):.3f} GHz; "
        f"flag: {M.suspicious(e200.result) or '-'}")

    # 3. three by hand, WITH the reference
    win.ref_combo.setCurrentIndex(win.ref_combo.findText(ref.label))
    select(win, BY_HAND)
    win.fit_all()
    say()
    say(f"by hand, / {ref.held['field'][0]:g} mT, 5 peaks:")
    for b in BY_HAND:
        e = show(win, b)
        got = sorted(e.result.values[f"p{k}_center"] for k in range(1, 6))
        tt = sorted(truth(b).values())
        say(f"    {b:5g} mT  worst |f0 - truth| = "
            f"{max(abs(a - c) for a, c in zip(got, tt)) * 1e3:.2f} MHz, "
            f"FWHM(uniform) = {min(2 * e.result.values[f'p{k}_hwhm'] for k in range(1, 6)) * 1e3:.1f} MHz")
    show(win, 200.0)
    zoom(app, win, 7.60, 8.12)                    # uniform mode + PSSW n = 1
    grab(app, win, "3-three-by-hand")

    # 4. a rough dispersion: roles by order, exchange field per mode
    tab = win.dispersion
    win.tabs.setCurrentWidget(tab)
    tab.assign_pssw()
    tab.fit()
    say()
    say(f"rough dispersion from {len([p for p in tab.points if p.use])} resonances "
        "(exchange field per mode):")
    dispersion_rows(tab, ["gamma", "Meff", "Hex1", "Hex2", "Hex3", "Hex4"])
    grab(app, win, "4-rough-dispersion")

    # 5. predict + fit the other nine
    tab.predict_and_fit()
    pred = [e for e in win.entries if e.predicted]
    say()
    say(f"predict + fit: {len(pred)} sweeps, "
        f"{sum(e.setup.n_peaks for e in pred)} peaks placed and bounded (+-3 linewidths)")
    say(f"    worst |f0 - truth| over all predicted peaks = "
        f"{worst_error(tab, pred) * 1e3:.2f} MHz")
    say(f"    flagged: {[e.curve.label for e in pred if M.suspicious(e.result)] or 'none'}")
    e25 = next(e for e in pred if e.curve.held["field"][0] == 25.0)
    say(f"    25 mT: roles {sorted(roles_of(tab, e25))} (uniform at "
        f"{truth(25.0)[0]:.3f} GHz, below the sweep: left out)")
    say(f"    dispersion refitted with {len([p for p in tab.points if p.use])} resonances")
    win.tabs.setCurrentIndex(0)
    show(win, 25.0)
    zoom(app, win, 2.0, 2.65)                     # the band edge + PSSW n = 1
    grab(app, win, "5-predicted-25mT")

    # 6. one exchange stiffness for all orders
    win.tabs.setCurrentWidget(tab)
    tab.pssw.setCurrentIndex(tab.pssw.findData("A"))
    tab.d_edit.setText("200")
    tab.ms_edit.setText("176")
    tab._model_changed()
    tab.fit()
    say()
    say("final dispersion, exchange stiffness A (d = 200 nm, mu0Ms = 176 mT):")
    dispersion_rows(tab, ["gamma", "Meff", "A", "dH0"])
    grab(app, win, "6-final-dispersion")
    win.close()

    (OUT / "numbers.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"  -> {(OUT / 'numbers.txt').relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
