"""origin.py -- put a view straight into a running Origin (OriginLab).

AaltoView did this through the OriginLab LabVIEW VIs (OA_NewWorksheet,
OA_Mat-SetData, ...). Here it is `originpro`, OriginLab's own Python package,
which drives Origin over COM from OUTSIDE Origin. Install with the extra:

    uv sync --extra gui --extra origin

What arrives in Origin:
    curves -> a workbook (X column(s) + one Y column per curve, long name /
              units / comments filled in) and a line+symbol graph of all of them
    map    -> a matrix with its X/Y extent set, plus a colour-map graph; if the
              axes are NOT evenly spaced (a matrix only knows start and end) it
              goes in as X/Y/Z columns instead, so no coordinate is invented

Attach, never own: `op.attach()` connects to the Origin you already have open
(or starts one), and `op.detach()` leaves it running with the data in it. The
alternative -- `originpro` creating its own instance -- would close Origin
together with this process and take the new workbook with it.

Why the viewer runs this in a CHILD PROCESS (`python -m aaltoview.origin
payload.npz`): starting Origin takes several seconds, COM calls block, and
originpro registers an exit handler that belongs to a short-lived process. A
child keeps the viewer responsive and keeps COM out of the Qt event loop; a
crash in there cannot take unsaved curves down with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .export import (Curve, Map, NORMS, Selection, axis_title, displayed_y,
                     is_uniform)


def available() -> str | None:
    """None if pushing can work here, otherwise the reason it cannot."""
    if sys.platform != "win32":
        return "Origin runs on Windows only"
    try:
        import originpro  # noqa: F401
    except Exception as exc:      # ImportError, or OriginExt failing to load
        return (f"originpro is not installed ({exc.__class__.__name__}); "
                "run: uv sync --extra gui --extra origin")
    return None


def _op():
    import originpro as op
    op.attach()
    op.set_show(True)
    return op


def _col(values) -> list:
    # NaN goes in as NaN; Origin shows it as a missing value ("--"), which is
    # exactly what an unmeasured point is.
    return [float(v) for v in np.asarray(values, dtype=float)]


def push_curves(curves: list[Curve], norm: str = "none", offset: float = 0.0,
                name: str = "AaltoView curves", graph: bool = True) -> str:
    """Visible curves -> a new workbook (+ graph). Returns the book's name."""
    shown = [c for c in curves if c.visible]
    if not shown:
        raise ValueError("no visible curves to send")
    ys = displayed_y(curves, norm, offset)
    op = _op()
    book = op.new_book("w", lname=name)
    wks = book[0]
    shared = all(c.x.shape == shown[0].x.shape and np.array_equal(c.x, shown[0].x)
                 for c in shown)
    pairs = []                         # (y column, x column) for the graph
    col = 0
    if shared:
        wks.from_list(col, _col(shown[0].x), lname=shown[0].x_name,
                      units=shown[0].x_unit, axis="X")
        xcol, col = 0, 1
    for c, y in zip(shown, ys):
        if not shared:
            wks.from_list(col, _col(c.x), lname=c.x_name, units=c.x_unit,
                          comments=c.label, axis="X")
            xcol, col = col, col + 1
        wks.from_list(col, _col(y), lname=c.y_name,
                      units=c.y_unit if norm == "none" else "",
                      comments=c.label if norm == "none" else f"{c.label} [{NORMS[norm]}]",
                      axis="Y")
        pairs.append((col, xcol))
        col += 1
    if graph:
        gl = op.new_graph(lname=name, template="line")[0]
        for ycol, xc in pairs:
            gl.add_plot(wks, coly=ycol, colx=xc, type="y")
        if len(pairs) > 1:
            gl.group()                 # one colour per curve, as on screen
        gl.rescale()
        # Origin's default legend is the LONG NAME, i.e. "pm16.power" three
        # times. Point every entry at its column's Comments row (@LC) instead,
        # which holds the curve label. A substitution rather than the label text
        # itself, so a label containing quotes cannot break the LabTalk line.
        entries = "%(CRLF)".join(f"\\l({k}) %({k}, @LC)" for k in range(1, len(pairs) + 1))
        gl.lt_exec(f'legend.text$ = "{entries}";')
    return book.name


def push_map(m: Map, name: str = "AaltoView map", graph: bool = True) -> str:
    """A map -> a matrix (or XYZ columns if the axes are uneven) + colour map."""
    op = _op()
    z_title = axis_title(m.z_name, m.z_unit)
    if is_uniform(m.x) and is_uniform(m.y):
        book = op.new_book("m", lname=name)
        ms = book[0]
        # Row j of the matrix is y[j]; xymap says where the first and last
        # row/column sit, so Origin draws the same orientation as the viewer.
        ms.from_np(np.asarray(m.z, dtype=float))
        ms.xymap = float(m.x[0]), float(m.x[-1]), float(m.y[0]), float(m.y[-1])
        ms.set_label("x", m.x_name, "L"); ms.set_label("x", m.x_unit, "U")
        ms.set_label("y", m.y_name, "L"); ms.set_label("y", m.y_unit, "U")
        # a matrix OBJECT has a long name and comments but no units row
        ms.set_label(0, z_title, "L")
        if m.source:
            ms.set_label(0, Path(m.source).name, "C")
        if graph:
            gl = op.new_graph(lname=name, template="contour")[0]
            gl.add_mplot(ms, 0, type="contour")
            gl.rescale()
        return book.name

    book = op.new_book("w", lname=name)
    wks = book[0]
    xx, yy = np.meshgrid(m.x, m.y)
    wks.from_list(0, _col(xx.ravel()), lname=m.x_name, units=m.x_unit, axis="X")
    wks.from_list(1, _col(yy.ravel()), lname=m.y_name, units=m.y_unit, axis="Y")
    wks.from_list(2, _col(np.asarray(m.z).ravel()), lname=m.z_name, units=m.z_unit,
                  comments=f"{z_title}: uneven axes, sent as XYZ columns", axis="Z")
    if graph:
        gl = op.new_graph(lname=name, template="contour")[0]
        gl.add_plot(wks, coly=1, colx=0, colz=2, type="contour")
        gl.rescale()
    return book.name


# ─────────────── the child-process hand-over: payload file + CLI ──────────────

def save_payload(path: str | Path, *, curves: list[Curve] | None = None,
                 m: Map | None = None, norm: str = "none", offset: float = 0.0,
                 name: str = "") -> Path:
    """Everything push_* needs, in one .npz (arrays as arrays, the rest as JSON)."""
    path = Path(path)
    arrays, meta = {}, {"norm": norm, "offset": offset, "name": name}
    if m is not None:
        arrays.update(z=m.z, x=m.x, y=m.y)
        meta["map"] = {k: getattr(m, k) for k in
                       ("x_name", "x_unit", "y_name", "y_unit", "z_name", "z_unit", "source")}
        meta["map"].update(levels=list(m.levels), selection=m.selection.to_dict())
    if curves is not None:
        meta["curves"] = []
        for i, c in enumerate(curves):
            arrays[f"cx{i}"], arrays[f"cy{i}"] = c.x, c.y
            meta["curves"].append({k: getattr(c, k) for k in
                                   ("label", "x_name", "x_unit", "y_name", "y_unit",
                                    "source", "visible")}
                                  | {"selection": c.selection.to_dict()})
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load_payload(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        out = {"norm": meta["norm"], "offset": meta["offset"], "name": meta["name"]}
        if "map" in meta:
            mm = meta["map"]
            out["map"] = Map(z=f["z"], x=f["x"], y=f["y"], levels=tuple(mm["levels"]),
                             selection=Selection.from_dict(mm["selection"]),
                             **{k: mm[k] for k in ("x_name", "x_unit", "y_name", "y_unit",
                                                   "z_name", "z_unit", "source")})
        if "curves" in meta:
            out["curves"] = [
                Curve(x=f[f"cx{i}"], y=f[f"cy{i}"],
                      selection=Selection.from_dict(c["selection"]),
                      **{k: c[k] for k in ("label", "x_name", "x_unit", "y_name",
                                           "y_unit", "source", "visible")})
                for i, c in enumerate(meta["curves"])]
    return out


def push_payload(path: str | Path) -> str:
    p = load_payload(path)
    if "map" in p:
        return push_map(p["map"], name=p["name"] or "AaltoView map")
    return push_curves(p["curves"], p["norm"], p["offset"], name=p["name"] or "AaltoView curves")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m aaltoview.origin payload.npz")
        return 2
    reason = available()
    if reason:
        print(f"ERROR {reason}")
        return 1
    try:
        name = push_payload(argv[0])
    except Exception as exc:
        # ASCII only: this goes to a pipe (gotcha #14)
        print(f"ERROR {exc.__class__.__name__}: {exc}".encode("ascii", "replace").decode())
        return 1
    print(f"OK {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
