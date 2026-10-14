"""Axes that go UP AND DOWN (a hysteresis loop), without a screen.

AaltoFlow's scan engine writes a field swept 0 -> 100 -> -100 -> 100 mT as ONE
dimension whose values repeat, plus `field_segment` / `field_direction` and the
runs as JSON (AaltoFlow docs/DEVELOPER_NOTES.md 4l). The files here are built
BY HAND to that contract (tools/make_demo_data.py), with a detector whose
value depends on the branch -- so a test can tell a right picture from one
where the branches were folded together.

What must hold everywhere: one line per run; a map shows one run (or every
point by number); a held value names its run; an average says whether it mixes
up and down; and every export (text, Origin, notebook, analysis modules)
carries the same split.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aaltoview import analysis_link as AL
from aaltoview import export as E
from aaltoview import segments as SG
from aaltoview import view as V
from aaltoview.view import Slice

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_demo_data as DEMO  # noqa: E402


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The demo files 11-13: a Kerr loop, a loop x log-frequency map, and an
    older typed loop without the segment coordinates."""
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    day = out / "2026-10-14"
    return {"loop": day / "091500_kerr_loop.nc",
            "map": day / "103000_fmr_loop_logfreq.nc",
            "old": day / "110000_kerr_loop_typed.nc"}


def _load(path):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return ds.load()


# ─────────────────────────── reading the runs ─────────────────────────────────

def test_the_contract_coordinates_are_read_as_runs(files):
    ds = _load(files["loop"])
    s = SG.segments_of(ds, "field")
    assert s.n == 3 and s.source == "recipe"
    assert [s.name(k) for k in range(3)] == ["seg 1 up", "seg 2 down", "seg 3 up"]
    # 21 + 40 + 40: the joints are measured once (joined runs drop a point)
    assert [s.indices(k).size for k in range(3)] == [21, 40, 40]
    assert s.span(1) == (21, 60)
    # a detector carries them too (the viewer reduces DataArrays)
    assert SG.segments_of(V.detector(ds, "kerr"), "field").n == 3
    assert SG.segments_of(ds, "nope") is None


def test_an_older_typed_loop_is_split_where_it_turns_back(files):
    new, old = _load(files["loop"]), _load(files["old"])
    a, b = SG.segments_of(new, "field"), SG.segments_of(old, "field")
    assert b.source == "reversals"
    np.testing.assert_array_equal(a.seg, b.seg)          # what the engine writes
    assert [b.direction(k) for k in range(3)] == [1, -1, 1]


def test_reversal_segments_is_the_engines_rule():
    """The same rule as scan-core's recipe.reversal_segments (copied: this
    package does not import the suite): a point belongs to the run of the
    step INTO it, a repeated value changes nothing, monotonic -> None."""
    seg, info = SG.reversal_segments([0, 50, 100, 50, 0, -50])
    assert seg.tolist() == [0, 0, 0, 1, 1, 1]
    assert info[1] == {"start": 100.0, "stop": -50.0, "num": 4, "direction": -1,
                       "joined": 1, "first": 3, "count": 3}
    seg, _ = SG.reversal_segments([0, 1, 1, 2, 1])
    assert seg.tolist() == [0, 0, 0, 0, 1]
    assert SG.reversal_segments([0, 1, 1, 2]) is None
    assert SG.reversal_segments([3, 2, 1]) is None
    assert SG.reversal_segments([0, 1]) is None
    assert SG.reversal_segments(["a", "b", "a"]) is None
    try:                                     # when scan-core is installed: identical
        from scan_core.recipe import reversal_segments as engine
    except Exception:
        return
    for v in ([0, 50, 100, 50, 0, -50], [0, 1, 1, 2, 1], np.r_[0:5, 3:-3:-1, -2:4]):
        mine, theirs = SG.reversal_segments(v), engine(v)
        assert mine[0].tolist() == theirs[0].tolist() and mine[1] == theirs[1]


def test_broken_json_falls_back_to_the_direction_coordinate(files):
    ds = _load(files["loop"])
    ds["field"].attrs["segments_json"] = "not json"
    s = SG.segments_of(ds, "field")
    assert [s.direction(k) for k in range(3)] == [1, -1, 1]


def test_a_value_names_its_run(files):
    ds = _load(files["loop"])
    assert V.coord_text(ds, "field", 20) == "100 mT · seg 1 up"
    assert V.coord_texts(ds, "field", [21, 61]) == ["95 mT · seg 2 down",
                                                    "-95 mT · seg 3 up"]


# ─────────────────────────── the reduction ────────────────────────────────────

def test_a_map_shows_one_run_ascending_never_two_in_a_cell(files):
    ds = _load(files["map"])
    da = V.detector(ds, "lockin")
    r = V.reduce_cube(da, "rf_freq", "field", runs={}, ds=ds)         # default: first run
    assert r.data.shape == (21, 61)
    # runs=None: nothing selected -- a caller that picked its run itself
    # (scan-core's live view isel()s first) gets the axis as it is
    r = V.reduce_cube(da.isel(field=slice(21, 61)), "rf_freq", "field")
    assert r.data.shape == (40, 61) and r.index == {}
    r = V.reduce_cube(da, "rf_freq", "field", runs={"field": 1}, ds=ds)
    assert r.data.shape == (40, 61)
    y = r.data["field"].values
    assert y[0] == -100 and y[-1] == 95 and np.all(np.diff(y) > 0)      # ascending
    assert r.index["field"].tolist() == list(range(60, 20, -1))         # file indices
    r = V.reduce_cube(da, "rf_freq", "field", runs={"field": "index"}, ds=ds)
    assert r.data.shape == (101, 61)
    assert r.data["field"].values.tolist() == list(range(101))
    # a line shows every run, in measured order
    r = V.reduce_cube(da, "field", None, {"rf_freq": Slice("at", 30)}, runs={}, ds=ds)
    assert r.data.shape == (101,)


def test_the_branches_differ_where_the_physics_says(files):
    """At 0 mT the film is magnetised + on the way down and - on the way up
    (coercive field 30 mT): the runs must not be averaged into one curve."""
    ds = _load(files["loop"])
    down = E.make_curve(ds, E.Selection("kerr", "field", runs={"field": 1}))
    up = E.make_curve(ds, E.Selection("kerr", "field", runs={"field": 2}))
    at0 = lambda c: float(np.asarray(c.y)[np.argmin(np.abs(c.x))])     # noqa: E731
    assert at0(down) > 8 and at0(up) < -8
    assert down.label == "kerr (field seg 2 down)" and down.segment is None


# ─────────────────────────────── curves ───────────────────────────────────────

def test_a_curve_along_a_loop_is_one_line_per_run_closed_at_the_joints(files):
    ds = _load(files["loop"])
    c = E.make_curve(ds, E.Selection("kerr", "field"), source=files["loop"])
    assert c.segment is not None and len(c.runs) == 3
    runs = E.curve_runs(c)
    assert [r for r, _ in runs] == ["seg 1 up", "seg 2 down", "seg 3 up"]
    # the down run is drawn from the up run's last point (100 mT): closed loop
    assert [idx.size for _, idx in runs] == [21, 41, 41]
    assert c.x[runs[1][1][0]] == 100.0 and c.x[runs[2][1][0]] == -100.0


def test_a_held_loop_value_and_averages_say_which_run(files):
    ds = _load(files["map"])
    c = E.make_curve(ds, E.Selection("lockin", "rf_freq", slices={"field": Slice("at", 61)}))
    assert c.label == "field = -95 mT · seg 3 up"
    c = E.make_curve(ds, E.Selection("lockin", "rf_freq",
                                     slices={"field": Slice("mean", 21, 60)}))
    assert c.label == "field mean of the seg 2 down run"
    c = E.make_curve(ds, E.Selection("lockin", "rf_freq", slices={"field": Slice("mean")}))
    assert c.label == "field mean (up and down mixed)"
    # "one per value of" a loop: every curve names its run
    cs = E.curves_along(ds, E.Selection("lockin", "rf_freq"), "field", [10, 30])
    assert [c.label for c in cs] == ["field = 50 mT · seg 1 up", "field = 50 mT · seg 2 down"]


def test_a_loop_is_written_with_its_run_columns(files, tmp_path):
    ds = _load(files["loop"])
    c = E.make_curve(ds, E.Selection("kerr", "field"), source=files["loop"])
    r = E.make_curve(ds, E.Selection("reflectivity", "field"), source=files["loop"])
    p = E.write_curves(tmp_path / "loop.csv", [c, r])
    rows = p.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].split(",") == ["field", "field_segment", "field_direction",
                                  "kerr", "reflectivity"]
    first_down = rows[3 + 21].split(",")                  # point 21: 95 mT, run 1
    assert first_down[:3] == ["95.0", "1", "-1"]
    # curves of different runs do not share an X column (their x differ)
    d = E.make_curve(ds, E.Selection("kerr", "field", runs={"field": 1}))
    p = E.write_curves(tmp_path / "two.dat", [c, d])
    head = p.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert head == ["field", "field_segment", "field_direction", "kerr", "field", "kerr"]


def test_figures_and_origin_draw_one_line_per_run(files, tmp_path):
    ds = _load(files["loop"])
    c = E.make_curve(ds, E.Selection("kerr", "field"), source=files["loop"])
    fig = E.figure_curves([c])
    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert labels == ["kerr · seg 1 up", "kerr · seg 2 down", "kerr · seg 3 up"]
    from aaltoview import origin as O
    shared, pieces = O.curve_columns([c])
    assert shared is None and [p[2] for p in pieces] == labels      # one X/Y pair per run
    assert pieces[1][0][0] == 100.0                                  # closed at the joint
    back = O.load_payload(O.save_payload(tmp_path / "p.npz", curves=[c]))["curves"][0]
    np.testing.assert_array_equal(back.segment, c.segment)
    assert back.runs == c.runs


def test_a_module_gets_one_curve_per_run_labelled(files):
    ds = _load(files["loop"])
    c = E.make_curve(ds, E.Selection("kerr", "field"), source=files["loop"])
    parts = E.split_runs(c)
    assert [p.label for p in parts] == ["kerr · seg 1 up", "kerr · seg 2 down",
                                        "kerr · seg 3 up"]
    # each part regenerates from its selection (the notebook relies on it)
    again = E.make_curve(ds, parts[1].selection)
    np.testing.assert_allclose(np.sort(again.x), np.sort(parts[1].x))
    back = AL.curve_from_dict(AL.curve_to_dict(parts[1]))
    assert back.label == parts[1].label and back.selection.runs == {"field": 1}


# ──────────────────────────────── maps ────────────────────────────────────────

def test_a_map_of_one_run_and_its_columns_for_a_module(files):
    ds = _load(files["map"])
    sel = E.Selection("lockin", "field", "rf_freq", runs={"field": 1})
    m, _ = E.make_map(ds, sel)
    assert m.z.shape == (61, 40) and m.x_tags[0] == "seg 2 down"
    md = E.make_map_data(ds, sel, source=files["map"])
    assert md.label == "lockin (field seg 2 down)"
    md = AL.map_from_dict(AL.map_to_dict(md))                     # over the wire
    cs = E.map_to_curves(md, [5])
    assert cs[0].label == "field = -75 mT · seg 2 down"
    assert cs[0].held["field"] == (-75.0, "mT")
    # the curve's selection points at the FILE index of that point
    k = cs[0].selection.slices["field"].i0
    assert ds["field"].values[k] == -75 and SG.segments_of(ds, "field").seg[k] == 1
    again = E.make_curve(ds, cs[0].selection)
    np.testing.assert_allclose(again.y, cs[0].y)


def test_a_map_by_point_number_names_the_real_value(files):
    ds = _load(files["map"])
    m, _ = E.make_map(ds, E.Selection("lockin", "rf_freq", "field", runs={"field": "index"}))
    assert m.y_unit == "point #" and m.y.tolist() == list(range(101))
    assert m.y_tags[30] == "50 mT · seg 2 down"


def test_a_reference_along_a_loop_keeps_every_run(files):
    """'the curve at' (1D panel) along the field: the reference curve is a
    file index of any run; the line keeps all its points."""
    ds = _load(files["map"])
    ref = {"x": "rf_freq", "y": "field", "mode": "row", "op": "divide", "i": 100}
    c = E.make_curve(ds, E.Selection("lockin", "rf_freq", slices={"field": Slice("at", 30)},
                                     ref=ref))
    z = V.detector(ds, "lockin").values
    np.testing.assert_allclose(c.y, np.abs(z[30] / z[100]))
    assert c.label.endswith("(÷ field = 100 mT · seg 3 up)")
    # and a map cut (row through the cursor) of a referenced one-run map
    style = E.MapStyle(ref="row", ref_i=0)
    m, _ = E.make_map(ds, E.Selection("lockin", "rf_freq", "field", runs={"field": 1}), style)
    row = E.make_curve(ds, E.Selection("lockin", "rf_freq", slices={"field": Slice("at", 59)},
                                       runs={"field": 1},
                                       ref={"x": "rf_freq", "y": "field", "mode": "row",
                                            "op": "divide", "i": 0}))
    np.testing.assert_allclose(row.y, m.z[1], equal_nan=True)    # 59 = 2nd row (-95 mT)


# ────────────────────────────── the notebook ──────────────────────────────────

def _run_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {}
    import matplotlib
    matplotlib.use("Agg")
    here = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]).replace("plt.show()", "plt.close('all')"), ns)
    finally:
        os.chdir(here)
    return ns


@pytest.mark.parametrize("runs", [{"field": 1}, {"field": "index"}, {}])
def test_the_notebook_recomputes_a_map_of_a_loop(files, tmp_path, runs):
    ds = _load(files["map"])
    style = E.MapStyle(norm="rows")
    m, _ = E.make_map(ds, E.Selection("lockin", "rf_freq", "field", runs=runs), style,
                      source=files["map"])
    ns = _run_notebook(E.write_notebook(tmp_path / "m.ipynb", m=m, style=style))
    np.testing.assert_allclose(ns["z"], m.z, equal_nan=True)
    np.testing.assert_allclose(ns["ys"], m.y)


def test_the_notebook_draws_a_loop_one_line_per_run(files, tmp_path):
    loop, old, fmap = _load(files["loop"]), _load(files["old"]), _load(files["map"])
    curves = [E.make_curve(loop, E.Selection("kerr", "field"), source=files["loop"]),
              E.make_curve(old, E.Selection("kerr", "field", runs={"field": 1}),
                           source=files["old"]),
              E.make_curve(fmap, E.Selection("lockin", "rf_freq",
                                             slices={"field": Slice("mean", 21, 60)}),
                           source=files["map"]),
              E.make_curve(fmap, E.Selection("lockin", "rf_freq",
                                             slices={"field": Slice("at", 30)},
                                             ref={"x": "rf_freq", "y": "field", "mode": "row",
                                                  "op": "divide", "i": 100}),
                           source=files["map"])]
    ns = _run_notebook(E.write_notebook(tmp_path / "c.ipynb", curves=curves))
    for (x, y), c in zip(ns["lines"], curves):
        np.testing.assert_allclose(x, c.x)
        np.testing.assert_allclose(y, c.y)
    labels = [ln.get_label() for ln in ns["ax"].get_lines()]
    assert labels[:3] == ["kerr · seg 1 up", "kerr · seg 2 down", "kerr · seg 3 up"]
    assert len(labels) == 6                       # 3 runs + 3 single curves
