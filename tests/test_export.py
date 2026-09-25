"""The data viewer's exports, tested without a screen.

What must not go wrong when a view leaves the viewer:
* the numbers in a file / notebook / Origin payload are the numbers on screen;
* normalisation and stacking change the export, never the stored curve;
* the notebook -- which carries its OWN copy of the reduction so it runs
  without this package -- computes exactly what aaltoview/view.py computes. That
  duplication is only safe because this file executes the generated code.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aaltoview import export as E
from aaltoview.data import find_measurements, summarize
from aaltoview.view import Slice


def _cube() -> xr.Dataset:
    """freq x y x x, with a value that says exactly where it came from."""
    f = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 20.0])
    x = np.array([0.0, 1.0])
    vals = f[:, None, None] * 100 + y[None, :, None] + x[None, None, :]
    return xr.Dataset(
        {"kerr": (("freq", "y", "x"), vals, {"units": "mdeg"})},
        coords={"freq": ("freq", f, {"units": "GHz"}),
                "y": ("y", y, {"units": "um"}),
                "x": ("x", x, {"units": "um"})},
        attrs={"name": "cube", "dims": "freq,y,x", "n_points": 24, "seconds": 12.0})


def _complex_file(tmp_path) -> Path:
    """A complex detector stored the engine's way: a real/imag pair."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(3, 5, 4)) + 1j * rng.normal(size=(3, 5, 4))
    attrs = {"units": "V"}
    ds = xr.Dataset(
        {"s21_real": (("field", "freq", "pos"), z.real,
                      {**attrs, "complex_pair": "s21", "complex_part": "real"}),
         "s21_imag": (("field", "freq", "pos"), z.imag,
                      {**attrs, "complex_pair": "s21", "complex_part": "imag"})},
        coords={"field": ("field", [0.0, 10.0, 20.0], {"units": "mT"}),
                "freq": ("freq", np.linspace(1, 2, 5), {"units": "GHz"}),
                "pos": ("pos", [0.0, 1.0, 2.0, 3.0], {"units": "um"})})
    p = tmp_path / "complex.nc"
    ds.to_netcdf(p, engine="h5netcdf")
    return p


# ──────────────────────────────── curves ──────────────────────────────────────

def test_a_curve_says_where_it_came_from():
    ds = _cube()
    sel = E.Selection("kerr", x="x", slices={"freq": Slice("at", 2), "y": Slice("mean")})
    c = E.make_curve(ds, sel, source="a.nc")
    assert c.y.tolist() == [310.0, 311.0]            # freq 3 GHz, mean over y = 10
    assert c.label == "freq = 3 GHz, y mean"
    assert (c.x_unit, c.y_unit, c.source) == ("um", "mdeg", "a.nc")


def test_a_curve_is_a_copy_not_a_view_of_the_dataset():
    """A curve added from a live run must not change when the next snapshot
    lands -- that is what "freezing" it means."""
    ds = _cube()
    c = E.make_curve(ds, E.Selection("kerr", x="x"))
    ds["kerr"].values[:] = -1
    assert c.y.tolist() == [100.0, 101.0]


def test_curves_along_a_dim_step_that_dim_and_keep_the_rest():
    ds = _cube()
    sel = E.Selection("kerr", x="x", slices={"y": Slice("at", 1)})
    cs = E.curves_along(ds, sel, "freq", [0, 3])
    assert [c.y.tolist() for c in cs] == [[110.0, 111.0], [410.0, 411.0]]
    assert cs[0].label.startswith("freq = 1 GHz")


@pytest.mark.parametrize("mode,expected", [
    ("none", [2.0, -4.0, np.nan, 1.0]),
    ("peak", [0.5, -1.0, np.nan, 0.25]),
    ("minmax", [1.0, 0.0, np.nan, 5 / 6]),
    ("first", [1.0, -2.0, np.nan, 0.5]),
    ("zero_mean", [2.0 + 1 / 3, -4.0 + 1 / 3, np.nan, 1.0 + 1 / 3]),
])
def test_normalisation_keeps_holes_as_holes(mode, expected):
    out = E.normalize(np.array([2.0, -4.0, np.nan, 1.0]), mode)
    np.testing.assert_allclose(out, expected)


def test_stacking_applies_after_normalising_and_skips_hidden_curves():
    ds = _cube()
    a, b, c = (E.make_curve(ds, E.Selection("kerr", x="x", slices={"freq": Slice("at", i)}))
               for i in range(3))
    b.visible = False
    ys = E.displayed_y([a, b, c], norm="peak", offset=1.0)
    np.testing.assert_allclose(ys[0], [100 / 101, 1.0])
    np.testing.assert_allclose(ys[1], [300 / 301 + 1, 2.0])     # c is the SECOND shown
    assert a.y.tolist() == [100.0, 101.0]                       # stored numbers untouched


def test_curves_with_one_x_share_one_column(tmp_path):
    ds = _cube()
    cs = E.curves_along(ds, E.Selection("kerr", x="x"), "freq", [0, 1])
    rows = (tmp_path / "c.csv")
    E.write_curves(rows, cs)
    lines = rows.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x,kerr,kerr"                    # long names
    assert lines[1] == "um,mdeg,mdeg"                   # units
    assert lines[2].startswith(',"freq = 1 GHz, y = 0 um"')   # labels, quoted: they
                                                                # contain commas
    assert lines[3] == "0.0,100.0,200.0"


def test_curves_with_different_x_get_their_own_columns(tmp_path):
    ds = _cube()
    c1 = E.make_curve(ds, E.Selection("kerr", x="x"))
    c2 = E.make_curve(ds, E.Selection("kerr", x="y"))
    p = E.write_curves(tmp_path / "c.dat", [c1, c2])
    rows = [r.split("\t") for r in p.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == ["x", "kerr", "y", "kerr"]
    assert rows[5] == ["", "", "20.0", "120.0"]          # the short curve ends empty, not 0


# ───────────────────────────────── maps ───────────────────────────────────────

def test_a_map_file_has_the_axes_and_the_orientation_on_screen(tmp_path):
    ds = _cube()
    m, _ = E.make_map(ds, E.Selection("kerr", x="x", y="freq", slices={"y": Slice("at", 0)}))
    assert m.z.shape == (4, 2)                            # (y, x)
    rows = [r.split("\t") for r in
            E.write_map(tmp_path / "m.dat", m, "matrix").read_text(encoding="utf-8").splitlines()]
    assert rows[0][1:] == ["0.0", "1.0"]                  # X across the top
    assert [r[0] for r in rows[1:]] == ["1.0", "2.0", "3.0", "4.0"]   # Y down the side
    assert rows[3][1:] == ["300.0", "301.0"]
    xyz = E.write_map(tmp_path / "m.csv", m, "xyz").read_text(encoding="utf-8").splitlines()
    assert xyz[0] == "x,freq,kerr" and xyz[1] == "um,GHz,mdeg"
    assert xyz[2 + 5] == "1.0,3.0,301.0"                  # row-major: (y=3, x=1)


def test_symmetric_limits_are_centred_on_zero():
    z = np.array([[-1.0, 0.2], [3.0, 0.5]])
    lo, hi = E.map_levels(z, E.MapStyle(auto=False, lo=-1, hi=3, symmetric=True))
    assert (lo, hi) == (-3.0, 3.0)


def test_line_normalisation_scales_each_row_to_its_own_peak():
    z = np.array([[1.0, 2.0], [10.0, -40.0], [np.nan, np.nan]])
    rows = E.style_values(z, E.MapStyle(norm="rows"))
    np.testing.assert_allclose(rows[:2], [[0.5, 1.0], [0.25, -1.0]])
    assert np.isnan(rows[2]).all()                          # unmeasured stays unmeasured
    cols = E.style_values(z, E.MapStyle(norm="columns"))
    np.testing.assert_allclose(cols[:2], [[0.1, 0.05], [1.0, -1.0]])


def test_log_of_zero_is_a_hole():
    z = E.style_values(np.array([[0.0, 100.0]]), E.MapStyle(log=True))
    assert np.isnan(z[0, 0]) and z[0, 1] == 2.0


def test_uneven_axes_are_recognised():
    assert E.is_uniform(np.linspace(0, 1, 11))
    assert not E.is_uniform(np.array([0.0, 1.0, 3.0]))


def test_figures_render_and_save(tmp_path):
    ds = _cube()
    m, _ = E.make_map(ds, E.Selection("kerr", x="x", y="y"), E.MapStyle(cmap="red-blue",
                                                                         symmetric=True))
    E.save_figure(E.figure_map(m), tmp_path / "m.png", dpi=50)
    cs = E.curves_along(ds, E.Selection("kerr", x="y"), "freq", range(4))
    E.save_figure(E.figure_curves(cs, "minmax", 0.5), tmp_path / "c.svg")
    assert (tmp_path / "m.png").stat().st_size > 1000
    assert (tmp_path / "c.svg").read_text(encoding="utf-8").startswith("<?xml")


# ─────────────────────────────── notebooks ────────────────────────────────────

def _run_notebook(path: Path) -> dict:
    """Execute every code cell the way Jupyter would, in the notebook's folder."""
    import matplotlib
    matplotlib.use("Agg")
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    ns: dict = {}
    here = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]).replace("plt.show()", "plt.close('all')"), ns)
    finally:
        os.chdir(here)
    return ns


def test_the_notebook_recomputes_exactly_what_the_viewer_shows(tmp_path):
    src = _complex_file(tmp_path)
    ds = xr.open_dataset(src, engine="h5netcdf").load()
    sel = E.Selection("s21", x="freq", y="field",
                      slices={"pos": Slice("mean", 1, 3)}, part="abs")
    m, _ = E.make_map(ds, sel, E.MapStyle(norm="rows"), source=src)
    c1 = E.make_curve(ds, E.Selection("s21", x="freq", part="arg",
                                      slices={"field": Slice("at", 2), "pos": Slice("mean")}),
                      source=src)
    c2 = E.make_curve(ds, E.Selection("s21", x="pos",
                                      slices={"field": Slice("mean"), "freq": Slice("at", 4)}),
                      source=src)

    nb_dir = tmp_path / "analysis"
    nb_dir.mkdir()
    nb = E.write_notebook(nb_dir / "view.ipynb", m=m, style=E.MapStyle(norm="rows"),
                          curves=[c1, c2], norm="peak", offset=0.25)
    ns = _run_notebook(nb)

    np.testing.assert_allclose(ns["z"], m.z)                   # the map, row-normalised
    (x1, y1), (x2, y2) = ns["lines"]
    np.testing.assert_allclose(y1, E.normalize(c1.y, "peak"))
    np.testing.assert_allclose(y2, E.normalize(c2.y, "peak") + 0.25)
    np.testing.assert_allclose(x2, c2.x)
    # and the data path is relative, so the notebook and data can move together
    assert "'../complex.nc'" in "".join(json.loads(nb.read_text(encoding="utf-8"))
                                        ["cells"][2]["source"])


def test_a_notebook_needs_a_saved_measurement(tmp_path):
    c = E.make_curve(_cube(), E.Selection("kerr", x="x"))          # source None: a live run
    with pytest.raises(ValueError, match="save the measurement first"):
        E.write_notebook(tmp_path / "v.ipynb", curves=[c])


# ──────────────────────────── the data folder ─────────────────────────────────

def test_the_file_list_reads_autosave_names_and_skips_half_written_files(tmp_path):
    day = tmp_path / "2026-09-16"
    day.mkdir()
    ds = _cube()
    ds.to_netcdf(day / "093000_early.nc", engine="h5netcdf")
    ds.to_netcdf(day / "171500_late.nc", engine="h5netcdf")
    ds.to_netcdf(day / "171600_late.writing.nc", engine="h5netcdf")
    (tmp_path / "broken.nc").write_bytes(b"not a netcdf file")

    found = find_measurements(tmp_path)
    names = [p.name for p in found]
    assert "171600_late.writing.nc" not in names
    assert names.index("171500_late.nc") < names.index("093000_early.nc")   # newest first

    s = summarize(day / "171500_late.nc")
    assert (s.measured.hour, s.measured.minute) == (17, 15)
    assert s.dims == ["freq", "y", "x"] and s.shape_text == "4 x 3 x 2"
    assert s.detectors == ["kerr"] and s.n_points == 24
    bad = summarize(tmp_path / "broken.nc")
    assert bad.error and bad.dims == []                      # reported, not raised


def test_a_complex_pair_is_listed_as_one_detector(tmp_path):
    assert summarize(_complex_file(tmp_path)).detectors == ["s21"]


# ──────────────────────────────── Origin ──────────────────────────────────────

def test_the_origin_payload_round_trips(tmp_path):
    from aaltoview import origin
    ds = _cube()
    cs = E.curves_along(ds, E.Selection("kerr", x="x"), "freq", [0, 2])
    cs[1].visible = False
    m, _ = E.make_map(ds, E.Selection("kerr", x="x", y="freq"), E.MapStyle(symmetric=True))
    got = origin.load_payload(origin.save_payload(tmp_path / "c.npz", curves=cs,
                                                  norm="peak", offset=1.5, name="n"))
    assert (got["norm"], got["offset"], got["name"]) == ("peak", 1.5, "n")
    assert [c.label for c in got["curves"]] == [c.label for c in cs]
    assert got["curves"][1].visible is False
    np.testing.assert_array_equal(got["curves"][0].y, cs[0].y)
    gm = origin.load_payload(origin.save_payload(tmp_path / "m.npz", m=m))["map"]
    np.testing.assert_array_equal(gm.z, m.z)
    assert gm.levels == m.levels and gm.selection.to_dict() == m.selection.to_dict()


@pytest.mark.skipif(os.environ.get("AALTOVIEW_TEST_ORIGIN") != "1",
                    reason="opens Origin on this PC; set AALTOVIEW_TEST_ORIGIN=1 to run")
def test_curves_and_a_map_arrive_in_origin():
    from aaltoview import origin
    assert origin.available() is None
    ds = _cube()
    cs = E.curves_along(ds, E.Selection("kerr", x="x"), "freq", [0, 1, 2])
    book = origin.push_curves(cs, name="pytest curves")
    m, _ = E.make_map(ds, E.Selection("kerr", x="x", y="freq"))
    mbook = origin.push_map(m, name="pytest map")
    import originpro as op
    op.attach()
    try:
        wks = op.find_book("w", book)[0]
        assert wks.cols == 4                               # one X + three Y
        assert wks.to_list(1) == cs[0].y.tolist()
        ms = op.find_book("m", mbook)[0]
        assert ms.shape == (4, 2)
        assert tuple(ms.xymap) == (0.0, 1.0, 1.0, 4.0)
    finally:
        op.detach()
