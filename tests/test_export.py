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
    lines = rows.read_text(encoding="utf-8-sig").splitlines()
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


BOM = "﻿".encode("utf-8")          # the UTF-8 byte-order mark


def test_a_csv_carries_a_bom_so_excel_reads_the_units_but_a_dat_does_not(tmp_path):
    # Excel opens a .csv as the Windows codepage unless the file starts with a
    # UTF-8 byte-order mark, and "µm" / "°" come out as garbage. A .dat is
    # read by numpy / gnuplot / Origin, which stumble over a BOM, so it stays plain.
    ds = _cube()
    ds["x"].attrs["units"] = "µm"
    cs = E.curves_along(ds, E.Selection("kerr", x="x"), "freq", [0])
    csv_bytes = E.write_curves(tmp_path / "c.csv", cs).read_bytes()
    dat_bytes = E.write_curves(tmp_path / "c.dat", cs).read_bytes()
    assert csv_bytes.startswith(BOM)
    assert not dat_bytes.startswith(BOM)
    assert "µm" in csv_bytes.decode("utf-8-sig")
    m, _ = E.make_map(ds, E.Selection("kerr", x="x", y="freq", slices={"y": Slice("at", 0)}))
    assert E.write_map(tmp_path / "m.csv", m).read_bytes().startswith(BOM)
    assert not E.write_map(tmp_path / "m.dat", m).read_bytes().startswith(BOM)


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
    xyz = E.write_map(tmp_path / "m.csv", m, "xyz").read_text(encoding="utf-8-sig").splitlines()
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


# ───────────────────── deep clean 2026-09-28: proven bugs ─────────────────────

@pytest.mark.parametrize("y", [np.zeros(4),                        # a dead channel
                               np.full(4, 3.0),                     # a flat line
                               np.array([np.nan, 0.0, 1.0, 2.0])])  # first real point 0
@pytest.mark.parametrize("mode", ["peak", "minmax", "first"])
def test_the_notebook_normalises_edge_cases_like_the_viewer(y, mode):
    """The notebook carries its own normalize(); on a flat or zero curve it
    divided by zero (NaN / inf) where the viewer leaves the curve as it is, so
    the notebook drew a different curve from the screen."""
    ns: dict = {}
    exec(E._NB_HELPERS, ns)
    with np.errstate(all="ignore"):
        got = ns["normalize"](y, mode)
    np.testing.assert_array_equal(got, E.normalize(y, mode))


def test_curves_with_equal_numbers_on_different_axes_keep_their_own_x(tmp_path):
    """A row cut (along x) and a column cut (along y) of a square raster have
    the SAME coordinate numbers, 0..2 um. They were written with one shared X
    column named after the first curve, so the y cut was labelled as a function
    of x."""
    ds = xr.Dataset({"kerr": (("y", "x"), np.arange(9.0).reshape(3, 3))},
                    coords={"y": ("y", [0.0, 1.0, 2.0], {"units": "um"}),
                            "x": ("x", [0.0, 1.0, 2.0], {"units": "um"})})
    row = E.make_curve(ds, E.Selection("kerr", x="x", slices={"y": Slice("at", 1)}))
    col = E.make_curve(ds, E.Selection("kerr", x="y", slices={"x": Slice("at", 1)}))
    p = E.write_curves(tmp_path / "c.dat", [row, col])
    names = p.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert names == ["x", "kerr", "y", "kerr"]


def test_axis_labels_of_mixed_curves_do_not_claim_one_unit():
    """Overlaying a Kerr curve (mdeg) and a reflectivity curve (V) labelled the
    exported Y axis "signal (mdeg)" -- the unit of whichever curve came first."""
    ds = _cube()
    ds["refl"] = (("freq", "y", "x"), ds["kerr"].values * 0 + 1.0, {"units": "V"})
    k = E.make_curve(ds, E.Selection("kerr", x="y"))
    r = E.make_curve(ds, E.Selection("refl", x="y"))
    fig = E.figure_curves([k, r])
    assert "mdeg" not in fig.axes[0].get_ylabel()
    fig = E.figure_curves([k, E.make_curve(ds, E.Selection("kerr", x="y"))])
    assert fig.axes[0].get_ylabel() == "kerr (mdeg)"


def test_a_half_written_complex_pair_is_not_offered_as_complex():
    """A file with `s_real` (tagged as a pair) but no `s_imag` listed "s" --
    which then could not be opened -- and hid `s_real`, the data that IS there."""
    from aaltoview.view import detector, detector_names
    ds = xr.Dataset({"s_real": (("f",), np.arange(3.0),
                                {"complex_pair": "s", "complex_part": "real"})})
    assert detector_names(ds) == ["s_real"]
    assert detector(ds, "s_real").values.tolist() == [0.0, 1.0, 2.0]


# ─────────────────────────────── map reference ────────────────────────────────

def _vna_map(tmp_path) -> Path:
    """S21 = background(f) x (1 + line(f, B)): a cable/amplifier background
    that does not depend on the field, times the sample's response; at 500 mT
    the line is far above the band (22.6 GHz), so that row is the background
    up to the line's tail."""
    f = np.linspace(2.0, 6.0, 81)
    b = np.array([20.0, 40.0, 60.0, 80.0, 500.0])
    bg = (0.3 + 0.1 * f) * np.exp(-2j * np.pi * 1.7 * f)          # ripple-free, delay
    f0 = 28.0e-3 * np.sqrt(b * (b + 800.0))                        # Kittel, GHz
    line = 0.2 * 0.05 / (f0[:, None] - f[None, :] - 0.05j)
    z = bg[None, :] * (1 + line)
    pair = {"complex_pair": "s21"}
    ds = xr.Dataset({"s21_real": (("field", "freq"), z.real, {**pair, "complex_part": "real"}),
                     "s21_imag": (("field", "freq"), z.imag, {**pair, "complex_part": "imag"})},
                    coords={"field": ("field", b, {"units": "mT"}),
                            "freq": ("freq", f, {"units": "GHz"})})
    p = tmp_path / "vna.nc"
    ds.to_netcdf(p, engine="h5netcdf")
    return p


def test_a_map_divided_by_its_reference_row_is_the_sample_alone(tmp_path):
    ds = xr.open_dataset(_vna_map(tmp_path), engine="h5netcdf").load()
    f, b = ds["freq"].values, ds["field"].values
    f0 = 28.0e-3 * np.sqrt(b * (b + 800.0))
    sample = 1 + 0.2 * 0.05 / (f0[:, None] - f[None, :] - 0.05j)
    truth = sample / sample[-1]        # the 500 mT row still has the line's far tail
    sel = E.Selection("s21", x="freq", y="field", part="abs")
    m, _ = E.make_map(ds, sel, E.MapStyle(ref="row", ref_i=-1))   # the 500 mT row
    np.testing.assert_allclose(m.z, np.abs(truth), rtol=1e-9)
    assert m.z_name == "|s21| (÷ field = 500 mT)" and m.z_unit == ""
    # the phase of the RATIO: the delay's 1.7 turns per GHz are gone
    sel.part = "arg"
    m, _ = E.make_map(ds, sel, E.MapStyle(ref="row", ref_i=4))
    np.testing.assert_allclose(m.z, np.angle(truth), atol=1e-9)
    # subtract keeps the unit
    sel.part = "abs"
    m, _ = E.make_map(ds, sel, E.MapStyle(ref="row", ref_op="subtract"))
    assert m.z[-1] == pytest.approx(np.zeros(len(f)))


def test_median_and_derivative_divide_need_no_reference_row():
    z = np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [9.0, 2.0, 4.0]], dtype=complex)
    med = E.reference_values(z, E.MapStyle(ref="median_rows"), np.arange(3), np.arange(3))
    np.testing.assert_allclose(med, z / np.array([[2.0, 2.0, 4.0]]))
    # d/dx / z of exp(k x) is k, whatever the background amplitude
    x = np.linspace(0.0, 1.0, 201)
    y = np.array([0.0, 1.0])
    zz = np.array([3.0, 0.5])[:, None] * np.exp(2.0 * x)[None, :]
    dd = E.reference_values(zz, E.MapStyle(ref="dd_x"), x, y)
    assert np.isnan(dd[:, 0]).all() and np.isnan(dd[:, -1]).all()
    np.testing.assert_allclose(dd[:, 1:-1], 2.0, rtol=1e-4)
    # a hole stays a hole; a zero reference is a hole, not infinity
    z[0, 1] = np.nan
    z[2, 2] = 0.0
    out = E.reference_values(z, E.MapStyle(ref="row", ref_i=2), np.arange(3), np.arange(3))
    assert np.isnan(out[0, 1]) and np.isnan(out[:, 2]).all()


def test_the_notebook_recomputes_a_referenced_map(tmp_path):
    src = _vna_map(tmp_path)
    ds = xr.open_dataset(src, engine="h5netcdf").load()
    nb_dir = tmp_path / "analysis"
    nb_dir.mkdir()
    for style in (E.MapStyle(ref="row", ref_i=-1, norm="rows"),
                  E.MapStyle(ref="dd_y", log=True),
                  E.MapStyle(ref="median_columns", ref_op="subtract")):
        for part in ("abs", "arg"):
            m, _ = E.make_map(ds, E.Selection("s21", x="freq", y="field", part=part),
                              style, source=src)
            ns = _run_notebook(E.write_notebook(nb_dir / "v.ipynb", m=m, style=style))
            np.testing.assert_allclose(ns["z"], m.z, equal_nan=True)


def test_a_cut_of_a_referenced_map_is_the_line_on_the_map(tmp_path):
    """Row / Column -> 1D of a referenced map: the numbers of the map through
    the cursor, complex z referenced too (an analysis module gets the sample
    alone), and the notebook regenerates it."""
    src = _vna_map(tmp_path)
    ds = xr.open_dataset(src, engine="h5netcdf").load()
    ref = {"x": "freq", "y": "field", "mode": "row", "op": "divide", "i": -1}
    for style, r in ((E.MapStyle(ref="row", ref_i=-1), ref),
                     (E.MapStyle(ref="dd_x"), {**ref, "mode": "dd_x"})):
        m, _ = E.make_map(ds, E.Selection("s21", x="freq", y="field"), style)
        col = E.make_curve(ds, E.Selection("s21", x="freq", slices={"field": Slice("at", 1)},
                                           ref=r), source=src)          # along freq, 40 mT
        np.testing.assert_allclose(col.y, m.z[1], equal_nan=True)
        row = E.make_curve(ds, E.Selection("s21", x="field", slices={"freq": Slice("at", 30)},
                                           ref=r), source=src)          # along field
        np.testing.assert_allclose(row.y, m.z[:, 30], equal_nan=True)
    col = E.make_curve(ds, E.Selection("s21", x="freq", part="arg",
                                       slices={"field": Slice("at", 1)}, ref=ref), source=src)
    assert col.label == "field = 40 mT (÷ field = 500 mT)" and col.y_unit == "rad"
    np.testing.assert_allclose(np.angle(col.z), col.y)
    back = E.Selection.from_dict(col.selection.to_dict())
    assert back.ref == ref
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    ns = _run_notebook(E.write_notebook(nb_dir / "c.ipynb", curves=[col, row]))
    (_, y1), (_, y2) = ns["lines"]
    np.testing.assert_allclose(y1, col.y)
    np.testing.assert_allclose(y2, row.y, equal_nan=True)
