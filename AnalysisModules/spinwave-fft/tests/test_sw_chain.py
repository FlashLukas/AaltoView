"""The whole chain on tools/make_demo_data.py file 10 -- a permalloy stripe whose
k(f) was computed THERE, not with this module: map -> FFT per line -> peaks ->
Kalinikos-Slavin + Guslienko fit -> the stripe's parameters back."""

import sys
from pathlib import Path

import numpy as np
import pytest

from aaltoview import export as E
from aaltoview.data import load
from aaltoview.view import Slice
from sw_fft import fft as F
from sw_fft import inputs as I
from sw_fft import waveguide as W

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_data as DEMO  # noqa: E402


@pytest.fixture(scope="module")
def stripe(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    path = next(out.glob("*/*_py_stripe_trmoke.nc"))
    ds = load(path).load()
    m = E.make_map_data(ds, E.Selection("lockin", x="pos_x", y="rf_freq",
                                        slices={"pos_y": Slice("at", 0)}), source=path)
    return I.from_map(m)


def points(inp, ps=None):
    sp = F.spectra(inp.x, inp.rows, F.Settings(), inp.x_unit)
    peaks = F.all_peaks(sp, ps or F.PeakSettings(k_min=0.5, side="positive"))
    fsrc = I.guess_source(inp, "freq")
    bsrc = I.guess_source(inp, "field")
    assert fsrc.how == "y" and bsrc.how == "constant"   # the field is only in the comment
    bsrc.value = 20.0
    return sp, [W.Point(k=p.k, f=I.quantity(inp, p.line, fsrc), B=I.quantity(inp, p.line, bsrc),
                        line=p.line) for p in peaks]


def truth_specs(**free):
    specs = W.default_specs()
    for n in ("Ms", "A", "d", "w", "B"):
        specs[n] = W.Spec(DEMO.STRIPE[n], False, *W.PARAMS[n][4:])
    specs["n"].value, specs["theta"].value, specs["gamma"].value = 1.0, 90.0, DEMO.G
    for n, start in free.items():
        specs[n].value, specs[n].vary = start, True
    return specs


def test_the_input_is_a_map_of_lines_with_the_frequency_in_mhz(stripe):
    assert stripe.complex and stripe.rows.shape == (81, 201)
    assert (stripe.y_name, stripe.y_unit, stripe.x_unit) == ("rf_freq", "MHz", "um")
    assert I.quantity(stripe, 0, I.Source("y")) == pytest.approx(5.0)


def test_every_peak_is_where_the_stripe_puts_it(stripe):
    sp, pts = points(stripe)
    assert len(pts) == 81                                # every line is inside the band
    want = DEMO.stripe_k(np.array([p.f for p in pts]))
    got = np.array([p.k for p in pts])
    # within a tenth of the resolution 2 pi / 20 um
    assert np.max(np.abs(got - want)) < 0.1 * sp.resolution
    assert sp.resolution == pytest.approx(2 * np.pi / 20)


def test_the_fit_gives_the_stripe_back(stripe):
    _, pts = points(stripe)
    res = W.fit(pts, truth_specs(Ms=800.0))
    assert res.values["Ms"] == pytest.approx(DEMO.STRIPE["Ms"], rel=5e-3)
    assert res.rms < 0.02                                # GHz
    # Ms and the width together: w_eff is what the band bottom sees
    res = W.fit(pts, truth_specs(Ms=800.0, w=3.0))
    assert res.values["Ms"] == pytest.approx(DEMO.STRIPE["Ms"], rel=0.02)
    assert res.values["w"] == pytest.approx(DEMO.STRIPE["w"], rel=0.05)
    assert res.derived["w_eff"][0] > res.values["w"]     # dipolar pinning: wider
    # unpinned edges read the same data as a WIDER stripe: the effective width
    un = W.fit(pts, truth_specs(Ms=800.0, w=3.0), pinning="unpinned")
    assert un.values["w"] == pytest.approx(res.derived["w_eff"][0], rel=0.02)
    assert un.values["w"] > DEMO.STRIPE["w"] * 1.03
