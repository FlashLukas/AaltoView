"""The spatial FFT and its peaks, against waves whose k is KNOWN."""

import numpy as np
import pytest

from sw_fft import fft as F


def _wave(k0, x, amp=1.0, decay=None):
    z = amp * np.exp(1j * k0 * x)
    if decay:
        z = z * np.exp(-x / decay)
    return z


def _peak(x, v, s=None, **ps):
    s = s or F.Settings()
    k, spec = F.line_spectrum(x, v, s)
    return F.find_peaks(k, np.abs(spec), F.PeakSettings(**ps))


def test_a_wave_travelling_to_plus_x_is_at_plus_k_and_its_amplitude_is_kept():
    x = np.linspace(0, 20, 161)                         # um, dx = 0.125
    k0 = 2 * np.pi / 1.7                                # lambda = 1.7 um
    for win in F.WINDOWS:
        p = _peak(x, _wave(k0, x, 3.0), F.Settings(window=win, pad=8))
        assert len(p) == 1
        assert p[0].k == pytest.approx(k0, abs=0.01 * 2 * np.pi / 20), win
        assert p[0].amplitude == pytest.approx(3.0, rel=0.05 if win != "none" else 0.1), win
    # the other way: -k0, nothing at +k0
    p = _peak(x, _wave(-k0, x), n_peaks=2, min_rel=0.2)
    assert len(p) == 1 and p[0].k == pytest.approx(-k0, abs=0.01)


def test_a_real_signal_is_its_mirror_image():
    x = np.linspace(0, 20, 161)
    k0 = 2 * np.pi / 2.5
    p = _peak(x, np.cos(k0 * x), F.Settings(part="real"), n_peaks=2)
    assert sorted(q.k for q in p) == pytest.approx([-k0, k0], abs=0.01)
    assert [q.amplitude for q in p] == pytest.approx([0.5, 0.5], rel=0.05)


def test_k_units_and_length_units():
    x_nm = np.linspace(0, 20000, 161)                   # the same scan in nm
    k0 = 2 * np.pi / 1.7
    v = _wave(k0, x_nm * 1e-3)
    k, spec = F.line_spectrum(x_nm, v, F.Settings(k_unit="1/um"), x_unit="nm")
    p = F.find_peaks(k, np.abs(spec), F.PeakSettings())
    assert p[0].k == pytest.approx(1 / 1.7, abs=2e-3)   # 1/lambda
    assert F.length_scale("µm") == 1.0 and F.length_scale("mm") == 1e3
    assert F.length_scale("GHz") is None


def test_holes_and_uneven_steps_do_not_move_the_peak():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 20, 150))                # uneven
    k0 = 2 * np.pi / 1.3
    v = _wave(k0, x, decay=8.0)
    v[40:45] = np.nan                                   # an aborted stretch
    p = _peak(x, v)
    assert p[0].k == pytest.approx(k0, abs=0.03)
    assert np.isnan(F.line_spectrum(x[:3], v[:3], F.Settings())[1]).all()


def test_detrend_removes_the_offset_the_k_range_the_dc():
    x = np.linspace(0, 20, 161)
    k0 = 2 * np.pi / 2.0
    v = 5.0 + 0.3 * x + _wave(k0, x)                    # a large offset and a slope
    p = _peak(x, v, F.Settings(detrend="linear"))
    assert p[0].k == pytest.approx(k0, abs=0.01)
    p = _peak(x, v, F.Settings(detrend="none"), k_min=1.0)
    assert p[0].k == pytest.approx(k0, abs=0.02)


def test_padding_interpolates_but_the_resolution_is_the_length():
    x = np.linspace(0, 10, 81)
    s = F.spectra(x, [_wave(3.0, x)], F.Settings(pad=1))
    s8 = F.spectra(x, [_wave(3.0, x)], F.Settings(pad=8))
    assert s8.k.size == 8 * s.k.size
    assert s.resolution == pytest.approx(2 * np.pi / 10) == s8.resolution


def test_a_map_gives_one_line_per_row_on_one_k_axis():
    x = np.linspace(0, 20, 161)
    ks = [2.0, 3.0, 4.0]
    rows = [_wave(k, x) for k in ks] + [np.full(x.size, np.nan)]   # a row not measured
    sp = F.spectra(x, rows, F.Settings())
    peaks = F.all_peaks(sp, F.PeakSettings())
    assert [p.line for p in peaks] == [0, 1, 2]
    assert [p.k for p in peaks] == pytest.approx(ks, abs=0.01)
    assert np.isnan(sp.F[3]).all()


def test_several_peaks_strongest_first_and_the_side():
    x = np.linspace(0, 40, 321)
    v = _wave(2.0, x, 1.0) + _wave(5.0, x, 0.5) + _wave(-3.5, x, 0.8)
    p = _peak(x, v, n_peaks=3)
    assert [round(q.k, 1) for q in p] == [2.0, -3.5, 5.0]
    p = _peak(x, v, n_peaks=3, side="positive")
    assert [round(q.k, 1) for q in p] == [2.0, 5.0]
    # Hann: half AMPLITUDE at +-1 bin (1.44 bins is the half-POWER width)
    assert p[0].fwhm == pytest.approx(2 * np.pi / 40 * 2.0, rel=0.1)
