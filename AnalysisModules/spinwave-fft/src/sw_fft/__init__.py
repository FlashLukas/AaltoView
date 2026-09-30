"""sw_fft -- AaltoView analysis module: the spatial FFT of spin-wave maps, the
wavevector of every line, and the dispersion of a stripe fitted to them
(Kalinikos-Slavin, Guslienko dipolar pinning).

    uv run python -m sw_fft

fft.py transforms and finds peaks, waveguide.py is the dispersion and its fit,
inputs.py turns what AaltoView sends into lines (all three no Qt, tested
headless); app.py is the window. AaltoView sends a map with Map -> Analysis,
or curves with 1D plots -> Analysis. The viewer finds the module through
../../module.toml.
"""

from pathlib import Path

__version__ = "0.1.0"

#: The module's name and key, from module.toml -- the file the viewer reads too.
MANIFEST = Path(__file__).resolve().parents[2] / "module.toml"


def info() -> dict:
    from aaltoview.analysis_link import read_manifest
    return read_manifest(MANIFEST)
