"""fmr_fit -- AaltoView analysis module: FMR resonances fitted on 1-D curves,
then the resonances fitted with a magnetic model (Kittel, anisotropy, damping).

    uv run python -m fmr_fit

model.py fits one curve, dispersion.py fits the resonances (both no Qt, tested
headless); app.py is the window. AaltoView sends curves here with 1D plots ->
Analysis. The viewer finds the module through ../../module.toml.
"""

from pathlib import Path

__version__ = "0.1.0"

#: The module's name and key, from module.toml -- the file the viewer reads too,
#: so the two can never disagree.
MANIFEST = Path(__file__).resolve().parents[2] / "module.toml"


def info() -> dict:
    from aaltoview.analysis_link import read_manifest
    return read_manifest(MANIFEST)
