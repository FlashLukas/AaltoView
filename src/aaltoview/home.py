"""home.py -- where the AaltoView CHECKOUT is: AnalysisModules/, LoadingScripts/.

Run from its own checkout (uv run aaltoview, an editable install), the viewer
finds both folders next to its code. Installed as a plain dependency -- inside
AaltoFlow's scan-core, whose Mission Control "Data viewer" button and catalogue
start it from there -- its code lives in site-packages, with no modules and no
scripts beside it: that viewer's Analysis menu was empty (2026-10-08).

So a viewer that runs from a checkout REMEMBERS it (one line in
%LOCALAPPDATA%\\AaltoView\\checkout.txt), and an installed copy uses the
remembered one. AALTOVIEW_HOME overrides both. No Qt here.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "AALTOVIEW_HOME"
#: the folder above src/ -- the repository for an editable install
HERE = Path(__file__).resolve().parents[2]


def is_checkout(path: Path) -> bool:
    path = Path(path)
    return ((path / "pyproject.toml").is_file() and (path / "src" / "aaltoview").is_dir()
            and (path / "AnalysisModules").is_dir())


def memo_file() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "AaltoView" / "checkout.txt"


def _remember(path: Path) -> None:
    try:
        f = memo_file()
        if f.exists() and f.read_text(encoding="utf-8").strip() == str(path):
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(path), encoding="utf-8")
    except Exception:                    # a read-only profile: just do not remember
        pass


def repo() -> Path:
    """The checkout whose AnalysisModules/ and LoadingScripts/ this viewer
    offers. Falls back to HERE (where nothing may be) when none is known."""
    env = os.environ.get(HOME_ENV)
    if env:
        return Path(env)
    if is_checkout(HERE):
        _remember(HERE)
        return HERE
    try:
        known = Path(memo_file().read_text(encoding="utf-8").strip())
        if is_checkout(known):
            return known
    except Exception:
        pass
    return HERE
