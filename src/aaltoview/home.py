"""home.py -- where the viewer looks for AnalysisModules/ and LoadingScripts/.

The two folders hold drop-ins: an analysis module is a folder with a
module.toml, a loading script a .py file. The viewer is run in two ways, and
both must find them:

* from its own CHECKOUT (uv run aaltoview, an editable install): the folders
  sit at the top of the repository, where a developer drops new ones;
* INSTALLED as a plain package -- inside AaltoFlow's scan-core, which is how
  the lab PC and every installer user get it (Mission Control's "Data viewer"
  button, the measurement suite's Data tab). There is no checkout there. Until
  2026-10-14 such a viewer only knew a checkout it had once been started from,
  so on a PC with only AaltoFlow its Analysis menu was empty.

So the bundled modules and scripts now travel INSIDE the package
(site-packages/aaltoview/AnalysisModules, .../LoadingScripts -- pyproject.toml
maps the two top-level folders into the wheel), and a per-user folder takes
the lab's own drop-ins without anyone touching a venv.

search_dirs(kind) looks, in this order:
  1. AALTOVIEW_HOME/<kind>          an explicit override (a folder or a checkout)
  2. this viewer's own checkout      when it runs from one (and it is remembered)
  3. the remembered checkout         %LOCALAPPDATA%\\AaltoView\\checkout.txt: the
                                     last checkout this user ran -- possibly newer
                                     modules than the installed copy has
  4. the bundled folder              inside the installed package
  5. the user folder                 %APPDATA%\\AaltoView\\<kind> (created when the
                                     operator asks for it, see user_dir)
The first folder that has a module (or script) of a given name wins; later
copies are skipped and logged. No Qt here.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "AALTOVIEW_HOME"
MODULES = "AnalysisModules"
SCRIPTS = "LoadingScripts"

#: the installed package folder (site-packages/aaltoview, or src/aaltoview)
PACKAGE = Path(__file__).resolve().parent
#: the folder above src/ -- the repository for an editable install
HERE = PACKAGE.parents[1]


def is_checkout(path: Path) -> bool:
    path = Path(path)
    return ((path / "pyproject.toml").is_file() and (path / "src" / "aaltoview").is_dir()
            and (path / MODULES).is_dir())


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


def checkouts() -> list[Path]:
    """This viewer's own checkout (remembered for later), then the remembered
    one if it is another and still there."""
    out = []
    if is_checkout(HERE):
        _remember(HERE)
        out.append(HERE)
    try:
        known = Path(memo_file().read_text(encoding="utf-8").strip())
        if is_checkout(known) and known not in out:
            out.append(known)
    except Exception:                    # nothing remembered, or unreadable
        pass
    return out


def bundled(kind: str) -> Path:
    """The folder shipped inside the package (absent in an editable install,
    where the checkout's own folder plays that part)."""
    return PACKAGE / kind


def user_base() -> Path:
    """Roaming %APPDATA%\\AaltoView: the user's own drop-ins follow them to
    another lab PC with a roaming profile. ~/.aaltoview elsewhere."""
    base = os.environ.get("APPDATA")
    return Path(base) / "AaltoView" if base else Path.home() / ".aaltoview"


def user_dir(kind: str, create: bool = False) -> Path:
    """Where an operator drops their OWN modules or scripts. Not created just
    by looking: only when asked (the menu's "open folder" entry)."""
    d = user_base() / kind
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def search_dirs(kind: str) -> list[Path]:
    """Every existing folder of this kind, in search order, each once."""
    cands = []
    env = os.environ.get(HOME_ENV)
    if env:
        cands.append(Path(env) / kind)
    cands += [c / kind for c in checkouts()]
    cands += [bundled(kind), user_dir(kind)]
    out, seen = [], set()
    for d in cands:
        if not d.is_dir():
            continue
        key = os.path.normcase(str(d.resolve()))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def workspace_of(folder: Path) -> Path | None:
    """The checkout whose AnalysisModules/ holds this module folder, if any --
    there the module is a uv workspace member and can be started with
    `uv run --all-packages` (which installs what a new module needs)."""
    for p in Path(folder).parents:
        if p.name == MODULES and is_checkout(p.parent):
            return p.parent
    return None
