"""The installed package carries the bundled analysis modules and loading
scripts (2026-10-14): an AaltoView installed into AaltoFlow's scan-core has no
checkout beside it, so what is not IN the wheel does not exist on the lab PC.

The wheel is built the way `uv` builds a git dependency: sdist first, then the
wheel from the sdist -- a file the sdist forgets never reaches the wheel.
"""

import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_every_subpackage_is_listed():
    """pyproject.toml lists the packages by hand (package-dir maps the two drop-in
    folders into the package): a new sub-package must not be left out."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = set(cfg["tool"]["setuptools"]["packages"])
    found = {".".join(p.parent.relative_to(ROOT / "src").parts)
             for p in (ROOT / "src" / "aaltoview").rglob("__init__.py")}
    assert found <= listed, f"add to [tool.setuptools] packages: {sorted(found - listed)}"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    out = tmp_path_factory.mktemp("dist")
    # --offline: tests stay offline (setuptools comes from uv's cache)
    r = subprocess.run([uv, "build", "--offline", "--out-dir", str(out), str(ROOT)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        text = r.stdout + r.stderr
        if "offline" in text.lower() or "network" in text.lower():
            pytest.skip("the build backend is not in uv's cache (offline)")
        pytest.fail(f"uv build failed:\n{text[-2000:]}")
    (whl,) = out.glob("aaltoview-*.whl")
    return set(zipfile.ZipFile(whl).namelist())


def test_the_wheel_carries_the_bundled_modules_and_scripts(wheel):
    pkg = "aaltoview/AnalysisModules"
    for mod in sorted(p.parent for p in (ROOT / "AnalysisModules").glob("*/module.toml")):
        for f in ("module.toml", "pyproject.toml"):
            assert f"{pkg}/{mod.name}/{f}" in wheel
        for code in (mod / "src").rglob("*.py"):
            assert f"{pkg}/{mod.name}/{code.relative_to(mod).as_posix()}" in wheel
    for script in (ROOT / "LoadingScripts").glob("*.py"):
        assert f"aaltoview/LoadingScripts/{script.name}" in wheel
    assert "aaltoview/apps/icon.svg" in wheel
    # not shipped: tests and build leftovers
    assert not any("/tests/" in n or ".egg-info" in n or "__pycache__" in n
                   for n in wheel if n.startswith(pkg))
