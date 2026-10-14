"""Where the viewer finds analysis modules and loading scripts (home.py).

An installed copy -- inside AaltoFlow's scan-core, the lab PC, every installer
user -- has no checkout: its Analysis menu was empty there (2026-10-14). Now the
bundled modules travel inside the package, and the search order is
AALTOVIEW_HOME, own checkout, remembered checkout, bundled, user folder.
"""

from pathlib import Path

import pytest

from aaltoview import analysis_link as AL
from aaltoview import home, loading


def _module(folder: Path, key: str, name: str) -> Path:
    (folder / "src").mkdir(parents=True)
    (folder / "module.toml").write_text(f'[module]\nkey = "{key}"\nname = "{name}"\n'
                                        f'python = "{key}"\n', encoding="utf-8")
    return folder


def _script(folder: Path, stem: str, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.py"
    p.write_text(f'"""{name}."""\nNAME = "{name}"\ndef load(ds, path):\n    return ds\n',
                 encoding="utf-8")
    return p


def _checkout(root: Path, module_name: str = "Demo fit") -> Path:
    (root / "src" / "aaltoview").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'aaltoview'\n", encoding="utf-8")
    _module(root / "AnalysisModules" / "demo-fit", "demo", module_name)
    _script(root / "LoadingScripts", "flip", "Flip")
    return root


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """An installed aaltoview: a package folder in site-packages with its
    bundled modules, no checkout anywhere, empty profile folders."""
    for v in (home.HOME_ENV, AL.MODULES_ENV, loading.SCRIPTS_ENV):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    pkg = tmp_path / "venv" / "Lib" / "site-packages" / "aaltoview"
    _module(pkg / "AnalysisModules" / "vna-fmr-fit", "fmr_fit", "VNA-FMR fit")
    _script(pkg / "LoadingScripts", "unfold", "Unfold")
    monkeypatch.setattr(home, "PACKAGE", pkg)
    monkeypatch.setattr(home, "HERE", pkg.parents[1])
    return pkg


def test_this_checkout_is_found_and_remembered(tmp_path, monkeypatch):
    monkeypatch.delenv(home.HOME_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert home.is_checkout(home.HERE)           # the tests run from the checkout
    assert home.checkouts()[0] == home.HERE
    assert home.memo_file().read_text(encoding="utf-8") == str(home.HERE)
    assert home.HERE / "AnalysisModules" in home.search_dirs(home.MODULES)


def test_an_installed_copy_offers_its_bundled_modules(installed):
    """The lab PC's case: nothing remembered, no AALTOVIEW_HOME -- the menu is
    NOT empty, and the loading scripts are there too."""
    assert home.checkouts() == []
    assert [m.name for m in AL.installed()] == ["VNA-FMR fit"]
    assert [s.name for s in loading.available()] == ["Unfold"]
    # started with the viewer's own Python, the module's src/ on the path
    # (no uv workspace in a venv), and run from its own folder
    m = AL.installed()[0]
    argv, env = AL.launch_command(m)
    assert argv[1:] == ["-m", "fmr_fit"]
    assert str(m.folder / "src") in env["PYTHONPATH"]
    assert AL.launch_cwd(m) == str(m.folder)
    assert not home.user_dir(home.MODULES).exists()      # looking does not create it


def test_the_user_folder_adds_modules_and_scripts(installed):
    user = home.user_dir(home.MODULES, create=True)
    assert user.is_dir() and user.parent.parent.name == "roaming"
    _module(user / "my-fit", "my_fit", "My fit")
    _script(home.user_dir(home.SCRIPTS), "fix", "Fix")
    assert [m.name for m in AL.installed()] == ["My fit", "VNA-FMR fit"]
    assert [s.name for s in loading.available()] == ["Fix", "Unfold"]


def test_search_order_and_first_found_wins(installed, tmp_path, monkeypatch, caplog):
    """Same key in every place: the earlier one wins, the others are logged."""
    user = home.user_dir(home.MODULES, create=True)
    _module(user / "fmr", "fmr_fit", "FMR (user copy)")
    remembered = _checkout(tmp_path / "AaltoView_dev")
    _module(remembered / "AnalysisModules" / "fmr", "fmr_fit", "FMR (checkout)")
    home.memo_file().parent.mkdir(parents=True)
    home.memo_file().write_text(str(remembered), encoding="utf-8")
    override = tmp_path / "override"
    _module(override / "AnalysisModules" / "fmr", "fmr_fit", "FMR (override)")

    def names():
        return {m.key: m.name for m in AL.installed()}

    assert names()["fmr_fit"] == "FMR (checkout)"            # remembered > bundled > user
    monkeypatch.setenv(home.HOME_ENV, str(override))
    assert home.search_dirs(home.MODULES) == [
        override / "AnalysisModules", remembered / "AnalysisModules",
        installed / "AnalysisModules", user]
    with caplog.at_level("INFO", logger="aaltoview.analysis_link"):
        assert names()["fmr_fit"] == "FMR (override)"
    assert sum("already found" in r.getMessage() for r in caplog.records) == 3
    assert names()["demo"] == "Demo fit"                     # the checkout's own one too
    # a checkout's module is a workspace member: started with uv from there
    demo = next(m for m in AL.installed() if m.key == "demo")
    assert home.workspace_of(demo.folder) == remembered
    argv, _ = AL.launch_command(demo)
    if argv[0].lower().endswith(("uv", "uv.exe")):
        assert argv[argv.index("--project") + 1] == str(remembered)


def test_a_duplicate_loading_script_is_logged(installed, caplog):
    _script(home.user_dir(home.SCRIPTS), "unfold", "Unfold (mine)")
    with caplog.at_level("INFO", logger="aaltoview.loading"):
        got = loading.available()
    assert [s.name for s in got] == ["Unfold"]                # the bundled one came first
    assert any("already found" in r.getMessage() for r in caplog.records)


def test_a_stale_memo_is_ignored(installed):
    home.memo_file().parent.mkdir(parents=True)
    home.memo_file().write_text(str(installed.parent / "deleted_worktree"), encoding="utf-8")
    assert home.checkouts() == []
    assert [m.name for m in AL.installed()] == ["VNA-FMR fit"]


def test_the_empty_menu_names_the_user_folder(installed, monkeypatch):
    """No module anywhere: the hint points at the user's folder, not at a path
    inside a venv that nobody should edit."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    from aaltoview.apps.analysis import AnalysisMenu
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(home, "PACKAGE", installed.parent / "nothing-bundled")
    menu = AnalysisMenu(None, get_curves=lambda: [], say=lambda *a, **k: None)
    menu._fill()
    texts = [a.text() for a in menu.actions()]
    assert "no analysis modules found" in texts
    hint = next(t for t in texts if t.startswith("(drop one into"))
    assert str(home.user_dir(home.MODULES)) in hint and "site-packages" not in hint
    assert "Open my analysis-modules folder" in texts
    menu.deleteLater()
