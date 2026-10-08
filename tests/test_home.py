"""Where the AaltoView checkout is (home.py): an installed copy -- inside
AaltoFlow's scan-core, started by Mission Control or the catalogue -- offers
the checkout's analysis modules and loading scripts, not an empty menu
(2026-10-08)."""

from pathlib import Path

from aaltoview import analysis_link as AL
from aaltoview import home, loading


def _checkout(root: Path) -> Path:
    (root / "src" / "aaltoview").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'aaltoview'\n", encoding="utf-8")
    mod = root / "AnalysisModules" / "demo-fit"
    mod.mkdir(parents=True)
    (mod / "module.toml").write_text('[module]\nkey = "demo"\nname = "Demo fit"\n'
                                     'python = "demo_fit"\n', encoding="utf-8")
    (root / "LoadingScripts").mkdir()
    (root / "LoadingScripts" / "flip.py").write_text(
        '"""Flip."""\ndef load(ds, path):\n    return ds\n', encoding="utf-8")
    return root


def test_this_checkout_is_found_and_remembered(tmp_path, monkeypatch):
    monkeypatch.delenv(home.HOME_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert home.is_checkout(home.HERE)           # the tests run from the checkout
    assert home.repo() == home.HERE
    assert home.memo_file().read_text(encoding="utf-8") == str(home.HERE)


def test_an_installed_copy_uses_the_remembered_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(home.HOME_ENV, raising=False)
    monkeypatch.delenv(AL.MODULES_ENV, raising=False)
    monkeypatch.delenv(loading.SCRIPTS_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    site = tmp_path / "site-packages"                 # an install: no modules beside it
    site.mkdir()
    monkeypatch.setattr(home, "HERE", site)
    assert AL.installed() == []                       # nothing known yet: empty, as it was
    repo = _checkout(tmp_path / "AaltoView_dev")
    home.memo_file().parent.mkdir(parents=True)
    home.memo_file().write_text(str(repo), encoding="utf-8")
    assert home.repo() == repo
    assert [m.name for m in AL.installed()] == ["Demo fit"]
    assert repo / "LoadingScripts" in loading.script_dirs()
    argv, _ = AL.launch_command(AL.installed()[0])
    if argv[0].lower().endswith(("uv", "uv.exe")):
        assert argv[argv.index("--project") + 1] == str(repo)   # started in ITS workspace


def test_a_stale_memo_is_ignored_and_the_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(home, "HERE", site)
    home.memo_file().parent.mkdir(parents=True)
    home.memo_file().write_text(str(tmp_path / "deleted_worktree"), encoding="utf-8")
    monkeypatch.delenv(home.HOME_ENV, raising=False)
    assert home.repo() == site                        # not a checkout any more
    repo = _checkout(tmp_path / "elsewhere")
    monkeypatch.setenv(home.HOME_ENV, str(repo))
    assert home.repo() == repo
