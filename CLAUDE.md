# AaltoView -- instructions for Claude Code (and for any contributor)

AaltoView is the data viewer of the [AaltoFlow](https://github.com/FlashLukas/AaltoFlow)
lab-automation suite: it opens the N-dimensional `.nc` files AaltoFlow writes,
shows any two dimensions as a map or curves (holding or averaging the rest), and
exports figures, data, Origin projects and Jupyter notebooks. Developed by the
NanoSpin group, Aalto University. Read `README.md` first.

## Layout
- `src/aaltoview/data.py` -- reading the files (complex pairs, the `dims`
  attribute, autosave names). This is the **contract with AaltoFlow's scan
  engine**: change both together.
- `src/aaltoview/view.py` -- the cube reduction (hold / average per dimension,
  complex reduced BEFORE |z| or arg). No Qt; tested headless.
- `src/aaltoview/export.py` -- figures, `.dat`/`.csv`, notebooks (the notebook
  carries its OWN reduction code; a test executes it and compares with view.py).
- `src/aaltoview/origin.py` -- Origin push, run in a child process.
- `src/aaltoview/apps/` -- the Qt window (`viewer.py`), shared widgets, theme.
- AaltoFlow's scan-core imports `view`, `data` and `apps.theme` from here: ONE
  copy of the code, and the SAME `COLORS` dict for both apps.

## Rules
- NaN-aware everywhere: a scan that is still running or was aborted has holes.
- The theme mechanism and the other shared conventions are AaltoFlow's
  (`docs/DEVELOPER_NOTES.md` in that repo); never rebind `COLORS`.
- Open files with `encoding="utf-8"`; printed text stays ASCII.
- Tests stay offline: `uv sync --extra gui; uv run pytest -q`. The Origin test
  runs only with `AALTOVIEW_TEST_ORIGIN=1`.
- After a GUI change: `uv run python tools/render_docs.py`, and look at both themes.

## Private notes
`CLAUDE.local.md` is for personal working notes: gitignored, loaded by Claude
Code on top of this file. Never commit it, and never put PC names, user
accounts, serial numbers or personal e-mail addresses into tracked files.
