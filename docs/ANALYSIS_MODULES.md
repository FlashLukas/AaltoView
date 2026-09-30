# Analysis modules

AaltoView shows data; **analysis modules** do something with it: a fit, a
correction, a model. Each module is a separate program with its own window.
The viewer sends curves to it from **1D plots → Analysis**.

```
AaltoView ──"Analysis" menu──►  FMR fit      (AnalysisModules/fmr-fit)
          ──────────────────►  <your module> (AnalysisModules/<name>)
```

## Drop in, and it is there

A module is a folder in **`AnalysisModules/`**. To add one, copy the folder in.
The next time the "Analysis" menu opens, the module is listed; there is
nothing to register or install by hand. The first **Start … and send**
installs the module's packages (e.g. lmfit), so that one start can take up to
a minute. After that a start takes a few seconds.

To remove a module, delete its folder. Module folders elsewhere on the PC can
be added with the `AALTOVIEW_MODULES` environment variable (folders separated
by `;`); they start with the viewer's own Python.

## Why separate programs

- A module that is busy (a long fit) or crashes cannot freeze the viewer or
  lose its curves.
- Several modules, or two windows of one module, can run at the same time.
- Each module brings its own dependencies (lmfit, ...). Somebody who only
  looks at data does not need them.

## What a module folder contains

```
AnalysisModules/kerr-fit/
    module.toml          what the "Analysis" menu shows (key, name, description)
    pyproject.toml       what the module needs (aaltoview, lmfit, ...)
    src/kerr_fit/        __init__.py, __main__.py, model.py (maths), app.py (window)
    tests/               run with everyone else's: uv run pytest -q
```

```toml
# module.toml
[module]
key = "kerr_fit"
name = "Kerr fit"
description = "Damped precession in TR-MOKE traces"
accepts = ["curves"]
python = "kerr_fit"            # started as: python -m kerr_fit
```

The repository's `pyproject.toml` makes every `AnalysisModules/*` folder a
member of one uv workspace, so the whole suite shares one `.venv` and one lock
file. For this reason **every folder in `AnalysisModules/` must have a
`pyproject.toml`**. A stray folder without one stops `uv` with an error that
names the folder.

## How the viewer and a module find each other

| Question | Answer | Where |
|---|---|---|
| Which modules are there? | every `AnalysisModules/*/module.toml` | `analysis_link.installed()` |
| How is one started? | `uv run --all-packages python -m <python>` from the AaltoView folder, with no console window; its output goes to `%LOCALAPPDATA%\AaltoView\analysis\<key>-start.log` | `analysis_link.launch()` |
| Which are running, and where? | a running module listens on a free port of 127.0.0.1 and writes a *beacon* file `{key, name, title, port, pid}` next to that log | `analysis_link.Listener` |
| How do the curves get there? | ZeroMQ request/reply with JSON, the same pattern as the AaltoFlow instrument services | `analysis_link.request` |

The requests are:

| `{"cmd": ...}` | Reply |
|---|---|
| `describe` | `{"ok": true, "key", "name", "title", "description", "accepts", "protocol", "pid"}` |
| `add_curves`, with `"curves": [...]` | `{"ok": true, "n": <number received>}` |
| `shutdown` | `{"ok": true}`, then the window closes |

An error is `{"ok": false, "error": "..."}`. The module always replies at once;
the work happens afterwards in its window.

A beacon is deleted when its module closes. If a module crashes, its beacon is
deleted the next time the viewer finds that process gone. If a module does
not start, the viewer's status line shows the last line of the start log and
where to find the whole log.

## What a curve carries

Each curve arrives as an `aaltoview.export.Curve`, exactly as it was frozen in
the viewer:

| Field | Meaning |
|---|---|
| `x`, `y` | the numbers; `y` is the part that was on screen (\|z\|, Re, Im, arg) |
| `z` | the **complex** values when the detector is complex, otherwise `None` |
| `x_name`, `x_unit`, `y_name`, `y_unit` | names and units from the file |
| `label`, `source` | the curve's label and its `.nc` file (`None` for an unsaved run) |
| `selection` | how the curve was cut from the cube (replayable) |
| `held` | `{dim: (value, unit)}`: where the curve was taken, e.g. `rf_freq = (8.0, "GHz")` |

Holes (NaN) from a running or aborted scan stay holes, so NaN-aware code is a
must.

## Writing a new module

```bash
uv run python tools/new_analysis_module.py kerr_fit "Kerr fit" "Damped precession in TR-MOKE traces"
uv run python -m kerr_fit              # or from AaltoView: 1D plots -> Analysis
```

This writes `AnalysisModules/kerr-fit/`, a working module that already
receives and plots curves. Then:

1. Put the maths in `model.py`, with no Qt, and test it (`tests/`). Check the
   fit against numbers you KNOW, such as simulated data with known parameters
   (see `AnalysisModules/fmr-fit/tests/test_fmr_model.py`).
2. Build the window in `app.py`. The one rule is that
   `add_curves(list[Curve])` exists. `run_module()` (in
   `aaltoview/apps/analysis.py`) handles the theme, the icon, the listener,
   and delivering curves on the GUI thread.
3. For results, reuse the viewer's exports: `aaltoview.export.write_*`,
   `save_figure`, and `aaltoview.origin.save_payload(..., table=...)` for
   Origin. Use the same three header rows (name / unit / comment) everywhere.

Conventions are the same as in the rest of AaltoView: NaN-aware, files opened
with `encoding="utf-8"`, printed text ASCII, and `set_theme()` before any
widget (run_module does this).
