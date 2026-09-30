"""new_analysis_module.py -- start a new AaltoView analysis module.

    uv run python tools/new_analysis_module.py kerr_fit "Kerr fit" "Damped precession in TR-MOKE traces"

Writes AnalysisModules/<key with - for _>/: module.toml (what the viewer's
"Analysis" menu lists), pyproject.toml (what the module needs), __init__.py,
__main__.py, model.py (the maths, no Qt), app.py (a window that lists and plots
the curves it receives) and a test. That is all: the viewer finds the folder
by itself, and the first "Start ... and send" installs its packages. Then fill
in model.py and app.py; docs/ANALYSIS_MODULES.md says what the viewer expects.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PYPROJECT = '''[project]
name = "aaltoview-{folder}"
version = "0.1.0"
description = "AaltoView analysis module: {description}"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "aaltoview[gui]",
]

# AaltoView finds this module through module.toml, next to this file

[project.scripts]
aaltoview-{folder} = "{key}.__main__:main"

[tool.uv.sources]
aaltoview = {{ workspace = true }}

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
'''

MANIFEST = '''# What AaltoView's "Analysis" menu shows. The folder is found by itself.
[module]
key = "{key}"
name = "{name}"
description = "{description}"
accepts = ["curves"]
python = "{key}"            # started as: python -m {key}
'''

INIT = '''"""{key} -- AaltoView analysis module: {description}

    uv run python -m {key}

model.py is the maths (no Qt, tested headless); app.py is the window.
AaltoView sends curves here with 1D plots -> Analysis.
"""

from pathlib import Path

__version__ = "0.1.0"

MANIFEST = Path(__file__).resolve().parents[2] / "module.toml"


def info() -> dict:
    from aaltoview.analysis_link import read_manifest
    return read_manifest(MANIFEST)
'''

MAIN = '''"""python -m {key} [--theme light]"""

from __future__ import annotations


def main(argv=None) -> int:
    from aaltoview.apps.analysis import run_module

    from . import info
    from .app import Window
    return run_module(info(), Window, argv)


if __name__ == "__main__":
    raise SystemExit(main())
'''

MODEL = '''"""model.py -- the maths of {name}. No Qt here: everything is tested headless.

Each curve arrives as an aaltoview.export.Curve: x, y (the part the viewer
showed), z (the COMPLEX values when the detector is complex, else None),
x_name / x_unit / y_name / y_unit, label, source (.nc file), held
({{dim: (value, unit)}}: where the curve was taken, e.g. rf_freq = 8 GHz).
"""

from __future__ import annotations

import numpy as np


def summary(x, y) -> dict[str, float]:
    """Placeholder analysis: replace with the real thing (and its test)."""
    ok = np.isfinite(x) & np.isfinite(y)          # NaN-aware: scans have holes
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    i = int(np.argmax(np.abs(y)))
    return {{"x_at_max": float(x[i]), "max": float(y[i]), "mean": float(np.mean(y))}}
'''

APP = '''"""app.py -- the {name} window."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from aaltoview.apps.theme import C
from aaltoview.apps.viewer import TAB10, _plain_axes
from aaltoview.export import Curve, axis_title

from . import model as M


class Window(QtWidgets.QWidget):
    """run_module() calls add_curves() with every batch the viewer sends."""

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.resize(1200, 800)
        self.curves: list[Curve] = []
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        title = QtWidgets.QLabel("{title}"); title.setObjectName("title")
        v.addWidget(title)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self.show_curve)
        split.addWidget(self.list)
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        _plain_axes(self.plot)
        split.addWidget(self.glw)
        split.setSizes([300, 900])
        v.addWidget(split, 1)
        self.status = QtWidgets.QLabel("Waiting for curves: in AaltoView, 1D plots -> Analysis")
        self.status.setStyleSheet(f"color:{{C['muted']}};")
        v.addWidget(self.status)

    def add_curves(self, curves: list[Curve]):
        self.curves += curves
        self.list.addItems([c.label for c in curves])
        self.list.setCurrentRow(len(self.curves) - len(curves))

    def show_curve(self, i: int):
        self.plot.clear()
        if not 0 <= i < len(self.curves):
            return
        c = self.curves[i]
        self.plot.plot(c.x, c.y, pen=pg.mkPen(TAB10[0], width=2), connect="finite")
        self.plot.setLabel("bottom", axis_title(c.x_name, c.x_unit))
        self.plot.setLabel("left", axis_title(c.y_name, c.y_unit))
        s = M.summary(c.x, c.y)
        self.status.setText("   ".join(f"{{k}} = {{v:.6g}}" for k, v in s.items()))
'''

TEST = '''"""{name}: the maths, headless."""

import numpy as np

from {key} import model as M


def test_summary_skips_holes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, np.nan, 5.0, 2.0])
    assert M.summary(x, y)["x_at_max"] == 2.0
'''


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__)
        return 2
    key, name, description = argv
    if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
        print("ERROR key must be a Python name in lower case: letters, digits, _")
        return 2
    folder = key.replace("_", "-")
    base = ROOT / "AnalysisModules" / folder
    if base.exists():
        print(f"ERROR {base} exists already")
        return 1
    fill = dict(key=key, folder=folder, name=name, title=name.upper(),
                description=description.replace('"', "'"))
    files = {base / "module.toml": MANIFEST, base / "pyproject.toml": PYPROJECT,
             base / "src" / key / "__init__.py": INIT,
             base / "src" / key / "__main__.py": MAIN, base / "src" / key / "model.py": MODEL,
             base / "src" / key / "app.py": APP, base / "tests" / f"test_{key}_model.py": TEST}
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.format(**fill), encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}")

    # nothing to register: the workspace glob (AnalysisModules/*) and the
    # viewer's folder scan pick the new folder up by themselves
    print(f"next: uv run python -m {key}   (or AaltoView: 1D plots -> Analysis)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
