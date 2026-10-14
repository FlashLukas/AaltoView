"""Shared by every test folder (tests/ and AnalysisModules/*/tests).

Free the test windows BEFORE Python starts shutting down.

The GUI tests build dozens of windows (viewer, fit windows, plots). A fixture
closes each one, but a window is kept alive by reference CYCLES (signals
connected to its own methods, pyqtgraph items pointing back at their view), so
it is not freed when the test drops it -- only when Python's cycle collector
runs. If that happens during interpreter shutdown, PySide6's atexit hook
(__moduleShutdown) and the collector tear Qt objects down in no particular
order: a pyqtgraph axis asks its label for a size after the label's C++ text
item is gone, and the run ends in a screenful of "Error calling Python override
of QGraphicsWidget::sizeHint() ... Internal C++ object (QGraphicsTextItem)
already deleted" (PySide6 6.11 + pyqtgraph 0.14, 2026-10-14; it needs enough
windows piled up, so one test file alone never shows it).

Not a bug in the viewer: a real program ends through app.exec(), and pyqtgraph
cleans up on aboutToQuit. A test session never quits, so we collect here, while
everything is still alive, and do pyqtgraph's quit step ourselves.
(Deleting every top-level widget instead does NOT work: pyqtgraph's context
menus are top-level widgets owned by their views, and deleting them one by one
aborts the process.)
"""

from __future__ import annotations

import gc
import sys


def pytest_sessionfinish(session, exitstatus):
    if "PySide6.QtWidgets" not in sys.modules:        # no GUI test ran: nothing to do
        return
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.processEvents()             # pending deleteLater()s and closes
    gc.collect()                    # the closed windows' cycles, freed in order now
    try:
        import pyqtgraph as pg
        pg.cleanup()                # what pyqtgraph runs on aboutToQuit
    except Exception:
        pass
