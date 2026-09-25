"""Theme: dark and light palettes with the shared amber accent.

This is the AaltoFlow suite palette. scan-core's apps/theme.py RE-EXPORTS this
module instead of keeping a copy: the viewer is embedded in the measurement
suite, and there must be ONE COLORS dict, or set_theme("light") in the suite
would leave the viewer's widgets painting dark colours.

Same mechanism as clMag-control: every widget sources its colors from the
module-level ``COLORS`` dict. ``set_theme("dark"|"light")`` swaps the palette IN
PLACE (mutates the same dict object), so anything that did ``from apps.theme
import COLORS`` (or the ``C`` alias) keeps seeing the active values. Call
``set_theme`` once at startup, before building widgets. ``build_stylesheet()``
returns the Qt style sheet for the active palette; ``apply_palette(app)`` sets a
matching QPalette (use with the Fusion style so scroll areas and popups follow).
Startup-only — there is no live toggle.
"""

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

# ---- the two palettes (neutral colors shared across all modules) -----------

DARK = {
    "bg": "#0e1013", "panel": "#171a1f", "panel_hi": "#1e222a", "border": "#2a2f37",
    "text": "#e8eaed", "muted": "#8b929c", "ok": "#3ddc84", "danger": "#ff5c5c",
    "grid": "#20242b", "pressed": "#402a12", "code_bg": "#0a0c0f",
    "accent": "#ff9e2c", "accent_hi": "#ffb454", "accent_dim": "#a8641a",
}
LIGHT = {
    "bg": "#f3f4f6", "panel": "#ffffff", "panel_hi": "#eceef1", "border": "#d3d7de",
    "text": "#1b1e24", "muted": "#6b7280", "ok": "#1a9e57", "danger": "#d13b3b",
    "grid": "#e3e6ea", "pressed": "#e6d7bd", "code_bg": "#f0f1f3",
    "accent": "#d9821a", "accent_hi": "#c26a12", "accent_dim": "#a05e12",
}
THEMES = {"dark": DARK, "light": LIGHT}

# active palette (starts dark); mutated in place so importers follow.
COLORS = dict(DARK)
C = COLORS          # back-compat alias — same object, so C[...] also follows set_theme


def set_theme(name: str) -> None:
    COLORS.clear()
    COLORS.update(THEMES.get((name or "dark").lower(), DARK))


def apply_palette(app: QtWidgets.QApplication) -> None:
    """Set a QPalette from COLORS. Use with app.setStyle('Fusion')."""
    c = COLORS
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.Window, QtGui.QColor(c["bg"]))
    p.setColor(QtGui.QPalette.WindowText, QtGui.QColor(c["text"]))
    p.setColor(QtGui.QPalette.Base, QtGui.QColor(c["panel_hi"]))
    p.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(c["panel"]))
    p.setColor(QtGui.QPalette.Text, QtGui.QColor(c["text"]))
    p.setColor(QtGui.QPalette.Button, QtGui.QColor(c["panel_hi"]))
    p.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(c["text"]))
    p.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(c["panel"]))
    p.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(c["text"]))
    p.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(c["muted"]))
    p.setColor(QtGui.QPalette.Highlight, QtGui.QColor(c["accent"]))
    p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#201400"))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(c["muted"]))
    app.setPalette(p)


def build_stylesheet() -> str:
    c = COLORS
    return f"""
QWidget#root {{ background:{c['bg']}; }}
QLabel {{ color:{c['text']}; }}
QFrame#card {{ background:{c['panel']}; border:1px solid {c['border']}; border-radius:10px; }}
QFrame#axis {{ background:{c['panel_hi']}; border:1px solid {c['border']}; border-radius:8px; }}
QLabel#title {{ color:{c['accent']}; font-size:18px; font-weight:800; letter-spacing:2px; }}
QLabel#tag {{ color:{c['muted']}; font-size:11px; font-weight:700; letter-spacing:1px; }}
QLabel#big {{ color:{c['accent']}; font-size:22px; font-weight:800; }}
QListWidget, QPlainTextEdit {{ background:{c['code_bg']}; border:1px solid {c['border']};
    border-radius:8px; color:{c['text']}; }}
QPushButton {{ background:{c['panel_hi']}; color:{c['text']}; border:1px solid {c['border']};
    border-radius:7px; padding:5px 10px; font-weight:600; }}
QPushButton:hover {{ border-color:{c['accent']}; }}
QPushButton:pressed {{ background:{c['pressed']}; }}
QPushButton#primary {{ background:{c['accent']}; color:#201400; border:none; }}
QPushButton#primary:hover {{ background:{c['accent_hi']}; }}
QPushButton#danger {{ background:{c['danger']}; color:#1a1a1a; border:none; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background:{c['panel_hi']}; color:{c['text']};
    border:1px solid {c['border']}; border-radius:6px; padding:2px 4px; }}
QComboBox QAbstractItemView {{ background:{c['panel_hi']}; border:1px solid {c['border']};
    selection-background-color:{c['pressed']}; }}
QCheckBox {{ color:{c['text']}; }}
QProgressBar {{ background:{c['bg']}; border:1px solid {c['border']}; border-radius:6px;
    text-align:center; color:{c['text']}; }}
QProgressBar::chunk {{ background:{c['accent']}; border-radius:5px; }}
"""


#: The viewer's own icon, next to this file (shipped as package data).
ICON_FILE = Path(__file__).resolve().parent / "icon.svg"

#: Its OWN taskbar identity, not scan-core's: the viewer is a separate program
#: and gets its own taskbar button rather than merging with the suite's.
APP_ID = "Aalto.AaltoView"


def apply_window_icon(app) -> None:
    """Title-bar, Alt-Tab and taskbar icon (the suite's apply_window_icon).

    On Windows both steps are needed: setWindowIcon for Qt, and an explicit
    AppUserModelID -- without it every pythonw process is grouped under one
    "Python" button with Python's icon. The ID is claimed only AFTER an icon
    is set: Windows caches the icon against the ID, and one run that claims it
    with no icon leaves it blank for good (AaltoFlow docs/DEVELOPER_NOTES.md gotcha #32).
    Never fatal.
    """
    if not ICON_FILE.exists():
        return
    app.setWindowIcon(QtGui.QIcon(str(ICON_FILE)))
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:          # not Windows, or the call is unavailable
        pass


def apply(app: QtWidgets.QApplication):
    """Convenience: Fusion + palette + stylesheet for the ACTIVE theme.

    Call set_theme(...) first if you want a non-default palette.
    """
    # Numbers with a '.' decimal point and no thousands separator, whatever the
    # Windows locale (AaltoFlow docs/DEVELOPER_NOTES.md gotcha #18): on the lab PC a scan range of
    # 400 nm showed as "400,000". Must happen before any spin box is built.
    loc = QtCore.QLocale.c()
    loc.setNumberOptions(QtCore.QLocale.OmitGroupSeparator)
    QtCore.QLocale.setDefault(loc)
    app.setStyle("Fusion")
    apply_palette(app)
    app.setStyleSheet(build_stylesheet())


#: Startup theme when nothing else says otherwise. There is no .ini here, so
#: this constant plus each app's --theme flag IS the setting. Lived in
#: scan_builder.py until the measurement suite needed it too.
DEFAULT_THEME = "dark"


# back-compat: some callers imported STYLESHEET as a constant
STYLESHEET = build_stylesheet()
