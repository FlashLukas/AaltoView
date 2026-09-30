"""python -m fmr_fit [--theme light]"""

from __future__ import annotations


def main(argv=None) -> int:
    from aaltoview.apps.analysis import run_module

    from . import info
    from .app import FitWindow
    return run_module(info(), FitWindow, argv)


if __name__ == "__main__":
    raise SystemExit(main())
