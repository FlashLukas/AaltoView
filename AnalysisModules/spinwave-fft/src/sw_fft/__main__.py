"""python -m sw_fft [--theme light]"""

from __future__ import annotations


def main(argv=None) -> int:
    from aaltoview.apps.analysis import run_module

    from . import info
    from .app import FFTWindow
    return run_module(info(), FFTWindow, argv)


if __name__ == "__main__":
    raise SystemExit(main())
