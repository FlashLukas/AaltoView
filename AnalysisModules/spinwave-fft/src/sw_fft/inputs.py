"""inputs.py -- what arrives from AaltoView, as lines to transform. No Qt.

A MAP (Map tab -> Analysis) is the natural input: pos_x x rf_freq, one line per
frequency. Curves (1D plots -> Analysis) sent together are stacked into one
input when they share their x; the value they differ in (the dim each was held
at) becomes the second axis. Either way an Input is x + rows + y.

Where does a peak's frequency and field come from? From the input's y axis if
it is a frequency (or a field), else from a held dim of that kind, else from a
value the operator types (a TR-MOKE file that has the field only in its
comment). `quantity()` decides by the unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aaltoview.export import Curve, MapData

#: unit -> (kind, factor to GHz or mT)
_UNITS = {"Hz": ("freq", 1e-9), "kHz": ("freq", 1e-6), "MHz": ("freq", 1e-3),
          "GHz": ("freq", 1.0), "THz": ("freq", 1e3),
          "T": ("field", 1e3), "mT": ("field", 1.0), "uT": ("field", 1e-3),
          "µT": ("field", 1e-3), "Oe": ("field", 0.1), "G": ("field", 0.1),
          "kOe": ("field", 100.0), "A/m": ("field", 4e-7 * np.pi * 1e3),
          "kA/m": ("field", 4e-4 * np.pi * 1e3)}


def unit_kind(unit: str) -> tuple[str | None, float]:
    """('freq', GHz per unit) / ('field', mT per unit) / (None, 1)."""
    return _UNITS.get((unit or "").strip(), (None, 1.0))


@dataclass
class Input:
    x: np.ndarray
    rows: np.ndarray                  # (n_lines, n_x), complex or real
    y: np.ndarray                     # one value per line
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    z_name: str
    z_unit: str
    label: str
    source: str | None = None
    held: dict[str, tuple[float, str]] = field(default_factory=dict)   # common to all lines
    complex: bool = False
    detector: str = ""                # the detector's own name ("lockin", not "|lockin|")

    def line_label(self, i: int) -> str:
        return f"{self.y_name} = {self.y[i]:g} {self.y_unit}".strip()


def from_map(m: MapData) -> Input:
    rows = m.z if m.z is not None else m.values
    return Input(x=np.asarray(m.x, float), rows=np.asarray(rows), y=np.asarray(m.y, float),
                 x_name=m.x_name, x_unit=m.x_unit, y_name=m.y_name, y_unit=m.y_unit,
                 z_name=m.z_name, z_unit=m.z_unit, label=m.label, source=m.source,
                 held=dict(m.held), complex=m.z is not None,
                 detector=m.selection.detector)


def from_curves(curves: list[Curve]) -> list[Input]:
    """Curves on one x grid -> one Input (y = the held dim they differ in, else
    their number); a curve on its own grid -> an Input of its own."""
    groups: list[list[Curve]] = []
    for c in curves:
        for g in groups:
            c0 = g[0]
            if (c0.x.shape == c.x.shape and np.allclose(c0.x, c.x, equal_nan=True)
                    and c0.x_unit == c.x_unit and (c0.z is None) == (c.z is None)):
                g.append(c)
                break
        else:
            groups.append([c])
    out = []
    for g in groups:
        c0 = g[0]
        dims = [d for d in c0.held if all(d in c.held for c in g)]
        vary = [d for d in dims if len({c.held[d][0] for c in g}) > 1]
        if vary:
            yd = vary[0]
            y = np.array([c.held[yd][0] for c in g])
            yname, yunit = yd, c0.held[yd][1]
        else:
            y = np.arange(len(g), dtype=float)
            yname, yunit = "curve", ""
        common = {d: c0.held[d] for d in dims if d not in vary}
        rows = np.array([c.z if c.z is not None else c.y for c in g])
        out.append(Input(x=np.asarray(c0.x, float), rows=rows, y=y, x_name=c0.x_name,
                         x_unit=c0.x_unit, y_name=yname, y_unit=yunit, z_name=c0.y_name,
                         z_unit=c0.y_unit, label=c0.label if len(g) == 1 else
                         f"{len(g)} curves along {yname}", source=c0.source,
                         held=common, complex=c0.z is not None,
                         detector=c0.selection.detector))
    return out


@dataclass
class Source:
    """Where a quantity (frequency, field) comes from: 'y', a held dim, or
    'constant' (the typed value, in GHz or mT)."""
    how: str = "constant"
    value: float = float("nan")


def guess_source(inp: Input, kind: str) -> Source:
    if unit_kind(inp.y_unit)[0] == kind:
        return Source("y")
    for d, (_, u) in inp.held.items():
        if unit_kind(u)[0] == kind:
            return Source(d)
    return Source("constant")


def quantity(inp: Input, line: int, src: Source) -> float:
    """The frequency (GHz) or field (mT) of one line, per its source."""
    if src.how == "constant":
        return float(src.value)
    if src.how == "y":
        return float(inp.y[line]) * unit_kind(inp.y_unit)[1]
    if src.how in inp.held:
        v, u = inp.held[src.how]
        return float(v) * unit_kind(u)[1]
    return float("nan")
