"""segments.py -- axes that go UP and DOWN (a hysteresis loop), read back.

AaltoFlow's scan-core can sweep an axis 0 -> 100 -> -100 -> 100 mT: the same
field is measured more than once, on different branches of the loop. The file
says which RUN every point is on (the contract: AaltoFlow's
docs/DEVELOPER_NOTES.md section 4l):

    <dim>              the values AS VISITED -- not unique, not monotonic
    <dim>_segment      int32 run number per point (0 = first), non-decreasing
    <dim>_direction    int8 +1 (the run goes up) / -1 (down)
    <dim>.attrs        segments_json (one object per run), segment_source,
                       n_segments

Why it matters for a plot: drawn as ONE line, a loop folds back onto itself
and the two branches -- the physics -- are indistinguishable. Drawn into a
map, two values land in one cell and one of them silently wins. So the rule
everywhere in the viewer is: one line per run; a map shows ONE run at a time
(or every point against its point number); a held value names its run
("50 mT · down"); an average says whether it mixes up and down.

A run is a CONTIGUOUS block of the axis (the run number never decreases), so
"run k" is the index range first .. first + count - 1, and every reduction
that already works on index ranges works on runs unchanged.

Use isel, never sel, on a segmented dim: its index has duplicate values.

A file written before the contract existed (a typed list that turns back,
no `<dim>_segment`) is split at its reversals by `reversal_segments` -- the
same function as scan-core's recipe.reversal_segments, copied because
AaltoView must not import the instrument suite; tests/test_segments.py pins
the two together on the engine's own examples.

No Qt here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

#: what can be shown of a segmented image axis (Selection.runs)
#:   "lines" : every run, one line each (a 1-D plot only)
#:   "index" : every point against its point number (no folding, no gaps)
#:   k (int) : run k alone, drawn ascending
LINES, INDEX = "lines", "index"


@dataclass
class Segments:
    """The runs of one dimension."""
    seg: np.ndarray                  # run number per point (int)
    info: list[dict]                 # one dict per run (segments_json)
    source: str = "recipe"           # "recipe" | "reversals" | "values"

    @property
    def n(self) -> int:
        return len(self.info)

    def indices(self, k: int) -> np.ndarray:
        """Indices (into the dim) of run k, in measured order."""
        return np.flatnonzero(self.seg == int(k))

    def span(self, k: int) -> tuple[int, int]:
        """(first, last) index of run k, inclusive -- a run is contiguous."""
        idx = self.indices(k)
        return (int(idx[0]), int(idx[-1])) if idx.size else (0, 0)

    def direction(self, k: int) -> int:
        return int(self.info[k].get("direction", 1) or 1)

    def name(self, k: int) -> str:
        return segment_name(self.info, k)

    def name_at(self, i: int) -> str:
        """The run of point i, by name ("down", "seg 3 up")."""
        return self.name(int(self.seg[int(i)]))

    def ascending(self, k: int, values=None) -> np.ndarray:
        """Indices of run k ordered by increasing value: a down run reversed.
        How a map draws one run (its cells must ascend)."""
        idx = self.indices(k)
        if values is not None and idx.size > 1:
            v = np.asarray(values, dtype=float)[idx]
            return idx if v[-1] >= v[0] else idx[::-1]
        return idx if self.direction(k) > 0 else idx[::-1]

    def line_indices(self, k: int) -> np.ndarray:
        """Indices to DRAW run k as a line: a run that starts one step after
        the previous run ended (joined) is drawn from that end, so the loop
        is closed on screen as it was measured."""
        idx = self.indices(k)
        if k > 0 and idx.size and self.info[k].get("joined") and idx[0] > 0:
            idx = np.concatenate([[idx[0] - 1], idx])
        return idx

    def directions(self) -> np.ndarray:
        """+1 / -1 per point (the file's <dim>_direction)."""
        return np.asarray([self.direction(int(k)) for k in self.seg], dtype=int)


def segment_name(info: list, k: int) -> str:
    """'up' / 'down' for a two-run axis, 'seg 3 up' (1-based) for more."""
    way = "up" if info[k].get("direction", 1) > 0 else "down"
    return way if len(info) <= 2 else f"seg {k + 1} {way}"


def reversal_segments(values) -> tuple[np.ndarray, list[dict]] | None:
    """A numeric list that TURNS BACK (0, 50, 100, 50, 0, -50 ...) split into
    monotonic runs; None for one that never turns back (or is not numeric).

    A point belongs to the run of the step that LEADS INTO it, so the turning
    point ends the earlier run and every later run is `joined`, its `start`
    the turning value. A step of zero (a value typed twice) changes nothing.
    Identical to scan-core's recipe.reversal_segments (see the module note).
    """
    v = np.asarray(values)
    if v.dtype == object or v.ndim != 1 or v.size < 3:
        return None
    try:
        v = v.astype(float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(v)):
        return None
    steps = np.sign(np.diff(v))
    steps = steps[steps != 0]
    if steps.size == 0 or np.all(steps == steps[0]):
        return None                              # monotonic (the usual axis): fast
    seg = np.zeros(v.size, dtype=np.int32)
    dirs: list[int] = []
    k, cur = 0, 0
    for i in range(1, v.size):
        step = int(np.sign(v[i] - v[i - 1]))
        if step != 0 and cur != 0 and step != cur:
            k += 1
            dirs.append(cur)
        if step != 0:
            cur = step
        seg[i] = k
    dirs.append(cur if cur != 0 else 1)
    if k == 0:
        return None                              # monotonic: nothing to split
    info = []
    for j in range(k + 1):
        idx = np.flatnonzero(seg == j)
        first, last = int(idx[0]), int(idx[-1])
        start = float(v[first - 1]) if j > 0 else float(v[first])
        info.append({"start": start, "stop": float(v[last]),
                     "num": int(idx.size + (1 if j > 0 else 0)),
                     "direction": int(dirs[j]), "joined": int(j > 0),
                     "first": first, "count": int(idx.size)})
    return seg, info


def segments_of(obj, dim: str) -> Segments | None:
    """The runs of `dim` in a Dataset or DataArray, or None for an ordinary axis.

    From the file's own coordinates when it has them; else, for a numeric
    coordinate that turns back (an older file of a typed loop), split at its
    reversals exactly as the engine would have written it."""
    if obj is None or dim is None or dim not in getattr(obj, "coords", {}):
        return None
    name = f"{dim}_segment"
    coords = obj.coords
    if name in coords and coords[name].dims == (dim,):
        seg = np.asarray(coords[name].values).astype(int)
        try:
            info = json.loads(coords[dim].attrs.get("segments_json") or "[]")
        except (ValueError, TypeError):
            info = []
        n = int(seg.max()) + 1 if seg.size else 0
        if len(info) != n:
            # no (or a broken) JSON: the directions from the file's own
            # <dim>_direction, else from the values
            vals = np.asarray(coords[dim].values, dtype=float)
            dcoord = coords.get(f"{dim}_direction")
            info = []
            for k in range(n):
                idx = np.flatnonzero(seg == k)
                if dcoord is not None and idx.size:
                    d = int(np.asarray(dcoord.values)[idx[0]])
                else:
                    d = int(np.sign(vals[idx[-1]] - vals[idx[0]])) if idx.size > 1 else 1
                info.append({"direction": d or 1, "joined": int(k > 0),
                             "first": int(idx[0]) if idx.size else 0,
                             "count": int(idx.size)})
        return Segments(seg, info, str(coords[dim].attrs.get("segment_source", "recipe")))
    try:
        found = reversal_segments(np.asarray(coords[dim].values))
    except Exception:
        found = None
    if found is None:
        return None
    return Segments(np.asarray(found[0]).astype(int), found[1], "reversals")


def run_options(segs: Segments, line: bool) -> list[tuple[str, object]]:
    """(text, choice) for a chooser of a segmented axis. A line plot can
    draw every run; a map cannot (two values in one cell), so its first
    option -- the default -- is the first run."""
    opts = [("every run, one line each", LINES)] if line else []
    opts += [(f"{segs.name(k)}  ({k + 1} of {segs.n})", k) for k in range(segs.n)]
    opts.append(("all, by point #", INDEX))
    return opts


def default_choice(line: bool):
    return LINES if line else 0


def valid_choice(segs: Segments, choice, line: bool):
    """`choice` if it means something for this axis and plot, else the default."""
    if choice == LINES and line:
        return LINES
    if choice == INDEX:
        return INDEX
    if isinstance(choice, (int, np.integer)) and not isinstance(choice, bool) \
            and 0 <= int(choice) < segs.n:
        return int(choice)
    return default_choice(line)


def choice_text(segs: Segments, choice) -> str:
    """What a choice shows, in words, for labels ("down", "by point #")."""
    if choice == INDEX:
        return "by point #"
    if choice == LINES:
        return "every run"
    return segs.name(int(choice))
