# Loading scripts

A loading script corrects a measurement **as it is read**: the viewer's
**load with** list (Files panel, above the file list) applies the one you
choose to every file you open. Because it happens at loading, the map, the 1D
plots, every export, the notebook and the analysis modules all see the
corrected data.

Drop a `.py` file into this folder and it is listed (press **Refresh**, or the
list's ↻). Files whose name starts with `_` are not listed, so they can hold
helpers that scripts share. More folders can be added with the
`AALTOVIEW_LOADING_SCRIPTS` environment variable (separated by `;`).

## The contract

```python
"""One line: what it does (shown as the tooltip)."""
NAME = "My correction"                  # optional; else the file name

def load(ds, path):                     # xarray.Dataset, pathlib.Path (or None)
    ...                                 # correct ds
    return ds
```

`ds` arrives as AaltoFlow wrote it: a complex detector is still its
`<name>_real` / `<name>_imag` pair. Return a Dataset with the same layout.
Helpers are in `aaltoview.loading`: `frequency_dim(ds)`, `alias(f, f_rep)` and
`tr_moke_unfold(ds, f_rep_mhz, invert=False)`.

## The scripts here

| script | what it does |
|---|---|
| `trmoke_unfold_80MHz.py` | TR-MOKE, 80 MHz laser: conjugates the complex lock-in signal back on the frequencies whose alias f − n·80 MHz is negative |
| `trmoke_unfold_100MHz.py` | the same for a 100 MHz laser |

**Why unfold?** The laser samples the precession f_rep times a second, so the
lock-in sees it at the alias f − n·f_rep. It cannot tell a negative alias from
a positive one, and on those frequencies it records the complex conjugate: a
wave running the other way. In a spatial FFT (the Spin-wave FFT module) the
branch then jumps between +k and −k every f_rep/2.

Some details:
- Frequencies exactly on a harmonic or half-way between two carry no direction
  and are left alone.
- Real detectors (e.g. rf power) are not touched.
- **Unfold only once.** The Spin-wave FFT module has the same switch for data
  loaded without a script. Leave it off for data loaded with one of these,
  because unfolding twice undoes the correction.
- If the branch comes out at −k instead of +k, your lock-in uses the other
  sign convention. Copy a script and call
  `tr_moke_unfold(ds, 80.0, invert=True)`.
