"""data.py -- reading back what the AaltoFlow scan engine (scan-core) wrote.

Complex detectors (a VNA's S-parameters, a lock-in's X+iY) are carried as
complex in memory but stored as two real variables, `<id>_real` and
`<id>_imag`. That is a deliberate trade.

h5netcdf WILL write a complex array -- it round-trips through Python exactly --
but it warns that the file is then not conforming netCDF-4 and "might not be
readable by other netcdf tools". Lab data does not stay in Python: it gets
opened in MATLAB, in Igor, with ncdump, and by collaborators who did not choose
your stack. A file that only your own scripts can read is a worse default than
one extra function call, so the split is the default and this is the function
call.

    from aaltoview.data import as_complex, load
    ds = load("out/fmr_map.nc")
    s21 = as_complex(ds, "s21")          # complex DataArray (field, vna_freq)
    abs(s21).plot()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import xarray as xr

#: "153752_builder_scan" -> "153752", the HHMMSS the autosave puts in front
_STAMP = re.compile(r"^(\d{6})_")


def complex_names(ds: xr.Dataset) -> list[str]:
    """Ids in `ds` that are stored as a real/imaginary pair."""
    out = []
    for name, var in ds.data_vars.items():
        pair = var.attrs.get("complex_pair")
        if pair and var.attrs.get("complex_part") == "real":
            out.append(pair)
    return out


def as_complex(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Recombine `<name>_real` + `<name>_imag` into one complex DataArray.

    Raises KeyError with the available names rather than a bare one, because
    the usual mistake is asking for `s21` when the file calls it `S21`.
    """
    re_name, im_name = f"{name}_real", f"{name}_imag"
    if re_name not in ds or im_name not in ds:
        if name in ds:                       # already complex, or plain real
            return ds[name]
        raise KeyError(
            f"no complex pair '{name}' in this dataset; "
            f"available: {sorted(complex_names(ds)) or 'none'}")
    da = ds[re_name] + 1j * ds[im_name]
    da.attrs.update({k: v for k, v in ds[re_name].attrs.items()
                     if k not in ("complex_part", "complex_pair")})
    da.name = name
    return da


def to_complex_dataset(ds: xr.Dataset) -> xr.Dataset:
    """A copy of `ds` with every real/imaginary pair merged into one variable.

    Convenient for analysis in a notebook. Do NOT write the result back out with
    `to_netcdf` -- that is exactly the non-conforming file this avoids.
    """
    merged = ds.copy()
    for name in complex_names(ds):
        merged[name] = as_complex(ds, name)
        merged = merged.drop_vars([f"{name}_real", f"{name}_imag"])
    return merged


def load(path: str | Path) -> xr.Dataset:
    """Open a scan written by the engine. Pairs are left split; see as_complex."""
    return xr.open_dataset(path, engine="h5netcdf")


# ─────────────────────── what is in a data folder ─────────────────────────────

@dataclass
class Summary:
    """One line of the viewer's file list. Read from the header, never the data."""
    path: Path
    name: str
    measured: datetime
    dims: list[str]                 # outer -> inner, as the scan ran
    sizes: dict[str, int]
    detectors: list[str]
    n_points: int | None = None
    seconds: float | None = None
    comment: str = ""
    error: str = ""                 # set instead of raising: one bad file must
                                    # not hide the other two hundred

    @property
    def shape_text(self) -> str:
        return " x ".join(str(self.sizes[d]) for d in self.dims)


def _measured(path: Path) -> datetime:
    """When it was measured: the autosave name says so exactly
    (<data dir>/<YYYY-MM-DD>/<HHMMSS>_<name>.nc) and survives copying to another
    PC, which the file's modification time does not. Anything else falls back
    to the modification time."""
    m = _STAMP.match(path.stem)
    if m:
        try:
            day = datetime.strptime(path.parent.name, "%Y-%m-%d")
            t = datetime.strptime(m.group(1), "%H%M%S")
            return day.replace(hour=t.hour, minute=t.minute, second=t.second)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def summarize(path: str | Path) -> Summary:
    from .view import detector_names         # view imports data; import late

    path = Path(path)
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            order = [d for d in str(ds.attrs.get("dims", "")).split(",") if d in ds.sizes]
            dims = order + [d for d in ds.sizes if d not in order]
            n = ds.attrs.get("n_points")
            secs = ds.attrs.get("seconds")
            return Summary(
                path=path, name=str(ds.attrs.get("name") or path.stem),
                measured=_measured(path), dims=dims,
                sizes={d: int(ds.sizes[d]) for d in dims},
                detectors=detector_names(ds),
                n_points=None if n is None else int(n),
                seconds=None if secs is None else float(secs),
                comment=str(ds.attrs.get("comment", "")))
    except Exception as exc:
        return Summary(path=path, name=path.stem, measured=_measured(path), dims=[],
                       sizes={}, detectors=[], error=f"{exc.__class__.__name__}: {exc}")


def find_measurements(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Every measurement under `folder`, newest first.

    Skips the autosave's temporary `.writing.nc`: that file is being written
    right now, and opening it could catch it half-finished.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = folder.rglob("*.nc") if recursive else folder.glob("*.nc")
    found = [p for p in files if p.is_file() and not p.name.endswith(".writing.nc")]
    return sorted(found, key=lambda p: (_measured(p), p.name), reverse=True)
