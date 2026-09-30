"""tools/make_demo_data.py: seeded, so the files are the same every time --
also when it runs several times in one process (the tests do)."""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_demo_data as DEMO  # noqa: E402


def test_two_runs_give_the_same_files(tmp_path):
    DEMO.main([str(tmp_path / "a")])
    DEMO.main([str(tmp_path / "b")])
    for fa in sorted((tmp_path / "a").glob("*/*.nc")):
        fb = tmp_path / "b" / fa.relative_to(tmp_path / "a")
        with (xr.open_dataset(fa, engine="h5netcdf") as A,
              xr.open_dataset(fb, engine="h5netcdf") as B):
            for v in A.data_vars:
                assert np.array_equal(A[v].values, B[v].values, equal_nan=True), (fa.name, v)
