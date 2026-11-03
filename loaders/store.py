"""Keeps the latest set of forecasts on disk as small UK-cropped files, and deletes everything older.

`ForecastStore.refresh()` gets the latest run of the benchmark, and of every other source out to 72 hours
after the benchmark's run time (beyond the benchmark's own forecast, which is compared over the hours
it covers), as standard forecasts (see `loaders.standard`):
- each source's run is only downloaded if it isn't stored already, so re-running is quick and only
  uses DataHub allowance when the Met Office has a new run
- raw downloads go to a temporary `raw/` folder that's deleted as soon as each source is saved
- each run is saved as one compressed NetCDF file, `standard/<source>/<YYYYMMDDHH>.nc`, with values
  packed to 16 bits (0.01 mm rain, 0.01 m s-1 wind); a full set is ~30 MB, mostly the benchmark
- afterwards, older runs and the raw caches of earlier versions of these loaders are deleted
- only one refresh runs at a time (others wait, then reuse what it stored), and a source that fails
  is skipped with a warning (and listed in `failures`), keeping its previous stored run if that still
  overlaps the benchmark's forecast

Run `python -m loaders.store [data_dir]` to refresh from the command line.
"""

import fcntl
import re
import shutil
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

from . import BENCHMARK, MetOfficeLoader, forecast_sources
from .standard import last_step_by, run_time

UK_EXTENT = (-13, 4, 49.5, 61)
STANDARD_DIR = "standard"
RAW_DIR = "raw"
# Raw caches the loaders used to keep directly in the data folder (DataHub orders are "o<digits>")
LEGACY_RAW_DIRS = ["ecmwf", "gfs", "aigfs", "icon-eu", "gdps", "global-deterministic-10km", "uk-deterministic-2km"]

# 16-bit packing: rain to 655 mm since the run started, wind to ±327 m s-1, both to 0.01
ENCODING = {
    "rain_mm": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": 65535, "zlib": True, "complevel": 4},
    "u10": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 4},
    "v10": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 4},
}


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class ForecastStore:
    """Standard forecasts saved under `data_dir/standard/`, keeping the newest `keep` runs of each source."""

    def __init__(self, data_dir="data", extent=UK_EXTENT, keep=1):
        self.data_dir = Path(data_dir)
        self.extent = tuple(extent)
        self.keep = keep
        self.failures = {}  # source name -> error message, from the last refresh

    @property
    def raw_dir(self):
        return self.data_dir / RAW_DIR

    def path(self, name, run):
        return self.data_dir / STANDARD_DIR / _slug(name) / f"{pd.Timestamp(run):%Y%m%d%H}.nc"

    def runs(self, name):
        """Stored runs of a source (naive UTC Timestamps), oldest first."""
        folder = self.data_dir / STANDARD_DIR / _slug(name)
        return sorted(pd.to_datetime(p.stem, format="%Y%m%d%H") for p in folder.glob("*.nc")) if folder.exists() else []

    def load(self, name, run=None):
        """A stored standard forecast (the newest if run is None), or None if it isn't stored."""
        if run is None:
            runs = self.runs(name)
            if not runs:
                return None
            run = runs[-1]
        path = self.path(name, run)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        if tuple(ds.attrs.pop("extent", ())) != self.extent and name != BENCHMARK:
            return None  # saved for a different box
        ds.attrs["step_hours"] = int(ds.attrs["step_hours"])
        return ds

    def save(self, ds):
        """Save a standard forecast, packed and compressed. Returns its path."""
        path = self.path(ds.attrs["source"], ds.attrs["run"])
        path.parent.mkdir(parents=True, exist_ok=True)
        # GRIB packing can leave rain a hair below zero, which unsigned packing can't hold
        out = ds.assign(rain_mm=ds.rain_mm.clip(min=0)).assign_attrs(extent=list(self.extent))
        tmp = path.with_suffix(".part")
        out.to_netcdf(tmp, encoding=ENCODING, engine="h5netcdf")
        tmp.rename(path)
        return path

    def refresh(self, order_id=None, sources=None, met_office=None, horizon_hours=72, progress=None):
        """The latest set of forecasts, benchmark first, downloading only runs not already stored.

        The benchmark is the latest run of a DataHub order (the first active one by default). Each other
        source uses its latest run that reaches `horizon_hours` (as a step), with steps up to
        `horizon_hours` after the benchmark's run time. A source that fails is skipped with a warning and
        recorded in `failures`, using its previous stored run if that still overlaps the benchmark's
        forecast. Older runs are then deleted (see `prune`). `progress` (e.g. `print`) is called with a
        line per source as it's reused or downloaded.
        """
        self._progress = progress or (lambda message: None)
        met_office = met_office or MetOfficeLoader(data_dir=self.raw_dir)
        sources = sources if sources is not None else forecast_sources(self.raw_dir)
        self.failures = {}
        with self._lock():
            order_id = order_id or met_office.list_orders()[0]["orderId"]
            bm_run = pd.Timestamp(met_office.latest_order_run(order_id))
            benchmark = self._get(BENCHMARK, bm_run, None, lambda: met_office.load_standard_latest(order_id))
            until = bm_run + timedelta(hours=horizon_hours)
            bm_end = pd.Timestamp(benchmark.valid_time.values[-1])

            forecasts = {BENCHMARK: benchmark}
            for name, source in sources.items():
                try:
                    run = pd.Timestamp(source.latest_run(step_hours=last_step_by(source, horizon_hours)))
                    last_step = last_step_by(source, int((until - run) / timedelta(hours=1)))
                    forecasts[name] = self._get(name, run, run + timedelta(hours=last_step),
                                                lambda: source.load_standard(run.to_pydatetime(), last_step,
                                                                             extent=self.extent))
                except Exception as e:
                    previous = self.load(name)
                    if previous is not None and not (run_time(previous) <= bm_end
                                                     and pd.Timestamp(previous.valid_time.values[-1]) >= bm_run):
                        previous = None  # too old to compare with this benchmark
                    kept = (f"keeping its stored {previous.attrs['run']} run" if previous is not None
                            else "no stored run overlapping the benchmark")
                    self.failures[name] = f"{type(e).__name__}: {e}"
                    warnings.warn(f"{name} failed ({self.failures[name]}); {kept}", stacklevel=2)
                    if previous is not None:
                        forecasts[name] = previous
            self.prune()
        return forecasts

    @contextmanager
    def _lock(self):
        """Hold the data folder's refresh lock, waiting for any other refresh (in any process) to finish."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_dir / ".refresh.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _get(self, name, run, last_valid, fetch):
        """A run from the store, or fetched, saved, and its raw files deleted. A stored run that stops
        before last_valid (when the benchmark now runs later) is fetched again."""
        ds = self.load(name, run)
        if ds is None or (last_valid is not None and pd.Timestamp(ds.valid_time.values[-1]) < last_valid):
            self._progress(f"{name}: downloading {run:%Y-%m-%d %H}Z run...")
            start = time.monotonic()
            try:
                self.save(fetch())
            finally:
                shutil.rmtree(self.raw_dir, ignore_errors=True)
            ds = self.load(name, run)
            self._progress(f"{name}: saved in {time.monotonic() - start:.0f} s")
        else:
            self._progress(f"{name}: {run:%Y-%m-%d %H}Z run already stored")
        return ds

    def prune(self):
        """Delete all but the newest `keep` runs of each source, plus raw caches. Returns the paths deleted."""
        deleted = []
        standard = self.data_dir / STANDARD_DIR
        for folder in (standard.iterdir() if standard.exists() else []):
            for path in sorted(folder.glob("*.nc"))[:-self.keep]:
                path.unlink()
                deleted.append(path)
        for folder in [RAW_DIR, *LEGACY_RAW_DIRS, *(d.name for d in self.data_dir.glob("o*")
                                                      if re.fullmatch(r"o\d+", d.name))]:
            path = self.data_dir / folder
            if path.is_dir():
                shutil.rmtree(path)
                deleted.append(path)
        return deleted

    def size_mb(self):
        """Total size of everything in the data folder (MB)."""
        return sum(p.stat().st_size for p in self.data_dir.rglob("*") if p.is_file()) / 1e6


if __name__ == "__main__":
    store = ForecastStore(sys.argv[1] if len(sys.argv) > 1 else "latestForecasts/data")
    for name, ds in store.refresh(progress=lambda message: print(message, flush=True)).items():
        print(f"{name}: {pd.Timestamp(ds.attrs['run']):%Y-%m-%d %H}Z run, {ds.valid_time.size} steps")
    for name, error in store.failures.items():
        print(f"{name} FAILED: {error}")
    print(f"Data folder: {store.size_mb():.0f} MB")
