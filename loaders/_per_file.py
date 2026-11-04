"""Base for sources that publish one GRIB2 file per run, step and field (DWD, Environment Canada).

Unlike ECMWF and GFS there's no index to read: each file's URL is built from its run, step and field
name, so subclasses only say how.
"""

from pathlib import Path

import xarray as xr

from ._http import download_to, exists, find_latest_run
from .standard import shift_longitude


class PerFileGribLoader:
    """Downloads and caches per-field GRIB2 files, and opens them as xarray Datasets.

    Files are only downloaded if not already on disk, and are cached (decompressed) under
    `data_dir/<cache_name>/<run>/<step>h-<field>.grib2` (run as YYYYMMDDHH).

    Subclasses set `name`, `step_hours`, `max_step`, `run_hours`, `retention`, `cache_name`,
    `probe_field` (a field present at every step, used to check what's published) and `file_url`.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def file_url(self, run, step_hours, field):
        raise NotImplementedError

    def latest_run(self, step_hours=0, max_age=None):
        """Most recent run (naive UTC datetime) that has published the given step.

        Pass the last step you need, since steps are published one after another and a run can be
        partly available. Looks back as far as the server keeps runs by default.
        """
        return find_latest_run(lambda run: exists(self.file_url(run, step_hours, self.probe_field)), self.run_hours,
                               max_age or self.retention, f"{self.name} run with step {step_hours}h published")

    def download(self, run, step_hours, field):
        """Download one field for a run and step, skipping it if already on disk. Returns its path."""
        out = self.data_dir / self.cache_name / f"{run:%Y%m%d%H}" / f"{step_hours}h-{field}.grib2"
        if not out.exists():
            url = self.file_url(run, step_hours, field)
            download_to(out, url, timeout=120, decompress_bz2=url.endswith(".bz2"))
        return out

    def open_file(self, path, step_hours):
        """One downloaded file (of step step_hours) as a Dataset."""
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})

    def load_run(self, run, steps, field):
        """One run's forecast for a field over several steps, stacked along "step", with longitudes in -180-180."""
        datasets = [self.open_file(self.download(run, s, field), s) for s in steps]
        ds = xr.concat(datasets, dim="step", coords="different", compat="equals", join="exact")
        return shift_longitude(ds)
