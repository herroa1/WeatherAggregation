"""Loader for ECMWF open data: the latest global forecasts from the IFS and AIFS models.

Free, with no account or API key needed (licence CC BY 4.0, so credit ECMWF when showing the data):
https://www.ecmwf.int/en/forecasts/datasets/open-data
Runs are at 00/06/12/18 UTC on a global 0.25° lat/lon grid, and data appears ~6-9 hours after the
run time. The ECMWF portal only keeps the last few days of runs; the AWS mirror (`source="aws"`)
keeps them from 2023 onwards.

Layout: `<YYYYMMDD>/<HH>z/<model>/0p25/oper/<run>-<step>h-oper-fc.grib2`, one GRIB2 file per run
and step holding every parameter. Each has a `.index` file next to it (one JSON line per field,
with its byte offset and length), so single parameters are downloaded with HTTP range requests.
Parameters use ECMWF short names, e.g. "2t", "10u", "10v", "msl", "tp" (see
https://codes.ecmwf.int/grib/param-db/).
"""

import json
from datetime import timedelta
from pathlib import Path

import xarray as xr

from ._http import download_to, exists, find_latest_run, session
from .standard import by_valid_time, crop, standard_dataset, standard_steps

SOURCES = {
    "ecmwf": "https://data.ecmwf.int/forecasts",
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
}
# Display name, step spacing (hours) and the last step with that spacing, for each model
MODELS = {"ifs": ("ECMWF IFS", 3, 144), "aifs-single": ("ECMWF AIFS", 6, 360)}
# How long each source keeps runs (None: from 2023 onwards)
RETENTION = {"ecmwf": timedelta(days=4), "aws": None}
# Units "tp" (total precipitation) comes in, and the factor to mm: IFS uses metres, AIFS kg m-2 (= mm)
TP_TO_MM = {"m": 1000, "kg m**-2": 1}


class ECMWFLoader:
    """Downloads and caches single-parameter ECMWF open-data GRIB2 files, and opens them as xarray Datasets.

    Files are only downloaded if not already on disk, and are cached under
    `data_dir/ecmwf/<model>/<run>/<step>h-<param>.grib2` (run as YYYYMMDDHH).
    `model` is "ifs" (physics-based) or "aifs-single" (machine-learning).
    """

    def __init__(self, data_dir="data", model="ifs", source="ecmwf"):
        self.data_dir = Path(data_dir)
        self.model = model
        self.name, self.step_hours, self.max_step = MODELS[model]
        self.run_hours = 6
        self.retention = RETENTION[source]
        self.base_url = SOURCES[source]

    def file_url(self, run, step_hours, suffix):
        """URL of a run and step's file, where suffix is "grib2" or "index"."""
        return (f"{self.base_url}/{run:%Y%m%d}/{run:%H}z/{self.model}/0p25/oper/"
                f"{run:%Y%m%d%H}0000-{step_hours}h-oper-fc.{suffix}")

    def latest_run(self, step_hours=0, max_age=timedelta(days=4)):
        """Most recent run (naive UTC datetime) that has published the given step.

        Pass the last step you need, since steps are published one after another and a run can be
        partly available.
        """
        return find_latest_run(lambda run: exists(self.file_url(run, step_hours, "index")), self.run_hours, max_age,
                               f"{self.name} run with step {step_hours}h published")

    def list_params(self, run, step_hours=0):
        """Index entries for a run and step, one dict per field (with "param", "levtype", "levelist", ...)."""
        r = session.get(self.file_url(run, step_hours, "index"), timeout=60)
        r.raise_for_status()
        return [json.loads(line) for line in r.text.splitlines() if line]

    def download(self, run, step_hours, param):
        """Download one single-level parameter for a run and step, skipping it if already on disk. Returns its path."""
        out = self.data_dir / "ecmwf" / self.model / f"{run:%Y%m%d%H}" / f"{step_hours}h-{param}.grib2"
        if not out.exists():
            fields = [f for f in self.list_params(run, step_hours) if f["param"] == param and "levelist" not in f]
            if len(fields) != 1:
                raise ValueError(f"Expected one single-level {param!r} field at step {step_hours}h, found {len(fields)}")
            start, length = fields[0]["_offset"], fields[0]["_length"]
            download_to(out, self.file_url(run, step_hours, "grib2"), timeout=120,
                        headers={"Range": f"bytes={start}-{start + length - 1}"})
        return out

    def load_run(self, run, steps, param):
        """One run's forecast for a parameter over several steps, stacked along "step"."""
        paths = [self.download(run, s, param) for s in steps]
        datasets = [xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}) for p in paths]
        return xr.concat(datasets, dim="step", coords="different", compat="equals", join="exact")

    def load_latest(self, steps, params):
        """The latest run that has all of `steps`, as a dict of Datasets keyed by parameter."""
        run = self.latest_run(max(steps))
        return {p: self.load_run(run, steps, p) for p in params}

    def load_standard(self, run, last_step, extent=None):
        """A run's standard forecast (see `loaders.standard`) for steps 0 to last_step, optionally cropped
        to extent (lon_min, lon_max, lat_min, lat_max)."""
        steps = standard_steps(self, last_step)

        def field(param, var):
            da = by_valid_time(self.load_run(run, steps, param)[var])
            return da if extent is None else crop(da, extent)

        tp = field("tp", "tp")
        rain_mm = tp * TP_TO_MM[tp.attrs["units"]]
        return standard_dataset(rain_mm, field("10u", "u10"), field("10v", "v10"),
                                source=self.name, run=run, step_hours=self.step_hours)
