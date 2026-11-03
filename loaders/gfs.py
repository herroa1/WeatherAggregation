"""Loader for NOAA's global models from the NOAA Open Data buckets on AWS: GFS and the AI model AIGFS.

Free, with no AWS account or API key needed (public domain, credit NOAA). Both run at 00/06/12/18 UTC
on a global 0.25° lat/lon grid with longitudes 0-360 (shifted to -180-180 when loaded).

- GFS (Global Forecast System, physics-based): https://registry.opendata.aws/noaa-gfs-bdp-pds/
  Hourly steps to 120 h then 3-hourly to 384 h; data appears ~3.5-5 hours after the run time, and
  runs are kept from 2021 onwards.
  Layout: `gfs.<YYYYMMDD>/<HH>/atmos/gfs.t<HH>z.pgrb2.0p25.f<step:03d>`
- AIGFS (NOAA's operational AI model, which replaced the experimental GraphCastGFS feed in May 2026):
  6-hourly steps to 384 h, runs kept from May 2026 onwards. Surface fields are in the "sfc" files.
  Layout: `aigfs.<YYYYMMDD>/<HH>/model/atmos/grib2/aigfs.t<HH>z.sfc.f<step:03d>.grib2`

Each GRIB2 file holds every field for one run and step, with a `.idx` file next to it (one
`num:offset:d=<run>:<variable>:<level>:<forecast>:` line per field), so single fields are downloaded
with HTTP range requests. Fields are identified by variable and level as written there, e.g.
("TMP", "2 m above ground"), ("UGRD", "10 m above ground"), ("APCP", "surface")
(see https://www.nco.ncep.noaa.gov/pmb/products/gfs/).
"""

from datetime import timedelta
from pathlib import Path

import xarray as xr

from ._http import download_to, exists, find_latest_run, session
from .standard import by_valid_time, crop, shift_longitude, standard_dataset, standard_steps, with_zero_start

MODELS = {
    "gfs": {
        "name": "NOAA GFS", "step_hours": 1, "max_step": 120,
        "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/atmos/"
               "gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}",
    },
    "aigfs": {
        "name": "NOAA AIGFS", "step_hours": 6, "max_step": 384,
        "url": "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/aigfs.{run:%Y%m%d}/{run:%H}/model/atmos/grib2/"
               "aigfs.t{run:%H}z.sfc.f{step:03d}.grib2",
    },
}


class GFSLoader:
    """Downloads and caches single-field GRIB2 files from GFS or AIGFS, and opens them as xarray Datasets.

    Files are only downloaded if not already on disk, and are cached under
    `data_dir/<model>/<run>/<step>h-<variable>-<level>.grib2` (run as YYYYMMDDHH).
    `model` is "gfs" (physics-based) or "aigfs" (AI).
    """

    run_hours = 6
    retention = None

    def __init__(self, data_dir="data", model="gfs"):
        self.data_dir = Path(data_dir)
        self.model = model
        config = MODELS[model]
        self.name, self.step_hours, self.max_step = config["name"], config["step_hours"], config["max_step"]
        self._url = config["url"]

    def file_url(self, run, step_hours):
        """URL of a run and step's GRIB2 file (add ".idx" for its index)."""
        return self._url.format(run=run, step=step_hours)

    def latest_run(self, step_hours=0, max_age=timedelta(days=2)):
        """Most recent run (naive UTC datetime) that has published the given step.

        Pass the last step you need, since steps are published one after another and a run can be
        partly available.
        """
        return find_latest_run(lambda run: exists(self.file_url(run, step_hours) + ".idx"), self.run_hours, max_age,
                               f"{self.name} run with step {step_hours}h published")

    def list_fields(self, run, step_hours=0):
        """Index entries for a run and step, as dicts with "variable", "level", "forecast", "offset" and
        "length" (None for the last field, which runs to the end of the file)."""
        r = session.get(self.file_url(run, step_hours) + ".idx", timeout=60)
        r.raise_for_status()
        rows = [line.split(":") for line in r.text.splitlines() if line]
        offsets = [int(row[1]) for row in rows] + [None]
        return [{"variable": row[3], "level": row[4], "forecast": row[5], "offset": start,
                 "length": None if end is None else end - start}
                for row, start, end in zip(rows, offsets, offsets[1:])]

    def download(self, run, step_hours, variable, level):
        """Download one field for a run and step, skipping it if already on disk. Returns its path.

        Accumulated and averaged fields (e.g. "APCP") are listed twice, once over the whole run so far
        and once over a period that resets every 6 hours; this takes the one over the whole run.
        """
        name = f"{step_hours}h-{variable}-{level.replace(' ', '_')}.grib2"
        out = self.data_dir / self.model / f"{run:%Y%m%d%H}" / name
        if not out.exists():
            fields = [f for f in self.list_fields(run, step_hours) if (f["variable"], f["level"]) == (variable, level)]
            if len(fields) > 1:
                fields = [f for f in fields if f["forecast"].startswith("0-")]
            if not fields:
                raise ValueError(f"No {variable!r} at {level!r} in step {step_hours}h")
            # When the run-so-far period is also the 0-6 h bucket (steps 1-6) both entries are identical
            field = fields[0]
            end = "" if field["length"] is None else field["offset"] + field["length"] - 1
            download_to(out, self.file_url(run, step_hours), timeout=120,
                        headers={"Range": f"bytes={field['offset']}-{end}"})
        return out

    def load_run(self, run, steps, variable, level):
        """One run's forecast for a field over several steps, stacked along "step".

        Longitudes are shifted from 0-360 to -180-180 (as for ECMWF), so regions crossing the
        Greenwich meridian, like the UK, can be selected with a single slice.
        """
        paths = [self.download(run, s, variable, level) for s in steps]
        datasets = [xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}) for p in paths]
        ds = xr.concat(datasets, dim="step", coords="different", compat="equals", join="exact")
        return shift_longitude(ds)

    def load_standard(self, run, last_step, extent=None):
        """A run's standard forecast (see `loaders.standard`) for steps 0 to last_step, optionally cropped
        to extent (lon_min, lon_max, lat_min, lat_max)."""
        steps = standard_steps(self, last_step)

        def field(steps, variable, level):
            ds = self.load_run(run, steps, variable, level)
            da = by_valid_time(next(iter(ds.data_vars.values())))
            return da if extent is None else crop(da, extent)

        u10 = field(steps, "UGRD", "10 m above ground")
        v10 = field(steps, "VGRD", "10 m above ground")
        # "APCP" is kg m-2 (= mm) since the run started. Step 0 has none, as nothing has accumulated yet
        rain_mm = with_zero_start(field(steps[1:], "APCP", "surface"), u10) if last_step > 0 else xr.zeros_like(u10)
        return standard_dataset(rain_mm, u10, v10, source=self.name, run=run, step_hours=self.step_hours)
