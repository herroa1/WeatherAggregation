"""Loader for Met Office atmospheric model data, from two sources.

Archive (historical runs): a 2-year rolling archive published as a public S3 bucket, with no AWS
account or API key needed (licence CC BY-SA 4.0):
https://aws.amazon.com/marketplace/pp/prodview-oiodcatwsyjwm
Layout: `<model>/<run>/<valid time>-PT<step>H00M-<variable>.nc`, one NetCDF file per run, step
and variable. Data appears ~3-6 hours after the run time. Models are "uk-deterministic-2km" (hourly
runs) and "global-deterministic-10km" (00/06/12/18 UTC, hourly steps to 54 h then 3-hourly to 168 h; ~5 MB
per global field, so ~350 MB per day-long run of rain and wind, downloaded and then discarded by the store). `MetOfficeGlobalLoader` gives the global
model as a standard forecast source.

Live (latest runs only): the Weather DataHub Atmospheric Models API, which serves the latest
run(s) of each order set up on your account as GRIB2 files
(see https://github.com/MetOffice/weather_datahub_utilities). Needs an API key, read from
`MET_OFFICE_API_KEY` in the project-root `.env`. Downloads count towards the account's allowance.
"""

import os
import xml.etree.ElementTree as ET
from datetime import timedelta

import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import quote

import cartopy.crs as ccrs
import requests
import xarray as xr
from dotenv import load_dotenv

from ._http import download_to, exists, find_latest_run
from .standard import by_valid_time, crop, shift_longitude, standard_dataset, standard_steps, with_zero_start

BUCKET_URL = "https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com"
DATAHUB_URL = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0"
DATAHUB_DATA_SPEC = "1.1.0"


class MetOfficeLoader:
    """Downloads and caches Met Office model files, and opens them as xarray Datasets.

    Files are only downloaded if not already on disk. Archive files are cached under
    `data_dir/<model>/<run>/`, and live files under `data_dir/<order id>/<run>/`.
    `model` only applies to the archive; live data comes from whatever model an order is for.
    """

    name = "Met Office"
    step_hours = 1

    def __init__(self, data_dir="data", model="uk-deterministic-2km", api_key=None):
        self.data_dir = Path(data_dir)
        self.model = model
        self._api_key = api_key

    # --- Archive (AWS S3) -------------------------------------------------------------------

    @staticmethod
    def file_name(run, step_hours, variable):
        valid = run + timedelta(hours=step_hours)
        return f"{valid:%Y%m%dT%H%MZ}-PT{step_hours:04d}H00M-{variable}.nc"

    def list_variables(self, run, step_hours=1):
        """Variable names available in the archive for a run at a given step."""
        prefix = f"{self.model}/{run:%Y%m%dT%H%MZ}/{self.file_name(run, step_hours, '').removesuffix('.nc')}"
        r = requests.get(BUCKET_URL, params={"list-type": "2", "prefix": prefix}, timeout=60)
        r.raise_for_status()
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys = [k.text for k in ET.fromstring(r.content).findall("s3:Contents/s3:Key", ns)]
        return sorted(k[len(prefix):-len(".nc")] for k in keys)

    def download(self, run, step_hours, variable):
        """Download one archive file, skipping it if already on disk. Returns its path."""
        run_folder = f"{run:%Y%m%dT%H%MZ}"
        name = self.file_name(run, step_hours, variable)
        out = self.data_dir / self.model / run_folder / name
        if not out.exists():
            download_to(out, f"{BUCKET_URL}/{self.model}/{run_folder}/{name}", timeout=120)
        return out

    def load_run(self, run, steps, variable):
        """One archived run's forecast for a variable over several steps, stacked along "time"."""
        return self._open_stacked([self.download(run, s, variable) for s in steps])

    def load_series(self, start, end, step, variable):
        """One archived field per valid time from start to end inclusive, stacked along "time".

        Each field comes from the run issued 1 hour before it (its 1-hour step), which is the
        freshest forecast available for that hour.
        """
        if step < timedelta(hours=1) or step % timedelta(hours=1):
            raise ValueError("step must be a whole number of hours")
        times = []
        t = start
        while t <= end:
            times.append(t)
            t += step
        return self._open_stacked([self.download(t - timedelta(hours=1), 1, variable) for t in times])

    @staticmethod
    def crs(ds):
        """Cartopy projection of a loaded archive Dataset's x/y grid, from its grid-mapping attributes.

        The UK 2 km grid is Lambert azimuthal equal-area, with x/y in metres (not lat/lon).
        """
        gm = ds["lambert_azimuthal_equal_area"].attrs
        return ccrs.LambertAzimuthalEqualArea(
            central_longitude=gm["longitude_of_projection_origin"],
            central_latitude=gm["latitude_of_projection_origin"],
            globe=ccrs.Globe(semimajor_axis=gm["semi_major_axis"], semiminor_axis=gm["semi_minor_axis"]),
        )

    # --- Live (Weather DataHub API) ---------------------------------------------------------

    def list_orders(self):
        """Active DataHub orders on the account."""
        return self._datahub_get("/orders", params={"detail": "MINIMAL"}).json()["orders"]

    def list_latest_files(self, order_id):
        """Files in an order's latest run, as dicts with "fileId", "runDateTime", etc."""
        r = self._datahub_get(f"/orders/{order_id}/latest", params={"dataSpec": DATAHUB_DATA_SPEC, "detail": "MINIMAL"})
        files = r.json()["orderDetails"]["files"]
        # Each parameter is listed twice ("_+00" and "_YYYYMMDDHH" IDs); keep the "_+" ones to avoid duplicate downloads
        return [f for f in files if "_+" in f["fileId"]]

    def latest_order_run(self, order_id):
        """Run time (naive UTC datetime) of an order's latest run, from its file listing (no data downloaded)."""
        files = self.list_latest_files(order_id)
        return pd.Timestamp(files[0]["runDateTime"]).tz_convert(None).to_pydatetime()

    def download_latest(self, order_id, file):
        """Download one file from an order's latest run, skipping it if already on disk. Returns its path."""
        run = file["runDateTime"][:13].replace("-", "").replace("T", "")  # e.g. 2026101900
        out = self.data_dir / order_id / run / f"{file['fileId']}.grib2"
        if not out.exists():
            # The "+" in file IDs must be URL-encoded, otherwise the API returns 404
            url = f"{DATAHUB_URL}/orders/{order_id}/latest/{quote(file['fileId'])}/data"
            download_to(out, url, timeout=300, params={"dataSpec": DATAHUB_DATA_SPEC},
                        headers=self._datahub_headers("application/x-grib"))
        return out

    def load_latest(self, order_id, files=None):
        """Download (if needed) and open files from an order's latest run, keyed by file ID.

        `files` is a list from `list_latest_files` (e.g. filtered to the parameters you want);
        by default every file in the latest run is loaded.
        """
        if files is None:
            files = self.list_latest_files(order_id)
        paths = [self.download_latest(order_id, f) for f in files]
        return self._open_grib(paths)

    def cached_runs(self, order_id):
        """Runs of an order already downloaded, as YYYYMMDDHH strings, oldest first."""
        order_dir = self.data_dir / order_id
        return sorted(d.name for d in order_dir.iterdir() if d.is_dir()) if order_dir.exists() else []

    def load_cached(self, order_id, run):
        """Open an already-downloaded run (YYYYMMDDHH) of an order, keyed by file ID like `load_latest`."""
        return self._open_grib(sorted((self.data_dir / order_id / run).glob("*.grib2")))

    def load_standard_latest(self, order_id):
        """Download (if needed) an order's latest run, as a standard forecast (see `loaders.standard`)."""
        return self.standardise(self.load_latest(order_id))

    @classmethod
    def standardise(cls, files):
        """Standard forecast from an order's files (as returned by `load_latest`/`load_cached`).

        The order must include 1-hour rainfall accumulations and 10 m u/v wind.
        """
        def field(key_part):
            (ds,) = [ds for key, ds in files.items() if key_part in key]
            # cfgrib doesn't recognise the wind parameters, so names them "unknown": take the file's one variable
            return shift_longitude(by_valid_time(next(iter(ds.data_vars.values()))))

        u10, v10 = field("u-component-of-wind"), field("v-component-of-wind")
        hourly = field("rainfall-accumulation-1h")  # mm (kg m-2) in the hour ending at each valid time, from step 1
        rain_mm = with_zero_start(hourly.cumsum("valid_time"), u10)
        run = u10.time.values
        return standard_dataset(rain_mm, u10, v10, source=cls.name, run=run, step_hours=cls.step_hours)

    def _datahub_headers(self, accept):
        if self._api_key is None:
            load_dotenv()
            self._api_key = os.environ.get("MET_OFFICE_API_KEY")
            if not self._api_key:
                raise RuntimeError("No DataHub API key: set MET_OFFICE_API_KEY in .env or pass api_key")
        return {"apikey": self._api_key, "Accept": accept}

    def _datahub_get(self, path, params):
        r = requests.get(f"{DATAHUB_URL}{path}", params=params, headers=self._datahub_headers("application/json"),
                         timeout=60)
        r.raise_for_status()
        return r

    # --- Shared -----------------------------------------------------------------------------

    @staticmethod
    def _open_grib(paths):
        return {p.stem: xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}) for p in paths}

    @staticmethod
    def _open_stacked(paths):
        return xr.concat([xr.open_dataset(p) for p in paths], dim="time", coords="minimal", compat="override")


class MetOfficeGlobalLoader(MetOfficeLoader):
    """The Met Office global 10 km model from the public archive, as a standard forecast source.

    Uses no DataHub allowance. See `MetOfficeLoader` for downloading and caching archive files.
    """

    name = "Met Office global"
    step_hours = 1
    max_step = 54
    later_step_hours = 3  # to 168 h
    run_hours = 6
    retention = None  # 2-year rolling archive

    def __init__(self, data_dir="data"):
        super().__init__(data_dir=data_dir, model="global-deterministic-10km")

    def latest_run(self, step_hours=0, max_age=timedelta(days=2)):
        """Most recent run (naive UTC datetime) that has published the given step."""
        def is_published(run):
            name = self.file_name(run, step_hours, "wind_speed_at_10m")
            return exists(f"{BUCKET_URL}/{self.model}/{run:%Y%m%dT%H%MZ}/{name}")

        return find_latest_run(is_published, self.run_hours, max_age, f"{self.name} run with step {step_hours}h published")

    def load_standard(self, run, last_step, extent=None):
        """A run's standard forecast (see `loaders.standard`) for steps 0 to last_step, optionally cropped
        to extent (lon_min, lon_max, lat_min, lat_max)."""
        steps = standard_steps(self, last_step)

        def field(steps, variable):
            fields = []
            for s in steps:
                ds = xr.open_dataset(self.download(run, s, variable))
                # Each file holds one 2D field, plus grid-bounds variables
                (name,) = [k for k, v in ds.data_vars.items() if set(v.dims) == {"latitude", "longitude"}]
                da = ds[name] if extent is None else crop(ds[name], extent)
                fields.append(da.expand_dims("time"))
            da = xr.concat(fields, dim="time", coords="minimal", compat="override").rename(time="valid_time")
            return da.drop_vars(["forecast_period", "forecast_reference_time"], errors="ignore")

        # The archive has wind speed and the direction it blows *from* (degrees clockwise from north)
        speed, direction = field(steps, "wind_speed_at_10m"), np.radians(field(steps, "wind_direction_at_10m"))
        u10, v10 = -speed * np.sin(direction), -speed * np.cos(direction)
        if last_step > 0:
            # Total precipitation (rain and snow, like the other global models) in metres over the period
            # ending at each valid time, from step 1: the hour before to 54 h, then the 3 hours before
            parts = [field([s for s in steps if 0 < s <= self.max_step], "precipitation_accumulation-PT01H")]
            if later := [s for s in steps if s > self.max_step]:
                parts.append(field(later, "precipitation_accumulation-PT03H"))
            for part in parts:
                if part.attrs.get("units") != "m":
                    raise ValueError(f"Unexpected precipitation units {part.attrs.get('units')!r}")
            periods = xr.concat(parts, dim="valid_time", coords="minimal", compat="override")
            rain_mm = with_zero_start(periods.cumsum("valid_time") * 1000, u10)
        else:
            rain_mm = xr.zeros_like(u10)
        return standard_dataset(rain_mm, u10, v10, source=self.name, run=run, step_hours=self.step_hours)
