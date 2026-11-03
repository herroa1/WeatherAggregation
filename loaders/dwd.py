"""Loader for DWD (German weather service) ICON-EU forecasts, from DWD's open data server.

Free, with no account or API key needed (licence CC BY 4.0, credit Deutscher Wetterdienst):
https://opendata.dwd.de/weather/nwp/icon-eu/grib/
ICON-EU is DWD's European model, on a 0.0625° (~7 km) regular lat/lon grid covering 23.5°W-62.5°E,
29.5°N-70.5°N. Runs are every 3 hours: 00/06/12/18 UTC go to 120 h (hourly to 78 h, then 3-hourly),
03/09/15/21 UTC to 48 h. Data appears ~2-4 hours after the run time.

**Only the latest run of each run time is kept**, so runs disappear after about a day; files already
downloaded are kept in the cache.

Layout: `<HH>/<field, lower case>/icon-eu_europe_regular-lat-lon_single-level_<YYYYMMDDHH>_<step:03d>_<FIELD>.grib2.bz2`,
one bzip2-compressed GRIB2 file per run, step and field. Fields use DWD's names, e.g. "T_2M",
"U_10M", "V_10M", "TOT_PREC" (see https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf).
"""

from datetime import timedelta

from ._per_file import PerFileGribLoader
from .standard import by_valid_time, crop, standard_dataset, standard_steps

BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"


class ICONEULoader(PerFileGribLoader):
    """DWD ICON-EU. See `PerFileGribLoader` for downloading, caching and opening files."""

    name = "DWD ICON-EU"
    step_hours = 1
    max_step = 78
    run_hours = 3
    retention = timedelta(hours=24)
    cache_name = "icon-eu"
    probe_field = "U_10M"

    def file_url(self, run, step_hours, field):
        return (f"{BASE_URL}/{run:%H}/{field.lower()}/icon-eu_europe_regular-lat-lon_single-level_"
                f"{run:%Y%m%d%H}_{step_hours:03d}_{field}.grib2.bz2")

    def load_standard(self, run, last_step, extent=None):
        """A run's standard forecast (see `loaders.standard`) for steps 0 to last_step, optionally cropped
        to extent (lon_min, lon_max, lat_min, lat_max)."""
        steps = standard_steps(self, last_step)

        def field(name):
            da = by_valid_time(next(iter(self.load_run(run, steps, name).data_vars.values())))
            return da if extent is None else crop(da, extent)

        # "TOT_PREC" is kg m-2 (= mm) since the run started, including a zero field at step 0
        return standard_dataset(field("TOT_PREC"), field("U_10M"), field("V_10M"),
                                source=self.name, run=run, step_hours=self.step_hours)
