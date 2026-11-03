"""Loader for Environment and Climate Change Canada's GDPS (GEM global model), from the MSC Datamart.

Free, with no account or API key needed (Open Government Licence - Canada, credit Environment and
Climate Change Canada): https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/
GDPS runs at 00/12 UTC on a global 0.15° (~15 km) lat/lon grid with longitudes 0-360 (shifted to
-180-180 when loaded), with hourly steps. Data appears ~4-5 hours after the run time.

The Datamart keeps about 30 days of runs, in one folder per day; files already downloaded are kept in
the cache.

Layout: `<YYYYMMDD>/WXO-DD/model_gdps/15km/<HH>/<step:03d>/<YYYYMMDD>T<HH>Z_MSC_GDPS_<field>_LatLon0.15_PT<step:03d>H.grib2`,
one GRIB2 file per run, step and field. Fields are named "<variable>_<level>", e.g. "AirTemp_AGL-2m",
"WindU_AGL-10m", "Precip-Accum_Sfc".
"""

from datetime import timedelta

import xarray as xr

from ._per_file import PerFileGribLoader
from .standard import by_valid_time, crop, standard_dataset, standard_steps, with_zero_start

BASE_URL = "https://dd.weather.gc.ca"


class GDPSLoader(PerFileGribLoader):
    """Environment Canada GDPS. See `PerFileGribLoader` for downloading, caching and opening files."""

    name = "Canada GDPS"
    step_hours = 1
    max_step = 120
    run_hours = 12
    retention = timedelta(days=30)
    cache_name = "gdps"
    probe_field = "WindU_AGL-10m"

    def file_url(self, run, step_hours, field):
        return (f"{BASE_URL}/{run:%Y%m%d}/WXO-DD/model_gdps/15km/{run:%H}/{step_hours:03d}/"
                f"{run:%Y%m%dT%H}Z_MSC_GDPS_{field}_LatLon0.15_PT{step_hours:03d}H.grib2")

    def load_standard(self, run, last_step, extent=None):
        """A run's standard forecast (see `loaders.standard`) for steps 0 to last_step, optionally cropped
        to extent (lon_min, lon_max, lat_min, lat_max)."""
        steps = standard_steps(self, last_step)

        def field(steps, name):
            da = by_valid_time(next(iter(self.load_run(run, steps, name).data_vars.values())))
            return da if extent is None else crop(da, extent)

        u10, v10 = field(steps, "WindU_AGL-10m"), field(steps, "WindV_AGL-10m")
        # "Precip-Accum" is WMO parameter 0-1-8 (total precipitation, kg m-2 = mm) since the run started;
        # cfgrib doesn't recognise it so calls it "unknown". Step 0 has none, as nothing has accumulated yet
        rain_mm = with_zero_start(field(steps[1:], "Precip-Accum_Sfc"), u10) if last_step > 0 else xr.zeros_like(u10)
        return standard_dataset(rain_mm, u10, v10, source=self.name, run=run, step_hours=self.step_hours)
