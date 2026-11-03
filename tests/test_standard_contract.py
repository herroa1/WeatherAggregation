"""Every source's real data meets the standard format (see `loaders/standard.py`) and looks physically sane.

Runs once per source, benchmark included. A new source passes these once its loader's `load_standard`
does all the source-specific fixing.
"""

from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from conftest import CORE_UK
from loaders.standard import UNITS, period_totals, run_time, wind_speed

# GRIB packing rounds values to steps set by the whole (global) field's range, e.g. 0.06-0.12 mm for
# IFS rain over the first day, so accumulations can dip slightly between steps anywhere: by up to this
# much (mm) plus RAIN_REL_TOLERANCE of the value. A resetting period total drops by far more where it rains.
RAIN_TOLERANCE = 0.25
RAIN_REL_TOLERANCE = 0.005


def test_has_standard_variables_and_units(forecast):
    assert {name: forecast[name].attrs.get("units") for name in forecast.data_vars} == UNITS


def test_has_standard_attributes(forecast):
    assert set(forecast.attrs) == {"source", "run", "step_hours"}
    assert isinstance(forecast.attrs["step_hours"], int) and forecast.attrs["step_hours"] > 0


def test_dimensions(forecast):
    spatial = {"latitude", "longitude"} if forecast.latitude.ndim == 1 else {"y", "x"}
    assert set(forecast.dims) == {"valid_time"} | spatial
    for name in forecast.data_vars:
        assert set(forecast[name].dims) == {"valid_time"} | spatial
    if forecast.latitude.ndim == 2:
        assert set(forecast.latitude.dims) == set(forecast.longitude.dims) == {"y", "x"}


def test_valid_times_start_at_run_and_follow_the_step_spacing(forecast):
    times = forecast.valid_time.values
    assert times[0] == run_time(forecast).to_datetime64()
    assert len(times) > 1
    gaps = np.diff(times) / np.timedelta64(timedelta(hours=1))
    # Every step_hours to begin with; some sources then switch to a single wider spacing
    assert gaps[0] == forecast.attrs["step_hours"]
    assert (np.diff(gaps) >= 0).all() and len(set(gaps)) <= 2, f"irregular steps: {sorted(set(gaps))}"


def test_coordinates_in_range(forecast):
    lon, lat = forecast.longitude.values, forecast.latitude.values
    assert ((lon >= -180) & (lon < 180)).all(), "longitudes must be -180-180"
    assert ((lat >= -90) & (lat <= 90)).all()


def test_covers_mainland_uk(forecast):
    lon_min, lon_max, lat_min, lat_max = CORE_UK
    lat, lon = (a.values.ravel() for a in xr.broadcast(forecast.latitude, forecast.longitude))
    for corner_lon, corner_lat in [(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_min), (lon_max, lat_max)]:
        nearest = np.min(np.hypot(lat - corner_lat, lon - corner_lon))
        assert nearest < 0.5, f"no grid point within 0.5° of ({corner_lon}, {corner_lat})"


def test_rain_starts_at_zero_and_never_decreases(forecast):
    rain = forecast.rain_mm
    np.testing.assert_allclose(rain.isel(valid_time=0), 0, atol=RAIN_TOLERANCE)
    # A real decrease means a period total (e.g. a resetting bucket) was taken instead of the total since the run started
    drop = -rain.diff("valid_time")
    allowed = RAIN_TOLERANCE + RAIN_REL_TOLERANCE * rain.isel(valid_time=slice(None, -1)).values
    assert (drop.values <= allowed).all(), f"rain total decreases by up to {float(drop.max()):.2f} mm"


def test_values_are_plausible(forecast):
    assert float(forecast.rain_mm.min()) >= -RAIN_TOLERANCE
    totals = period_totals(forecast, hours=6)
    assert float(totals.max()) < 300, "over 300 mm in 6 hours: check the rain units"
    speed = wind_speed(forecast)
    assert float(speed.max()) < 75, "over 75 m/s: check the wind units"
    assert float(speed.mean()) > 0.5, "near-zero wind everywhere: check the wind units"


@pytest.mark.parametrize("name", ["u10", "v10", "rain_mm"])
def test_fields_have_data(forecast, name):
    da = forecast[name]
    assert float(np.isfinite(da).mean()) > 0.99, "more than 1% of values missing"
    if name != "rain_mm":  # rain can genuinely be zero everywhere early in a run
        spread = da.std([d for d in da.dims if d != "valid_time"])
        assert float(spread.min()) > 0, "a field is constant across the grid"
