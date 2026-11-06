"""Interpolating sources onto every hour, and the weights the page's means give interpolated values. No network."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from loaders.standard import period_totals, standard_dataset, to_hourly, wind_speed
from website import blend

RUN = datetime(2026, 11, 3, 0)
EXTENT = (-1.5, 0.5, 50.5, 52.5)


def forecast(name, values, step_hours=1, run=RUN, rain=None, u=None):
    """A standard forecast with wind (u, 0) and accumulated rain over a 2 x 2 grid, at each step."""
    times = pd.Timestamp(run) + pd.to_timedelta(np.arange(len(values)) * step_hours, unit="h")
    coords = {"valid_time": times, "latitude": [51.0, 52.0], "longitude": [-1.0, 0.0]}
    grid = lambda v: xr.DataArray(np.asarray(v, float)[:, None, None] * np.ones((1, 2, 2)),
                                  dims=("valid_time", "latitude", "longitude"), coords=coords)
    u10 = grid(values if u is None else u)
    rain_mm = grid(np.cumsum(values) - values[0] if rain is None else rain)
    return standard_dataset(rain_mm, u10, xr.zeros_like(u10), source=name, run=run, step_hours=step_hours)


def test_to_hourly_interpolates_wind_and_keeps_rain_totals():
    ds = forecast("6-hourly", [0, 6, 12], step_hours=6, rain=[0, 3, 9])
    hourly = to_hourly(ds)
    assert hourly.valid_time.size == 13 and hourly.attrs == ds.attrs
    np.testing.assert_allclose(hourly.u10.isel(latitude=0, longitude=0), np.arange(13))
    # 3 mm then 6 mm spread evenly over each 6 hours, so the 6-hourly totals are unchanged
    totals = period_totals(hourly, 6).isel(latitude=0, longitude=0)
    np.testing.assert_allclose(totals.sel(valid_time=ds.valid_time[1:]), [3, 6])
    np.testing.assert_allclose(totals.sel(valid_time=RUN + timedelta(hours=9)), 4.5)
    assert to_hourly(forecast("hourly", [1, 2, 3])) is not None
    xr.testing.assert_identical(to_hourly(hourly), hourly)


def test_positions_give_the_gap_and_hours_into_it():
    real = np.array(["2026-11-03T00", "2026-11-03T06", "2026-11-03T09"], dtype="datetime64[h]")
    times = np.arange(real[0], real[-1] + 1)
    gap, offset = blend.positions(real, times)
    assert gap.tolist() == [0, 6, 6, 6, 6, 6, 0, 3, 3, 0]
    assert offset.tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 0]


def test_model_variance_is_half_the_mean_squared_difference_between_sources():
    a, b = forecast("a", [5.0] * 49), forecast("b", [7.0] * 49)
    var = blend.model_variance([blend.coarse(ds, EXTENT) for ds in (a, b)], RUN, 6)
    assert var["wind"] == [pytest.approx(2.0)] * 3  # (7 - 5)² / 2 in every lead band
    assert var["rain"] == [pytest.approx(2 * 2 * 6 ** 2 / 2)] * 3  # 6-hour totals differ by 6 x 2 mm


def _wavy(name, step_hours=1, shift=0.0):
    """Wind and rain varying smoothly over 3 days, so interpolation is wrong most in mid-gap; `shift`
    makes another source that disagrees a little."""
    hours = np.arange(0, 73, step_hours)
    wind = 8 + shift + 4 * np.sin(2 * np.pi * hours / 13)
    rain = 3 * (hours - 17 / (2 * np.pi) * np.cos(2 * np.pi * hours / 17))  # smooth, rising
    return forecast(name, wind, step_hours, rain=(1 + shift / 10) * (rain - rain[0]))


def test_weights_are_1_at_real_times_and_lowest_mid_gap():
    forecasts = {"hourly": _wavy("hourly"), "other": _wavy("other", shift=0.5),
                 "6-hourly": _wavy("6-hourly", 6)}
    weights = blend.Weights(forecasts, EXTENT, 6, RUN)
    times = np.arange(np.datetime64("2026-11-03T12"), np.datetime64("2026-11-03T19"), np.timedelta64(1, "h"))
    for field in ("wind", "rain"):
        w = weights.weights("6-hourly", field, times)
        assert w[0] == w[-1] == 1 and (w[1:-1] <= blend.MAX_INTERPOLATED_WEIGHT).all()
        assert w[3] == w.min() and w[1] > w[2] > w[3]
        np.testing.assert_array_equal(weights.weights("hourly", field, times), 1)
    summary = weights.summary()
    assert set(summary["interp_variance"]["wind"]) == {f"6h+{k}" for k in range(1, 6)}
    assert len(summary["model_variance"]["wind"]) == blend.LEAD_BANDS


def test_without_hourly_sources_interpolated_values_still_count_less_than_real():
    forecasts = {"a": _wavy("a", 3), "b": _wavy("b", 6)}
    w = blend.Weights(forecasts, EXTENT, 6, RUN).weights("b", "wind", [np.datetime64("2026-11-03T06"),
                                                                       np.datetime64("2026-11-03T09")])
    assert w.tolist() == [1.0, blend.MAX_INTERPOLATED_WEIGHT]
