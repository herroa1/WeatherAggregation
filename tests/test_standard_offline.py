"""Standard-format conversion and comparison helpers, on small synthetic data (no network or files).

Each loader's source-specific fixes are checked here: unit conversion, longitude shifting, the missing
step-0 rain field, and turning hourly accumulations into totals since the run started.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from loaders import ECMWFLoader, GDPSLoader, GFSLoader, ICONEULoader, MetOfficeGlobalLoader, MetOfficeLoader
from loaders.standard import crop, period_totals, regrid_to_box, shift_longitude, standard_dataset

RUN = datetime(2026, 10, 22, 0)


def cfgrib_like(values, steps, name, units, lat=None, lon=None, attrs=None):
    """A Dataset shaped like cfgrib's output: dims (step, latitude, longitude) or (step, y, x) with
    2D lat/lon, plus "time" and "valid_time" coordinates. `values` is (step, ny, nx)."""
    values = np.asarray(values, dtype="float32")
    _, ny, nx = values.shape
    lat = np.linspace(60, 50, ny) if lat is None else lat
    lon = np.linspace(-10, 2, nx) if lon is None else lon
    step = pd.to_timedelta(steps, unit="h")
    coords = {"step": step, "time": np.datetime64(RUN), "valid_time": ("step", np.datetime64(RUN) + step)}
    if np.ndim(lat) == 2:
        dims = ("step", "y", "x")
        coords |= {"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)}
    else:
        dims = ("step", "latitude", "longitude")
        coords |= {"latitude": lat, "longitude": lon}
    da = xr.DataArray(values, dims=dims, coords=coords, attrs={"units": units, **(attrs or {})})
    return da.to_dataset(name=name)


def fields(n_steps, value, ny=3, nx=4):
    return np.full((n_steps, ny, nx), value)


# --- Per-source conversion -------------------------------------------------------------------

@pytest.mark.parametrize("model, units, stored", [
    ("ifs", "m", 0.002),          # IFS: metres
    ("aifs-single", "kg m**-2", 2.0),  # AIFS: kg m-2, i.e. mm
])
def test_ecmwf_rain_is_converted_to_mm(monkeypatch, model, units, stored):
    loader = ECMWFLoader(model=model)
    steps = [0, loader.step_hours]
    synthetic = {
        "tp": cfgrib_like([fields(1, 0)[0], fields(1, stored)[0]], steps, "tp", units),
        "10u": cfgrib_like(fields(2, 3), steps, "u10", "m s**-1"),
        "10v": cfgrib_like(fields(2, 4), steps, "v10", "m s**-1"),
    }
    monkeypatch.setattr(loader, "load_run", lambda run, steps, param: synthetic[param])
    ds = loader.load_standard(RUN, loader.step_hours)
    np.testing.assert_allclose(ds.rain_mm.isel(valid_time=-1), 2.0)


@pytest.mark.parametrize("model", ["gfs", "aigfs"])
def test_gfs_rain_is_zero_at_step_0(monkeypatch, model):
    loader = GFSLoader(model=model)

    def load_run(run, steps, variable, level):
        if variable == "APCP":
            assert 0 not in steps  # GFS and AIGFS have no APCP at step 0, so it mustn't be requested
            return cfgrib_like(fields(len(steps), 1.5), steps, "tp", "kg m**-2")
        return cfgrib_like(fields(len(steps), 5), steps, variable.lower(), "m s**-1")

    monkeypatch.setattr(loader, "load_run", load_run)
    ds = loader.load_standard(RUN, 2 * loader.step_hours)
    np.testing.assert_allclose(ds.rain_mm.max(["latitude", "longitude"]), [0, 1.5, 1.5])
    assert ds.valid_time.size == 3


def test_gdps_rain_is_zero_at_step_0(monkeypatch):
    loader = GDPSLoader()

    def load_run(run, steps, field):
        if field == "Precip-Accum_Sfc":
            assert 0 not in steps  # GDPS has no precipitation at step 0, so it mustn't be requested
            return cfgrib_like(fields(len(steps), 1.5), steps, "unknown", "unknown")
        return cfgrib_like(fields(len(steps), 5), steps, field, "m s**-1")

    monkeypatch.setattr(loader, "load_run", load_run)
    ds = loader.load_standard(RUN, 2)
    np.testing.assert_allclose(ds.rain_mm.max(["latitude", "longitude"]), [0, 1.5, 1.5])


def test_icon_eu_rain_is_used_as_is(monkeypatch):
    loader = ICONEULoader()
    rain = [fields(1, v)[0] for v in (0, 1, 3)]  # ICON-EU has a zero rain field at step 0
    synthetic = {"TOT_PREC": cfgrib_like(rain, [0, 1, 2], "tp", "kg m**-2"),
                 "U_10M": cfgrib_like(fields(3, 5), [0, 1, 2], "u10", "m s**-1"),
                 "V_10M": cfgrib_like(fields(3, 5), [0, 1, 2], "v10", "m s**-1")}
    monkeypatch.setattr(loader, "load_run", lambda run, steps, field: synthetic[field])
    ds = loader.load_standard(RUN, 2)
    np.testing.assert_allclose(ds.rain_mm.max(["latitude", "longitude"]), [0, 1, 3])


def test_met_office_global_converts_wind_and_rain(monkeypatch, tmp_path):
    """The archive gives wind speed and the direction it's *from*, and precipitation in metres over the
    hour before each step up to max_step, then over the 3 hours before."""
    loader = MetOfficeGlobalLoader(data_dir=tmp_path)
    monkeypatch.setattr(loader, "max_step", 1)  # so steps are 0, 1, then 3-hourly: 4
    values = {"wind_speed_at_10m": ("wind_speed", "m s-1", 10.0),
              # From the west (270°) at step 0, from the north (0°) at step 1, from the south at step 4
              "wind_direction_at_10m": ("wind_from_direction", "degrees", {0: 270.0, 1: 0.0, 4: 180.0}),
              "precipitation_accumulation-PT01H": ("lwe_thickness_of_precipitation_amount", "m", 0.002),
              "precipitation_accumulation-PT03H": ("lwe_thickness_of_precipitation_amount", "m", 0.006)}

    def download(run, step, variable):
        name, units, value = values[variable]
        value = value[step] if isinstance(value, dict) else value
        da = xr.DataArray(np.full((3, 4), value), dims=("latitude", "longitude"),
                          coords={"latitude": [50.0, 51.0, 52.0], "longitude": [-2.0, -1.0, 0.0, 1.0],
                                  "time": np.datetime64(RUN) + np.timedelta64(step, "h")},
                          attrs={"units": units})
        path = tmp_path / f"{step}-{variable}.nc"
        da.to_dataset(name=name).to_netcdf(path)
        return path

    monkeypatch.setattr(loader, "download", download)
    ds = loader.load_standard(RUN, 4)
    point = ds.isel(latitude=0, longitude=0)
    np.testing.assert_allclose(point.u10, [10, 0, 0], atol=1e-9)   # westerly blows towards the east (+u)
    np.testing.assert_allclose(point.v10, [0, -10, 10], atol=1e-9)  # northerly towards the south (-v)
    np.testing.assert_allclose(point.rain_mm, [0, 2, 8])  # 2 mm in the first hour, then 6 mm over 3 hours


def test_met_office_hourly_rain_becomes_total_since_run_start():
    lat2d, lon2d = np.meshgrid(np.linspace(50, 60, 4), np.linspace(350, 362, 5), indexing="ij")
    grid = {"lat": lat2d, "lon": lon2d}
    files = {
        # Real file IDs; rain has steps 1-3 (hour ending), wind 0-3. Wind's variable is named "unknown"
        "ground_rainfall-accumulation-1h_+00": cfgrib_like(fields(3, 2, 4, 5), [1, 2, 3], "tirf", "kg m**-2", **grid),
        "agl_u-component-of-wind-surface-adjusted_10_+00": cfgrib_like(fields(4, 1, 4, 5), [0, 1, 2, 3], "unknown", "m s**-1", **grid),
        "agl_v-component-of-wind-surface-adjusted_10_+00": cfgrib_like(fields(4, 1, 4, 5), [0, 1, 2, 3], "unknown", "m s**-1", **grid),
    }
    ds = MetOfficeLoader.standardise(files)
    np.testing.assert_allclose(ds.rain_mm.max(["y", "x"]), [0, 2, 4, 6])
    assert float(ds.longitude.max()) == pytest.approx(2)  # 362° shifted to 2°
    assert ds.attrs == {"source": "Met Office", "run": "2026-10-22T00:00:00", "step_hours": 1}


# --- Standard dataset ------------------------------------------------------------------------

def _standard(rain_values, step_hours=3):
    steps = list(range(0, step_hours * len(rain_values), step_hours))
    da = lambda v, name: cfgrib_like(v, steps, name, "x")[name].swap_dims(step="valid_time").drop_vars("step")
    rain = da([np.full((3, 4), v) for v in rain_values], "rain")
    wind = da(fields(len(steps), 1), "wind")
    return standard_dataset(rain, wind, wind, source="Test", run=RUN, step_hours=step_hours)


def test_standard_dataset_has_only_standard_variables_coords_and_units():
    ds = _standard([0, 1, 2])
    assert set(ds.data_vars) == {"rain_mm", "u10", "v10"}
    assert set(ds.coords) == {"valid_time", "latitude", "longitude"}
    assert {name: ds[name].attrs["units"] for name in ds.data_vars} == {"rain_mm": "mm", "u10": "m s-1", "v10": "m s-1"}


def test_standard_dataset_rejects_misaligned_valid_times():
    ds = _standard([0, 1, 2])
    with pytest.raises(ValueError):
        standard_dataset(ds.rain_mm, ds.u10.isel(valid_time=slice(1, None)), ds.v10, source="T", run=RUN, step_hours=3)


def test_period_totals_differences_accumulations():
    totals = period_totals(_standard([0, 1, 3, 6, 10]), hours=6)  # 3-hourly
    np.testing.assert_allclose(totals.max(["latitude", "longitude"]), [3, 5, 7])
    assert pd.Timestamp(totals.valid_time.values[0]) == pd.Timestamp(RUN) + pd.Timedelta(hours=6)


def test_period_totals_are_never_negative():
    # A dry spell where rounding makes the accumulation dip slightly
    totals = period_totals(_standard([0, 1.0, 0.95, 0.95]), hours=3)
    assert float(totals.min()) == 0


def test_period_totals_rejects_periods_that_are_not_whole_steps():
    with pytest.raises(ValueError, match="no valid times 3 h apart"):
        period_totals(_standard([0, 1, 2], step_hours=6), hours=3)


def test_period_totals_work_where_steps_get_further_apart():
    # Hourly to 3 h, then 3-hourly: 6 h totals exist where there's a valid time 6 h earlier
    ds = _standard([0, 1, 2, 3, 4, 5, 6], step_hours=1)
    ds = ds.isel(valid_time=[0, 1, 2, 3, 6])  # valid times 0, 1, 2, 3, 6 h
    totals = period_totals(ds, hours=3)
    assert list(pd.to_datetime(totals.valid_time.values) - pd.Timestamp(RUN)) == [pd.Timedelta(hours=h) for h in (3, 6)]
    np.testing.assert_allclose(totals.max(["latitude", "longitude"]), [3, 3])


# --- Grid helpers ----------------------------------------------------------------------------

def test_shift_longitude_reorders_1d_grids():
    da = xr.DataArray(np.arange(4), dims="longitude", coords={"longitude": [0, 90, 180, 270]})
    shifted = shift_longitude(da)
    assert list(shifted.longitude.values) == [-180, -90, 0, 90]
    assert list(shifted.values) == [2, 3, 0, 1]


@pytest.mark.parametrize("lat", [np.arange(48, 63, 0.5), np.arange(62, 47, -0.5)], ids=["ascending", "descending"])
def test_crop_works_whichever_way_latitude_runs(lat):
    da = xr.DataArray(np.zeros((lat.size, 40)), dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": np.arange(-15, 5, 0.5)})
    out = crop(da, (-13, 4, 49.5, 61))
    assert (float(out.latitude.min()), float(out.latitude.max())) == (49.5, 61)
    assert (float(out.longitude.min()), float(out.longitude.max())) == (-13, 4)


def test_regrid_to_box_averages_points_in_each_cell():
    # 0.25° grid of value = longitude, averaged onto 1° cells: each cell's mean is its centre longitude
    lon = np.arange(-1.875, 2, 0.25)
    lat = np.arange(50.125, 52, 0.25)
    da = xr.DataArray(np.broadcast_to(lon, (lat.size, lon.size)), dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon})
    out = regrid_to_box(da, (-2, 2, 50, 52), res=1)
    np.testing.assert_allclose(out.isel(latitude=0), [-1.5, -0.5, 0.5, 1.5])
    assert out.shape == (2, 4)


def test_regrid_to_box_handles_2d_grids_and_empty_cells():
    lat2d, lon2d = np.meshgrid([50.2, 50.7], [0.2, 0.7], indexing="ij")
    da = xr.DataArray(np.ones((1, 2, 2)), dims=("valid_time", "y", "x"),
                      coords={"valid_time": [np.datetime64(RUN)], "latitude": (("y", "x"), lat2d),
                              "longitude": (("y", "x"), lon2d)})
    out = regrid_to_box(da, (0, 2, 50, 51), res=1)
    np.testing.assert_array_equal(out.isel(valid_time=0, latitude=0), [1, np.nan])
