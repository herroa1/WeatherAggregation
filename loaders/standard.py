"""The standard forecast format every loader can produce, and helpers for comparing sources.

A standard forecast is an xarray Dataset with:
- `rain_mm`: precipitation accumulated since the run started (mm), so 0 at the run time
- `u10`, `v10`: 10 m wind components (m s-1)
- a `valid_time` dimension (one per step, from the run time), plus spatial dimensions: `latitude` and
  `longitude` for regular grids, or `y` and `x` with 2D `latitude`/`longitude` coordinates
- longitudes in -180-180
- attributes `source` (display name), `run` (ISO run time, UTC) and `step_hours` (spacing of the first
  valid times; later ones can be further apart)

Loaders do all source-specific fixing (units, longitudes, missing step-0 fields, variable names) so
that code using a standard forecast never needs to know where it came from.

Every forecast source (see `loaders.forecast_sources`) has:
- `name`: display name
- `step_hours`: spacing of its steps up to `max_step` hours; after that, steps are `later_step_hours`
  apart if the source sets it (e.g. hourly to 54 h, then 3-hourly), or not available otherwise
- `run_hours`: spacing of its runs (e.g. 6 for 00/06/12/18 UTC)
- `retention`: roughly how long its server keeps a run (a timedelta), or None if kept for years
- `latest_run(step_hours)`: most recent run with that step published
- `load_standard(run, last_step, extent=None)`: a run's standard forecast for steps 0 to last_step
"""

import numpy as np
import pandas as pd
import xarray as xr

UNITS = {"rain_mm": "mm", "u10": "m s-1", "v10": "m s-1"}


def by_valid_time(da):
    """Index a cfgrib DataArray by "valid_time" instead of "step"."""
    return da.swap_dims(step="valid_time").drop_vars("step")


def shift_longitude(da):
    """Longitudes from 0-360 to -180-180, re-sorted if they are a 1D dimension."""
    da = da.assign_coords(longitude=(da.longitude + 180) % 360 - 180)
    return da.sortby("longitude") if "longitude" in da.dims else da


def crop(da, extent):
    """Crop a regular lat/lon DataArray to (lon_min, lon_max, lat_min, lat_max), whichever way latitude runs."""
    lon_min, lon_max, lat_min, lat_max = extent
    lat = slice(lat_max, lat_min) if da.latitude[0] > da.latitude[-1] else slice(lat_min, lat_max)
    return da.sel(longitude=slice(lon_min, lon_max), latitude=lat)


def standard_steps(source, last_step):
    """A source's steps from 0 to last_step: every `step_hours` up to `max_step`, then every
    `later_step_hours` (for sources that have it). last_step must be one of them."""
    later = getattr(source, "later_step_hours", None)
    early = list(range(0, min(last_step, source.max_step) + 1, source.step_hours))
    if 0 <= last_step <= source.max_step and last_step % source.step_hours == 0:
        return early
    if last_step > source.max_step and later and (last_step - source.max_step) % later == 0:
        return early + list(range(source.max_step + later, last_step + 1, later))
    spacing = f", then {later}-hourly" if later else " (and not beyond)"
    raise ValueError(f"{source.name} has no {last_step} h step: steps are only {source.step_hours}-hourly "
                     f"up to {source.max_step} h{spacing}")


def last_step_by(source, hours):
    """A source's last step at or before `hours` (see `standard_steps`); its last evenly spaced step if it
    has no later steps."""
    hours = max(hours, 0)
    later = getattr(source, "later_step_hours", None)
    if hours <= source.max_step:
        return hours // source.step_hours * source.step_hours
    if not later:
        return source.max_step
    return source.max_step + (hours - source.max_step) // later * later


def with_zero_start(rain_mm, like):
    """Prepend zero rain at the run time (the first valid time of `like`), for sources with no rain field
    at step 0 because nothing has accumulated yet."""
    zero = xr.zeros_like(like.isel(valid_time=[0]))
    return xr.concat([zero, rain_mm], dim="valid_time", coords="minimal", compat="override")


def standard_dataset(rain_mm, u10, v10, *, source, run, step_hours):
    """Build a standard forecast from DataArrays already indexed by valid_time and in standard units."""
    fields = {"rain_mm": rain_mm, "u10": u10, "v10": v10}
    keep = {"valid_time", "latitude", "longitude"}
    fields = {name: da.drop_vars([c for c in da.coords if c not in keep]).rename(name)
              .assign_attrs(units=UNITS[name]) for name, da in fields.items()}
    ds = xr.merge(fields.values(), join="exact", compat="equals", combine_attrs="drop")
    for name, da in fields.items():
        ds[name].attrs = {"units": UNITS[name]}
    return ds.assign_attrs(source=source, run=pd.Timestamp(run).isoformat(), step_hours=step_hours)


def run_time(ds):
    return pd.Timestamp(ds.attrs["run"])


def period_totals(ds, hours):
    """Rain (mm) over the `hours` before each valid time, from a standard forecast: at every valid time
    that has another valid time `hours` before it (so this works when steps are further apart later on)."""
    rain = ds["rain_mm"]
    earlier = rain.assign_coords(valid_time=rain.valid_time + np.timedelta64(hours, "h"))
    totals = rain - earlier  # aligns on valid_time, keeping only the times both have
    if totals.valid_time.size == 0:
        raise ValueError(f"{ds.attrs['source']} has no valid times {hours} h apart to total rain over")
    # GRIB and storage rounding can make a dry period's total come out a hair below zero
    return totals.clip(min=0)


def wind_speed(ds):
    return np.hypot(ds["u10"], ds["v10"]).rename("wind_speed").assign_attrs(units="m s-1")


def regrid_to_box(da, extent, res=0.5):
    """Average a field onto a regular `res`° grid covering extent, so different grids can be compared.

    Works for 1D and 2D lat/lon; cells with no source points are NaN. Keeps any "valid_time" dimension.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.arange(lon_min + res / 2, lon_max, res)
    lats = np.arange(lat_min + res / 2, lat_max, res)
    lat, lon = xr.broadcast(da.latitude, da.longitude)
    lat, lon = lat.values.ravel(), lon.values.ravel()
    iy = np.floor((lat - lat_min) / res).astype(int)
    ix = np.floor((lon - lon_min) / res).astype(int)
    inside = (iy >= 0) & (iy < len(lats)) & (ix >= 0) & (ix < len(lons))
    cell = iy[inside] * len(lons) + ix[inside]

    spatial = [d for d in da.dims if d != "valid_time"]
    values = da.transpose(..., *spatial).values.reshape(-1, lat.size)  # (time, points)
    out = np.full((values.shape[0], len(lats) * len(lons)), np.nan)
    for i, row in enumerate(values):
        v = row[inside]
        ok = np.isfinite(v)
        counts = np.bincount(cell[ok], minlength=out.shape[1])
        sums = np.bincount(cell[ok], weights=v[ok], minlength=out.shape[1])
        with np.errstate(invalid="ignore"):
            out[i] = sums / counts
    out = out.reshape(-1, len(lats), len(lons))
    coords = {"latitude": lats, "longitude": lons}
    if "valid_time" in da.dims:
        return xr.DataArray(out, dims=("valid_time", "latitude", "longitude"),
                            coords={"valid_time": da.valid_time.values, **coords}, name=da.name)
    return xr.DataArray(out[0], dims=("latitude", "longitude"), coords=coords, name=da.name)
