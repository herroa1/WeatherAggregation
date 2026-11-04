"""Builds the password-protected forecast website: a static page plus an encrypted data file.

The page (`website/index.html`) is public and holds no forecast data. Everything it shows comes
from `forecast.bin`, the latest set of forecasts (see `loaders/store.py`) summarised as JSON, gzipped
and encrypted with AES-GCM under a key derived from the site password (PBKDF2-SHA256). The page asks
for the password and decrypts the data in the browser, so the hosted files are useless without it.
Anyone can download `forecast.bin` and try passwords offline, so use a long passphrase.

Run `python -m website.build [--data-dir DIR] [--out DIR] [--no-refresh]`, with the password in
`SITE_PASSWORD` (environment or the project-root `.env`). The output folder holds `index.html` and
`forecast.bin`, ready to host.
"""

import argparse
import gzip
import hashlib
import json
import os
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

from loaders import BENCHMARK, forecast_sources
from loaders.standard import period_totals, regrid_to_box, run_time, wind_speed
from loaders.store import UK_EXTENT, ForecastStore

HERE = Path(__file__).parent
HORIZON_HOURS = 72
RAIN_HOURS = 6
# UK maps are averaged onto this grid (degrees), every MAP_EVERY hours, to keep the data file small
MAP_RES = 0.5
MAP_EVERY = 3
# London maps show each source at its own resolution, sampled (nearest grid point) onto this grid:
# 0.02° lat x 0.03° lon is ~2.2 x 2.1 km, about the finest sources' spacing
LONDON_EXTENT = (-0.55, 0.35, 51.25, 51.72)
LONDON_RES = (0.02, 0.03)  # (lat, lon)
# Mainland UK, which lies inside every source's grid: used for the scores
SCORE_EXTENT = (-8, 2, 50, 59)
LOCATIONS = {
    "Central London": (51.51, -0.13), "North London (Enfield)": (51.65, -0.08),
    "East London (Romford)": (51.58, 0.18), "South London (Croydon)": (51.37, -0.10),
    "West London (Heathrow)": (51.47, -0.45),
    "Birmingham": (52.48, -1.90), "Manchester": (53.48, -2.24), "Cardiff": (51.48, -3.18),
    "Edinburgh": (55.95, -3.19), "Belfast": (54.60, -5.93),
}
# Must match index.html. The salt is fixed so a key remembered on a phone works for later builds
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT = b"uk-forecast-comparison"
MIN_PASSWORD_LENGTH = 12


def grid_km(da):
    """A field's grid spacing (km, north-south), for 1D or 2D lat/lon grids."""
    if da.latitude.ndim == 1:
        return float(abs(np.diff(da.latitude.values)).mean()) * 111.2
    lat, lon = da.latitude.values, da.longitude.values
    y, x = lat.shape[0] // 2, lat.shape[1] // 2
    return float(np.hypot(lat[y + 1, x] - lat[y, x], (lon[y + 1, x] - lon[y, x]) * np.cos(np.radians(lat[y, x])))) * 111.2


def sample_nearest(da, lats, lons):
    """A field at each (lat, lon) of a regular grid, from the source's nearest grid point: dims
    (valid_time, latitude, longitude). NaN where the source has no grid point within one grid spacing,
    so a point outside its area (e.g. a regional source's) isn't given the value at its edge."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    spacing = grid_km(da) / 111.2  # degrees of latitude
    if da.latitude.ndim == 1:
        out = da.sel(latitude=lats, longitude=lons, method="nearest")
        near_lat = np.abs(out.latitude.values - lats) <= spacing
        near_lon = np.abs(out.longitude.values - lons) * np.cos(np.radians(lats.mean())) <= spacing
        out = out.assign_coords(latitude=lats, longitude=lons)
        return out.where(xr.DataArray(near_lat[:, None] & near_lon[None, :], dims=("latitude", "longitude")))
    # 2D lat/lon (the Met Office grid): brute-force nearest among points near the target grid
    near = ((da.latitude >= lats.min() - 3 * spacing) & (da.latitude <= lats.max() + 3 * spacing)
            & (da.longitude >= lons.min() - 5 * spacing) & (da.longitude <= lons.max() + 5 * spacing)).values
    shape = (len(lats), len(lons))
    if not near.any():
        return xr.DataArray(np.full((da.valid_time.size, *shape), np.nan), dims=("valid_time", "latitude", "longitude"),
                            coords={"valid_time": da.valid_time.values, "latitude": lats, "longitude": lons})
    src_lat, src_lon = da.latitude.values[near], da.longitude.values[near]
    tgt_lat, tgt_lon = (a.ravel() for a in np.meshgrid(lats, lons, indexing="ij"))
    scale = np.cos(np.radians(tgt_lat.mean()))
    dist = np.hypot(tgt_lat[:, None] - src_lat[None, :], (tgt_lon[:, None] - src_lon[None, :]) * scale)
    nearest = dist.argmin(axis=1)
    values = da.transpose("valid_time", ...).values[:, near][:, nearest]
    values[:, dist[np.arange(len(nearest)), nearest] > spacing] = np.nan
    return xr.DataArray(values.reshape(-1, *shape), dims=("valid_time", "latitude", "longitude"),
                        coords={"valid_time": da.valid_time.values, "latitude": lats, "longitude": lons})


def at_point(da, lat, lon):
    """A field at the grid point nearest (lat, lon), or NaN if the source has none nearby (see `sample_nearest`)."""
    return sample_nearest(da, [lat], [lon]).isel(latitude=0, longitude=0)


def _iso(t):
    return f"{pd.Timestamp(t):%Y-%m-%dT%H:%M}"


def _series(da):
    """{"t": [ISO valid times], "v": [values]}, leaving out missing values."""
    ok = np.isfinite(da.values)
    return {"t": [_iso(t) for t in da.valid_time.values[ok]], "v": np.round(da.values[ok], 2).tolist()}


def _map_frames(fields, times, to_grid):
    """Each source's field put on the map grid by to_grid, at `times`, as tenths (ints), None where it has no data."""
    out = {}
    for name, da in fields.items():
        grid = to_grid(da).reindex(valid_time=times)
        tenths = np.round(grid.values * 10).reshape(len(times), -1)
        out[name] = [[None if np.isnan(v) else int(v) for v in frame] for frame in tenths]
    return out


def _map_times(fields):
    """Every MAP_EVERY-hourly valid time any source has."""
    times = np.unique(np.concatenate([da.valid_time.values for da in fields.values()]))
    return times[pd.DatetimeIndex(times).hour % MAP_EVERY == 0]


def _scores(fields):
    """Agreement of each source with the benchmark on the common grid, at the valid times they share."""
    bm = regrid_to_box(fields[BENCHMARK], SCORE_EXTENT)
    rows = {}
    for name, da in fields.items():
        if name == BENCHMARK:
            continue
        a, b = xr.align(bm, regrid_to_box(da, SCORE_EXTENT), join="inner")
        ok = np.isfinite(a.values) & np.isfinite(b.values)
        if not ok.any():
            continue
        diff = b.values[ok] - a.values[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(a.values[ok], b.values[ok])[0, 1]
        rows[name] = {"times": int(a.valid_time.size), "bias": round(float(diff.mean()), 2),
                      "rmse": round(float(np.sqrt((diff ** 2).mean())), 2),
                      "corr": round(float(corr), 2) if np.isfinite(corr) else None}  # None for a constant field
    return rows


def _centres(extent, lat_res, lon_res):
    """(lats, lons) of the cell centres of a regular grid covering extent."""
    lon_min, lon_max, lat_min, lat_max = extent
    return (np.arange(lat_min + lat_res / 2, lat_max, lat_res).round(4),
            np.arange(lon_min + lon_res / 2, lon_max, lon_res).round(4))


def _maps(rain, wind, lats, lons, to_grid, outline):
    """One map area's data: its grid, outline, and each field's frames at every MAP_EVERY hours."""
    out = {"lats": lats.tolist(), "lons": lons.tolist(), "outline": json.loads((HERE / outline).read_text())}
    for field, values in [("rain", rain), ("wind", wind)]:
        times = _map_times(values)
        out[field] = {"times": [_iso(t) for t in times], "values": _map_frames(values, times, to_grid)}
    return out


def site_data(forecasts, failures=None, built=None):
    """Everything the page shows, as JSON-ready data, from standard forecasts (benchmark first)."""
    rain = {name: period_totals(ds, RAIN_HOURS) for name, ds in forecasts.items()}
    wind = {name: wind_speed(ds) for name, ds in forecasts.items()}
    benchmark = forecasts.get(BENCHMARK)
    return {
        "built": _iso(built or datetime.now(timezone.utc).replace(tzinfo=None)),
        "benchmark": BENCHMARK,
        "benchmark_end": _iso(benchmark.valid_time.values[-1]) if benchmark is not None else None,
        "rain_hours": RAIN_HOURS,
        "sources": [{"name": name, "run": _iso(run_time(ds)), "end": _iso(ds.valid_time.values[-1]),
                     "grid_km": round(grid_km(ds), 1)}
                    for name, ds in forecasts.items()],
        "failures": dict(failures or {}),
        "locations": {place: {"rain": {name: _series(at_point(da, *where)) for name, da in rain.items()},
                              "wind": {name: _series(at_point(da, *where)) for name, da in wind.items()}}
                      for place, where in LOCATIONS.items()},
        "maps": {"uk": _maps(rain, wind, *_centres(UK_EXTENT, MAP_RES, MAP_RES),
                             lambda da: regrid_to_box(da, UK_EXTENT, MAP_RES), "uk_coastline.json"),
                 "london": _maps(rain, wind, *_centres(LONDON_EXTENT, *LONDON_RES),
                                 lambda da, ll=_centres(LONDON_EXTENT, *LONDON_RES): sample_nearest(da, *ll),
                                 "london_outline.json")},
        "scores": {"rain": _scores(rain), "wind": _scores(wind)} if benchmark is not None else {},
    }


def derive_key(password):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), PBKDF2_SALT, PBKDF2_ITERATIONS, dklen=32)


def encrypt(data, password):
    """gzipped JSON (with no NaN, which browsers can't parse) encrypted with AES-256-GCM: a 12-byte IV, then the ciphertext and tag."""
    iv = secrets.token_bytes(12)
    payload = gzip.compress(json.dumps(data, separators=(",", ":"), allow_nan=False).encode())
    return iv + AESGCM(derive_key(password)).encrypt(iv, payload, None)


def decrypt(blob, password):
    """The data `encrypt` encrypted (raises cryptography's InvalidTag for a wrong password)."""
    return json.loads(gzip.decompress(AESGCM(derive_key(password)).decrypt(blob[:12], blob[12:], None)))


def build(out, password, forecasts, failures=None):
    """Write index.html and forecast.bin to `out`."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"SITE_PASSWORD must be at least {MIN_PASSWORD_LENGTH} characters")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob = encrypt(site_data(forecasts, failures), password)
    (out / "forecast.part").write_bytes(blob)
    (out / "forecast.part").rename(out / "forecast.bin")
    shutil.copyfile(HERE / "index.html", out / "index.html")
    (out / ".nojekyll").touch()
    return out / "forecast.bin"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="latestForecasts/data", help="forecast store folder")
    parser.add_argument("--out", default="_site", help="folder to write the site to")
    parser.add_argument("--no-refresh", action="store_true", help="use the stored forecasts, downloading nothing")
    args = parser.parse_args()

    load_dotenv()
    password = os.environ.get("SITE_PASSWORD")
    if not password:
        raise SystemExit("Set SITE_PASSWORD (in the environment or .env) to build the site")
    store = ForecastStore(args.data_dir, extent=UK_EXTENT)
    if args.no_refresh:
        forecasts = {name: ds for name in [BENCHMARK, *forecast_sources()] if (ds := store.load(name)) is not None}
    else:
        forecasts = store.refresh(horizon_hours=HORIZON_HOURS, progress=lambda message: print(message, flush=True))
    if BENCHMARK not in forecasts:
        raise SystemExit("No benchmark forecast stored: run without --no-refresh")
    for name, error in store.failures.items():
        print(f"{name} FAILED: {error}")
    path = build(args.out, password, forecasts, store.failures)
    print(f"Wrote {args.out}/index.html and {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
