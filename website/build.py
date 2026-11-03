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
# Maps are averaged onto this grid (degrees), every MAP_EVERY hours, to keep the data file small
MAP_RES = 0.5
MAP_EVERY = 3
# Mainland UK, which lies inside every source's grid: used for the scores
SCORE_EXTENT = (-8, 2, 50, 59)
LOCATIONS = {
    "London": (51.51, -0.13), "Birmingham": (52.48, -1.90), "Manchester": (53.48, -2.24),
    "Cardiff": (51.48, -3.18), "Edinburgh": (55.95, -3.19), "Belfast": (54.60, -5.93),
}
# Must match index.html. The salt is fixed so a key remembered on a phone works for later builds
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT = b"uk-forecast-comparison"
MIN_PASSWORD_LENGTH = 12


def at_point(da, lat, lon):
    """A field at the grid cell nearest (lat, lon), for 1D or 2D lat/lon grids."""
    if da.latitude.ndim == 1:
        return da.sel(latitude=lat, longitude=lon, method="nearest")
    dist = (da.latitude - lat) ** 2 + ((da.longitude - lon) * np.cos(np.radians(lat))) ** 2
    y, x = np.unravel_index(np.argmin(dist.values), dist.shape)
    return da.isel(y=y, x=x)


def _iso(t):
    return f"{pd.Timestamp(t):%Y-%m-%dT%H:%M}"


def _series(da):
    """{"t": [ISO valid times], "v": [values]}, leaving out missing values."""
    ok = np.isfinite(da.values)
    return {"t": [_iso(t) for t in da.valid_time.values[ok]], "v": np.round(da.values[ok], 2).tolist()}


def _map_frames(fields, times):
    """Each source's field on the common grid at `times`, as tenths (ints), None where it has no data."""
    out = {}
    for name, da in fields.items():
        grid = regrid_to_box(da, UK_EXTENT, MAP_RES).reindex(valid_time=times)
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


def site_data(forecasts, failures=None, built=None):
    """Everything the page shows, as JSON-ready data, from standard forecasts (benchmark first)."""
    rain = {name: period_totals(ds, RAIN_HOURS) for name, ds in forecasts.items()}
    wind = {name: wind_speed(ds) for name, ds in forecasts.items()}
    lon_min, lon_max, lat_min, lat_max = UK_EXTENT
    maps = {"lons": np.arange(lon_min + MAP_RES / 2, lon_max, MAP_RES).round(3).tolist(),
            "lats": np.arange(lat_min + MAP_RES / 2, lat_max, MAP_RES).round(3).tolist()}
    for field, values in [("rain", rain), ("wind", wind)]:
        times = _map_times(values)
        maps[field] = {"times": [_iso(t) for t in times], "values": _map_frames(values, times)}
    benchmark = forecasts.get(BENCHMARK)
    return {
        "built": _iso(built or datetime.now(timezone.utc).replace(tzinfo=None)),
        "benchmark": BENCHMARK,
        "benchmark_end": _iso(benchmark.valid_time.values[-1]) if benchmark is not None else None,
        "rain_hours": RAIN_HOURS,
        "sources": [{"name": name, "run": _iso(run_time(ds)), "end": _iso(ds.valid_time.values[-1])}
                    for name, ds in forecasts.items()],
        "failures": dict(failures or {}),
        "locations": {place: {"rain": {name: _series(at_point(da, *where)) for name, da in rain.items()},
                              "wind": {name: _series(at_point(da, *where)) for name, da in wind.items()}}
                      for place, where in LOCATIONS.items()},
        "maps": maps,
        "coast": json.loads((HERE / "uk_coastline.json").read_text()),
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
