"""The website build, from synthetic forecasts: the data the page shows, and its encryption. No network."""

import re
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr
from cryptography.exceptions import InvalidTag

from loaders import BENCHMARK
from test_store_offline import synthetic
from website import build as site

RUN = datetime(2026, 11, 3, 0)
PASSWORD = "correct horse battery"


@pytest.fixture
def forecasts():
    # Over the UK box, so they cover the locations and maps
    def uk(name, run, steps, step_hours=1):
        ds = synthetic(name, run, steps, step_hours)
        return ds.interp(latitude=[50.0, 53.0, 56.0], longitude=[-6.0, -2.0, 1.0], method="nearest",
                         kwargs={"fill_value": "extrapolate"})
    return {BENCHMARK: uk(BENCHMARK, RUN, 23), "NOAA AIGFS": uk("NOAA AIGFS", RUN + timedelta(hours=6), 12, 6)}


def test_site_data_has_each_sources_series_at_every_location(forecasts):
    data = site.site_data(forecasts, failures={"X": "RuntimeError: down"})
    assert [s["name"] for s in data["sources"]] == [BENCHMARK, "NOAA AIGFS"]
    assert data["benchmark_end"] == "2026-11-03T22:00" and data["failures"] == {"X": "RuntimeError: down"}
    assert set(data["locations"]) == set(site.LOCATIONS)
    london = data["locations"]["Central London"]
    # 6-hour totals start 6 hours into each run; 0.5 mm an hour on the benchmark, 3 mm per 6-hourly step
    assert london["rain"][BENCHMARK]["t"][0] == "2026-11-03T06:00"
    assert london["rain"][BENCHMARK]["v"][0] == pytest.approx(3.0)
    assert london["rain"]["NOAA AIGFS"]["t"][:2] == ["2026-11-03T12:00", "2026-11-03T18:00"]
    assert london["wind"][BENCHMARK]["v"][0] == pytest.approx(5.0)  # (3, -4) m s-1


@pytest.mark.parametrize("area", ["uk", "london"])
def test_maps_are_every_3_hours_with_gaps_where_a_source_has_no_time(forecasts, area):
    maps = site.site_data(forecasts)["maps"][area]
    assert maps["outline"]["lon"] and len(maps["outline"]["lon"]) == len(maps["outline"]["lat"])
    cells = len(maps["lons"]) * len(maps["lats"])
    assert all(t[11:13] in {"00", "03", "06", "09", "12", "15", "18", "21"} for t in maps["wind"]["times"])
    for name, frames in maps["wind"]["values"].items():
        assert len(frames) == len(maps["wind"]["times"]) and all(len(f) == cells for f in frames)
    # AIGFS (06Z run, 6-hourly) has nothing at 09:00
    i = maps["wind"]["times"].index("2026-11-03T09:00")
    assert all(v is None for v in maps["wind"]["values"]["NOAA AIGFS"][i])
    assert 50 in maps["wind"]["values"][BENCHMARK][i]  # 5.0 m s-1, in tenths


def test_scores_compare_each_source_with_the_benchmark(forecasts):
    scores = site.site_data(forecasts)["scores"]
    assert list(scores["wind"]) == ["NOAA AIGFS"]
    assert scores["wind"]["NOAA AIGFS"]["bias"] == 0
    assert scores["wind"]["NOAA AIGFS"]["corr"] is None  # constant wind has no correlation (not NaN, for JSON)


def test_encrypted_data_needs_the_password(forecasts):
    data = site.site_data(forecasts)
    blob = site.encrypt(data, PASSWORD)
    assert b"London" not in blob and b"Met Office" not in blob
    assert site.decrypt(blob, PASSWORD) == data
    with pytest.raises(InvalidTag):
        site.decrypt(blob, "wrong password!!")


def test_build_writes_the_page_and_data(forecasts, tmp_path):
    path = site.build(tmp_path, PASSWORD, forecasts)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".nojekyll", "forecast.bin", "index.html"]
    assert site.decrypt(path.read_bytes(), PASSWORD)["benchmark"] == BENCHMARK


def test_short_passwords_are_refused(forecasts, tmp_path):
    with pytest.raises(ValueError, match="at least 12"):
        site.build(tmp_path, "short", forecasts)


def test_the_page_derives_keys_like_the_build():
    page = (site.HERE / "index.html").read_text()
    assert int(re.search(r"PBKDF2_ITERATIONS = (\d+)", page)[1]) == site.PBKDF2_ITERATIONS
    assert re.search(r'PBKDF2_SALT = "([^"]+)"', page)[1].encode() == site.PBKDF2_SALT


def _field(lats, lons, values):
    return xr.DataArray(np.asarray(values, float)[None], dims=("valid_time", "latitude", "longitude"),
                        coords={"valid_time": [np.datetime64("2026-11-03T00:00")], "latitude": lats, "longitude": lons})


def test_sampling_takes_each_points_nearest_grid_value():
    coarse = _field([51.0, 51.25, 51.5, 51.75], [-0.5, -0.25, 0.0, 0.25], np.arange(16).reshape(4, 4))
    out = site.sample_nearest(coarse, [51.49, 51.51, 51.7], [-0.13, -0.1, 0.2])
    np.testing.assert_array_equal(out.isel(valid_time=0), [[9, 10, 11], [9, 10, 11], [13, 14, 15]])
    assert list(out.latitude.values) == [51.49, 51.51, 51.7]  # the target grid, not the source's


def test_sampling_gives_nan_outside_a_sources_area():
    regional = _field([51.0, 51.02, 51.04], [0.0, 0.02, 0.04], np.ones((3, 3)))
    out = site.sample_nearest(regional, [51.02, 51.02], [0.02, -1.0])
    assert out.isel(valid_time=0, latitude=0, longitude=0) == 1
    assert np.isnan(out.isel(valid_time=0, latitude=0, longitude=1))  # far west of its grid: not its edge value
    assert np.isnan(site.at_point(regional, 55.0, 0.02).item())


def test_sampling_a_2d_grid():
    lat, lon = np.meshgrid(np.arange(51.0, 52.0, 0.02), np.arange(-0.5, 0.5, 0.03), indexing="ij")
    curved = xr.DataArray((lat * 100 + lon)[None], dims=("valid_time", "y", "x"),
                          coords={"valid_time": [np.datetime64("2026-11-03T00:00")],
                                  "latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)})
    out = site.sample_nearest(curved, [51.5, 53.0], [0.01])
    assert float(out.isel(valid_time=0, latitude=0, longitude=0)) == pytest.approx(51.5 * 100 + 0.01, abs=1.6)
    assert np.isnan(out.isel(valid_time=0, latitude=1, longitude=0))  # beyond the grid
