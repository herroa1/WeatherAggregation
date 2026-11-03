"""The website build, from synthetic forecasts: the data the page shows, and its encryption. No network."""

import re
from datetime import datetime, timedelta

import pytest
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
    london = data["locations"]["London"]
    # 6-hour totals start 6 hours into each run; 0.5 mm an hour on the benchmark, 3 mm per 6-hourly step
    assert london["rain"][BENCHMARK]["t"][0] == "2026-11-03T06:00"
    assert london["rain"][BENCHMARK]["v"][0] == pytest.approx(3.0)
    assert london["rain"]["NOAA AIGFS"]["t"][:2] == ["2026-11-03T12:00", "2026-11-03T18:00"]
    assert london["wind"][BENCHMARK]["v"][0] == pytest.approx(5.0)  # (3, -4) m s-1


def test_maps_are_every_3_hours_with_gaps_where_a_source_has_no_time(forecasts):
    maps = site.site_data(forecasts)["maps"]
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
