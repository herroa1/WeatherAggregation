"""Shared fixtures.

Data tests check the latest set of forecasts in the forecast store (see `loaders/store.py`), which the
notebook keeps up to date, so by default they download nothing and make no DataHub calls. The
benchmark is the stored Met Office run, and each other source is its stored run covering the same
period. Sources missing from the store are skipped.

Pass `--refresh` to download the latest set first (as the notebook does): do this after changing a
loader, so the tests check its new output. It only uses DataHub allowance when there's a new Met
Office run.

Environment variables:
- WEATHER_TEST_DATA: data folder holding the store (default: latestForecasts/data)
"""

import os
from pathlib import Path

import pytest

from loaders import BENCHMARK, forecast_sources
from loaders.store import UK_EXTENT, ForecastStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WEATHER_TEST_DATA", ROOT / "latestForecasts" / "data"))
FIXTURES = Path(__file__).parent / "fixtures"

# Mainland UK, which lies inside every UK-wide source's grid (the Met Office grid is curved, so misses UK_EXTENT's corners)
CORE_UK = (-8, 2, 50, 59)

SOURCE_NAMES = list(forecast_sources())
ALL_NAMES = [BENCHMARK, *SOURCE_NAMES]


def covered_area(name):
    """The part of mainland UK a source has data everywhere in: CORE_UK, or less for a regional source
    (see `coverage` in `loaders.standard`)."""
    coverage = getattr(forecast_sources().get(name), "coverage", None)
    if coverage is None:
        return CORE_UK
    (a0, a1, a2, a3), (b0, b1, b2, b3) = CORE_UK, coverage
    return max(a0, b0), min(a1, b1), max(a2, b2), min(a3, b3)


def pytest_addoption(parser):
    parser.addoption("--refresh", action="store_true",
                     help="download the latest set of forecasts into the store before testing")


@pytest.fixture(scope="session")
def store(request):
    store = ForecastStore(DATA_DIR, extent=UK_EXTENT)
    if request.config.getoption("--refresh"):
        store.refresh()
    return store


@pytest.fixture(scope="session")
def benchmark(store):
    """The stored Met Office run, as a standard forecast."""
    ds = store.load(BENCHMARK)
    if ds is None:
        pytest.skip("No forecasts stored: run latestForecasts/aggregateForecasts.ipynb or `pytest --refresh`")
    return ds


@pytest.fixture(scope="session")
def forecasts(store, benchmark):
    """Function returning a source's stored standard forecast, loaded on first use."""
    cache = {BENCHMARK: benchmark}

    def get(name):
        if name not in cache:
            cache[name] = store.load(name)
        if cache[name] is None:
            pytest.skip(f"No {name} forecast stored: run the notebook or `pytest --refresh`")
        return cache[name]

    return get


@pytest.fixture(params=ALL_NAMES)
def forecast(request, forecasts):
    """Each source's standard forecast in turn, including the benchmark."""
    return forecasts(request.param)
