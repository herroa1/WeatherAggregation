"""The forecast store, with fake sources: saving, reuse of stored runs, re-fetching and pruning. No network."""

import threading
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from loaders import BENCHMARK
from loaders.standard import standard_dataset
from loaders.store import UK_EXTENT, ForecastStore

RUN = datetime(2026, 10, 27, 0)


def synthetic(name, run, steps, step_hours=1, rain_step=0.5):
    """A small standard forecast: rain rising by rain_step per step, wind of (3, -4) m s-1."""
    times = pd.Timestamp(run) + pd.to_timedelta(np.arange(steps) * step_hours, unit="h")
    coords = {"valid_time": times, "latitude": [51.0, 52.0], "longitude": [-1.0, 0.0, 1.0]}
    dims = ("valid_time", "latitude", "longitude")
    rain = xr.DataArray(np.arange(steps)[:, None, None] * rain_step * np.ones((1, 2, 3)), dims=dims, coords=coords)
    u, v = (xr.full_like(rain, value) for value in (3.0, -4.0))
    return standard_dataset(rain, u, v, source=name, run=run, step_hours=step_hours)


class FakeSource:
    def __init__(self, name, run, step_hours=1):
        self.name, self.run, self.step_hours, self.max_step = name, run, step_hours, 96
        self.fetches = 0

    def latest_run(self, step_hours):
        return self.run

    def load_standard(self, run, last_step, extent=None):
        self.fetches += 1
        return synthetic(self.name, run, last_step // self.step_hours + 1, self.step_hours)


class FakeMetOffice:
    def __init__(self, run, steps=23):
        self.run, self.steps, self.fetches = run, steps, 0

    def list_orders(self):
        return [{"orderId": "o1"}]

    def latest_order_run(self, order_id):
        return self.run

    def load_standard_latest(self, order_id):
        self.fetches += 1
        return synthetic(BENCHMARK, self.run, self.steps)


@pytest.fixture
def store(tmp_path):
    return ForecastStore(tmp_path)


def test_saved_forecasts_load_back_in_standard_format(store):
    ds = synthetic("NOAA GFS", RUN, 5)
    store.save(ds)
    loaded = store.load("NOAA GFS")
    assert loaded.attrs == ds.attrs and isinstance(loaded.attrs["step_hours"], int)
    assert {v: loaded[v].attrs["units"] for v in loaded.data_vars} == {v: ds[v].attrs["units"] for v in ds.data_vars}
    for v in ds.data_vars:
        np.testing.assert_allclose(loaded[v], ds[v], atol=0.005)  # 16-bit packing to 0.01


def test_tiny_negative_rain_is_saved_as_zero(store):
    ds = synthetic("NOAA GFS", RUN, 3)
    store.save(ds.assign(rain_mm=ds.rain_mm - 0.02))
    assert float(store.load("NOAA GFS").rain_mm.min()) == 0


def test_forecasts_saved_for_another_box_are_not_reused(tmp_path):
    ForecastStore(tmp_path, extent=(-25, 15, 45, 65)).save(synthetic("NOAA GFS", RUN, 3))
    assert ForecastStore(tmp_path, extent=UK_EXTENT).load("NOAA GFS") is None


def test_refresh_only_downloads_runs_not_already_stored(store):
    met_office, gfs = FakeMetOffice(RUN), FakeSource("NOAA GFS", RUN)
    first = store.refresh(sources={"NOAA GFS": gfs}, met_office=met_office)
    assert list(first) == [BENCHMARK, "NOAA GFS"]
    assert (met_office.fetches, gfs.fetches) == (1, 1)
    store.refresh(sources={"NOAA GFS": gfs}, met_office=met_office)
    assert (met_office.fetches, gfs.fetches) == (1, 1)  # both reused from the store

    gfs.run = RUN + timedelta(hours=6)  # a newer run is out
    latest = store.refresh(sources={"NOAA GFS": gfs}, met_office=met_office)
    assert gfs.fetches == 2
    assert latest["NOAA GFS"].attrs["run"] == "2026-10-27T06:00:00"
    assert store.runs("NOAA GFS") == [pd.Timestamp(gfs.run)]  # the older run was pruned


def test_steps_reach_72_hours_after_the_benchmark_run_beyond_the_benchmark_itself(store):
    latest = store.refresh(sources={"NOAA GFS": FakeSource("NOAA GFS", RUN + timedelta(hours=6))},
                           met_office=FakeMetOffice(RUN, steps=23))
    assert pd.Timestamp(latest["NOAA GFS"].valid_time.values[-1]) == pd.Timestamp(RUN) + pd.Timedelta(hours=72)
    assert latest[BENCHMARK].valid_time.size == 23  # the benchmark is unchanged


def test_the_horizon_can_be_changed(store):
    latest = store.refresh(sources={"NOAA GFS": FakeSource("NOAA GFS", RUN)}, met_office=FakeMetOffice(RUN),
                           horizon_hours=24)
    assert latest["NOAA GFS"].valid_time.size == 25


def test_a_stored_run_that_stops_too_early_is_fetched_again(store):
    gfs = FakeSource("NOAA GFS", RUN)
    store.save(synthetic("NOAA GFS", RUN, 10))  # e.g. from before the horizon was extended
    store.refresh(sources={"NOAA GFS": gfs}, met_office=FakeMetOffice(RUN, steps=23))
    assert gfs.fetches == 1
    assert store.load("NOAA GFS").valid_time.size == 73


def test_prune_keeps_the_newest_runs_and_removes_raw_caches_only(tmp_path):
    store = ForecastStore(tmp_path, keep=2)
    for hours in (0, 6, 12):
        store.save(synthetic("NOAA GFS", RUN + timedelta(hours=hours), 3))
    for folder in ["raw/gfs/2026102700", "gfs/2026102700", "global-deterministic-10km/x", "o144900492117/2026102700",
                   "my-notes"]:
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "readme.txt").write_text("keep me")

    store.prune()
    assert store.runs("NOAA GFS") == [pd.Timestamp(RUN + timedelta(hours=h)) for h in (6, 12)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["my-notes", "readme.txt", "standard"]


class FailingSource(FakeSource):
    def __init__(self, name, run, raw_dir):
        super().__init__(name, run)
        self.raw_dir = raw_dir

    def load_standard(self, run, last_step, extent=None):
        (self.raw_dir / "half-downloaded").mkdir(parents=True)
        raise RuntimeError("server error")


def test_a_failing_source_is_skipped_and_its_raw_files_deleted(store):
    gfs = FakeSource("NOAA GFS", RUN)
    with pytest.warns(UserWarning, match="X failed .*server error.*no stored run"):
        latest = store.refresh(sources={"X": FailingSource("X", RUN, store.raw_dir), "NOAA GFS": gfs},
                               met_office=FakeMetOffice(RUN))
    assert list(latest) == [BENCHMARK, "NOAA GFS"]  # the other sources still refresh
    assert not store.raw_dir.exists()


def test_a_failing_source_keeps_its_previous_run_if_it_overlaps_the_benchmark(store):
    store.save(synthetic("X", RUN - timedelta(hours=6), 12))  # valid to 5 h into the benchmark
    with pytest.warns(UserWarning, match="keeping its stored 2026-10-26T18:00:00 run"):
        latest = store.refresh(sources={"X": FailingSource("X", RUN, store.raw_dir)}, met_office=FakeMetOffice(RUN))
    assert latest["X"].attrs["run"] == "2026-10-26T18:00:00"
    assert store.failures == {"X": "RuntimeError: server error"}


def test_a_failing_source_drops_a_previous_run_too_old_to_compare(store):
    store.save(synthetic("X", RUN - timedelta(days=2), 24))  # ends a day before the benchmark starts
    with pytest.warns(UserWarning, match="no stored run overlapping the benchmark"):
        latest = store.refresh(sources={"X": FailingSource("X", RUN, store.raw_dir)}, met_office=FakeMetOffice(RUN))
    assert "X" not in latest


def test_simultaneous_refreshes_take_turns(tmp_path):
    """Two refreshes of one data folder (e.g. the notebook in two kernels) mustn't interleave downloads."""
    active, overlapped, errors = [0], [], []

    class SlowSource(FakeSource):
        def load_standard(self, run, last_step, extent=None):
            active[0] += 1
            overlapped.append(active[0] > 1)
            time.sleep(0.2)
            active[0] -= 1
            return super().load_standard(run, last_step, extent)

    def refresh():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a source failing (e.g. a clashing download) counts too
                ForecastStore(tmp_path).refresh(sources={"NOAA GFS": SlowSource("NOAA GFS", RUN)},
                                                met_office=FakeMetOffice(RUN))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=refresh) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert overlapped == [False]  # the second refresh waited, then found the run already stored
