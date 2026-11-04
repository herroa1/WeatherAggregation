"""Each source agrees roughly with the Met Office benchmark over mainland UK (or, for a regional source,
the part it covers).

The tolerances are loose on purpose: these catch broken data (wrong units, flipped or shifted grids,
the wrong field), not judge forecast skill. Both forecasts are averaged onto a common 0.5° grid and
compared at the valid times they share.
"""

import numpy as np
import pytest
import xarray as xr

from conftest import CORE_UK, SOURCE_NAMES
from loaders.standard import period_totals, regrid_to_box, wind_speed

RES = 0.5
RAIN_HOURS = 6


def _on_common_grid(benchmark, other, field):
    """Both forecasts' field on the common grid, at their shared valid times."""
    a, b = (regrid_to_box(field(ds), CORE_UK, RES) for ds in (benchmark, other))
    a, b = xr.align(a, b, join="inner")
    if a.valid_time.size == 0:
        pytest.fail("no valid times shared with the benchmark")
    # Only where both have data, so a regional source is compared over its own area
    both = np.isfinite(a) & np.isfinite(b)
    return a.where(both), b.where(both)


def _correlation(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a.values[ok], b.values[ok])[0, 1])


@pytest.fixture(params=SOURCE_NAMES)
def pair(request, forecasts):
    """(benchmark, other source) for each non-benchmark source."""
    return forecasts("Met Office"), forecasts(request.param)


def test_wind_pattern_matches_benchmark(pair):
    a, b = _on_common_grid(*pair, wind_speed)
    for t in a.valid_time.values:
        r = _correlation(a.sel(valid_time=t), b.sel(valid_time=t))
        # A flipped, shifted or wrong field correlates near zero (or negatively)
        assert r > 0.5, f"wind pattern correlation {r:.2f} at {t}"


def test_wind_direction_matches_benchmark(pair):
    # Cosine similarity of the (u, v) vector fields: 1 for the same directions, -1 for reversed ones
    # (e.g. a "from" direction used as "to"), which wind speed alone can't show
    (ua, ub), (va, vb) = (_on_common_grid(*pair, lambda ds, c=c: ds[c]) for c in ("u10", "v10"))
    ok = np.isfinite(ua) & np.isfinite(ub) & np.isfinite(va) & np.isfinite(vb)
    ua, ub, va, vb = (x.values[ok] for x in (ua, ub, va, vb))
    similarity = (ua * ub + va * vb).sum() / np.sqrt((ua ** 2 + va ** 2).sum() * (ub ** 2 + vb ** 2).sum())
    assert similarity > 0.7, f"wind direction similarity {similarity:.2f}"


def test_wind_level_matches_benchmark(pair):
    a, b = _on_common_grid(*pair, wind_speed)
    ratio = float(b.mean() / a.mean())
    assert 0.5 < ratio < 2, f"mean wind is {ratio:.2f}x the benchmark's"


def test_rain_level_matches_benchmark(pair):
    a, b = _on_common_grid(*pair, lambda ds: period_totals(ds, RAIN_HOURS))
    for t in a.valid_time.values:
        bench, other = float(a.sel(valid_time=t).mean()), float(b.sel(valid_time=t).mean())
        if bench >= 0.5:
            # A unit error (x1000 or /1000) lands far outside this
            assert 0.25 < other / bench < 4, f"mean {RAIN_HOURS} h rain {other:.2f} mm vs benchmark {bench:.2f} mm at {t}"
        else:
            assert abs(other - bench) < 2, f"mean {RAIN_HOURS} h rain {other:.2f} mm vs benchmark {bench:.2f} mm at {t}"


def test_rain_pattern_matches_benchmark(pair):
    a, b = _on_common_grid(*pair, lambda ds: period_totals(ds, RAIN_HOURS))
    # Summed over the shared times, so there's enough rain for a meaningful pattern
    a, b = a.sum("valid_time", min_count=1), b.sum("valid_time", min_count=1)
    if float(a.mean()) < 0.5:
        pytest.skip("too little benchmark rain to compare patterns")
    r = _correlation(a, b)
    assert r > 0.3, f"rain pattern correlation {r:.2f}"
