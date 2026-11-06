"""Weights for the means of sources whose steps are further apart than an hour.

Sources are interpolated onto every hour (`loaders.standard.to_hourly`) so the means use every source
at every hour. An interpolated value is less certain than a real one, so it counts for less: each
value's weight is σ_model² / (σ_model² + σ_interp²), from inverse-variance weighting with every source
assumed equally accurate. A real value has weight 1.

- σ_interp² is how far interpolation strays from what a source really forecast, by gap length and
  hours into the gap. It's measured on the hourly sources: keep only every gap-th hour, interpolate
  back and compare with their real hourly values.
- σ_model² is a source's own forecast error. If two sources have independent errors of the same size,
  the variance of their difference is 2σ_model², so it's half the mean squared difference between
  pairs of sources, at the times both really forecast. No observations are needed. It grows with lead
  time, so it's worked out for each LEAD_BAND_HOURS band after the reference time.

Both are measured on the 0.5° grid over the given extent, from this build's forecasts only.
"""

from itertools import combinations

import numpy as np
import xarray as xr

from loaders.standard import period_totals, regrid_to_box, run_time, to_hourly, wind_speed

LEAD_BAND_HOURS = 24
LEAD_BANDS = 3  # 0-24 h, 24-48 h, then everything after 48 h
HOUR = np.timedelta64(1, "h")
# The page draws values with weight 1 as real and anything less as interpolated, after rounding to 2 dp
MAX_INTERPOLATED_WEIGHT = 0.99


def positions(real_times, times):
    """For each of `times`: (gap, offset), the hours between the real valid times either side of it and
    the hours since the earlier one. (0, 0) at a real time."""
    real_times, times = np.asarray(real_times), np.asarray(times)
    after = np.searchsorted(real_times, times, side="left").clip(0, len(real_times) - 1)
    before = (np.searchsorted(real_times, times, side="right") - 1).clip(0, len(real_times) - 1)
    gap = ((real_times[after] - real_times[before]) // HOUR).astype(int)
    offset = ((times - real_times[before]) // HOUR).astype(int)
    return gap, np.where(gap > 0, offset, 0)


def field_positions(field, real_times, times, rain_hours):
    """`positions` for a field's values: a rain total is placed by its end, or by its start when only that
    is interpolated (for gaps that divide rain_hours, both are the same hours into their gaps)."""
    gap, offset = positions(real_times, times)
    if field == "rain":
        g0, k0 = positions(real_times, np.asarray(times) - rain_hours * HOUR)
        gap, offset = np.where(gap > 0, gap, g0), np.where(gap > 0, offset, k0)
    return gap, offset


def fields(ds, rain_hours):
    """{"rain": rain totals, "wind": wind speed} from a standard forecast."""
    return {"rain": period_totals(ds, rain_hours), "wind": wind_speed(ds)}


def coarse(ds, extent):
    """A standard forecast averaged onto the 0.5° grid over extent (see `regrid_to_box`)."""
    return xr.Dataset({name: regrid_to_box(da, extent) for name, da in ds.data_vars.items()}, attrs=ds.attrs)


def _subsampled(ds, gap):
    """A forecast keeping only every gap-th hour from its first valid time, as a coarser source would have."""
    hours = (ds.valid_time.values - ds.valid_time.values[0]) // HOUR
    return ds.isel(valid_time=np.flatnonzero(hours % gap == 0))


def interp_variance(hourly_sources, gaps, rain_hours):
    """{field: {(gap, offset): σ_interp²}} from hourly sources (coarse standard forecasts), for each gap
    in `gaps` and each hour into it. Pooled over sources, grid cells and times."""
    sums = {}
    for ds in hourly_sources:
        truth = fields(ds, rain_hours)
        for gap in gaps:
            real = _subsampled(ds, gap)
            if real.valid_time.size < 2:
                continue
            for field, da in fields(to_hourly(real), rain_hours).items():
                a, b = xr.align(truth[field], da, join="inner")
                g, k = field_positions(field, real.valid_time.values, a.valid_time.values, rain_hours)
                sq = ((b - a) ** 2).values.reshape(a.valid_time.size, -1)
                for i in np.flatnonzero(g == gap):
                    ok = np.isfinite(sq[i])
                    s, n = sums.setdefault((field, gap, int(k[i])), [0.0, 0])
                    sums[(field, gap, int(k[i]))] = [s + sq[i][ok].sum(), n + ok.sum()]
    out = {"rain": {}, "wind": {}}
    for (field, gap, k), (s, n) in sums.items():
        if n:
            out[field][(gap, k)] = s / n
    return out


def lead_band(times, reference):
    """Which LEAD_BAND_HOURS band after `reference` each valid time is in (0 to LEAD_BANDS - 1)."""
    hours = (np.asarray(times) - np.datetime64(reference)) / HOUR
    return (hours // LEAD_BAND_HOURS).clip(0, LEAD_BANDS - 1).astype(int)


def model_variance(sources, reference, rain_hours):
    """{field: [σ_model² for each lead band]} from coarse standard forecasts at their real valid times:
    half the mean squared difference over every pair of sources, grid cells and shared times."""
    by_source = [fields(ds, rain_hours) for ds in sources]
    out = {}
    for field in ("rain", "wind"):
        s, n = np.zeros(LEAD_BANDS), np.zeros(LEAD_BANDS)
        for x, y in combinations(by_source, 2):
            a, b = xr.align(x[field], y[field], join="inner")
            if a.valid_time.size == 0:
                continue
            sq = ((a - b) ** 2).values.reshape(a.valid_time.size, -1)
            ok = np.isfinite(sq)
            np.add.at(s, lead_band(a.valid_time.values, reference), np.where(ok, sq, 0).sum(axis=1))
            np.add.at(n, lead_band(a.valid_time.values, reference), ok.sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s / n / 2
        # A band no pair shares (e.g. only one source reaches it) takes the nearest band's value
        known = np.flatnonzero(np.isfinite(var))
        out[field] = [float(var[known[np.abs(known - b).argmin()]]) if len(known) else None for b in range(LEAD_BANDS)]
    return out


class Weights:
    """Each source's weight at each hour, from the forecasts as loaded (real valid times only)."""

    def __init__(self, forecasts, extent, rain_hours, reference=None):
        self.rain_hours = rain_hours
        self.real_times = {name: ds.valid_time.values for name, ds in forecasts.items()}
        self.reference = reference if reference is not None else min(run_time(ds) for ds in forecasts.values())
        small = {name: coarse(ds, extent) for name, ds in forecasts.items()}
        gaps = sorted({int(g) for t in self.real_times.values() for g in np.diff(t) // HOUR if g > 1})
        hourly = [ds for name, ds in small.items() if (np.diff(self.real_times[name]) == HOUR).all()]
        self.interp = interp_variance(hourly, gaps, rain_hours)
        self.model = model_variance(list(small.values()), self.reference, rain_hours)

    def weights(self, name, field, times):
        """Weights of a source's values of field ("rain" or "wind") at valid times `times`: 1 where real,
        at most MAX_INTERPOLATED_WEIGHT where interpolated."""
        times = np.asarray(times)
        g, k = field_positions(field, self.real_times[name], times, self.rain_hours)
        model = np.array(self.model[field], dtype=float)[lead_band(times, self.reference)]
        interp = np.array([self.interp[field].get((gi, ki), np.nan) if gi else 0.0 for gi, ki in zip(g, k)])
        with np.errstate(invalid="ignore", divide="ignore"):
            w = model / (model + interp)
        # Without the variances to work it out (no hourly sources, or sources that never overlap), count
        # almost fully. Interpolated values are kept below 1 so the page can tell them from real ones
        w = np.where(np.isfinite(w), w, 1.0)
        return np.where(g > 0, np.minimum(w, MAX_INTERPOLATED_WEIGHT), 1.0)

    def summary(self):
        """The variances, JSON-ready: σ_model² by lead band and σ_interp² by gap and hour into it."""
        return {"lead_band_hours": LEAD_BAND_HOURS,
                "model_variance": {f: [None if v is None else round(v, 3) for v in vs] for f, vs in self.model.items()},
                "interp_variance": {f: {f"{g}h+{k}": round(v, 3) for (g, k), v in sorted(t.items())}
                                    for f, t in self.interp.items()}}
