"""Loader logic with a fake HTTP session: index parsing, field choice, URLs, caching and download safety.

No network access; the index files in fixtures/ are trimmed copies of real ones.
"""

import bz2
from datetime import datetime, timedelta, timezone

import pytest
import requests

from conftest import FIXTURES
from loaders import ECMWFLoader, GDPSLoader, GFSLoader, ICONEULoader, MetOfficeGlobalLoader
from loaders._http import download_to, session
from loaders.standard import last_step_by, standard_steps

RUN = datetime(2026, 10, 22, 0)


class FakeResponse:
    def __init__(self, status=200, body=b"", fail_after=None):
        self.status_code, self.content, self.fail_after = status, body, fail_after
        self.ok = status < 400
        self.text = body.decode(errors="replace")

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            if self.fail_after is not None and i >= self.fail_after:
                raise requests.ConnectionError("connection dropped")
            yield self.content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Serves `routes` (URL -> FakeResponse; anything else is a 404) and records each request."""

    def __init__(self, routes=None, head=None):
        self.routes, self.head_ok, self.requests = routes or {}, head, []

    def get(self, url, headers=None, **kwargs):
        self.requests.append((url, headers or {}))
        return self.routes.get(url, FakeResponse(404))

    def head(self, url, **kwargs):
        return FakeResponse(200 if self.head_ok(url) else 404)


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeSession in place of the loaders' shared session; configure it via its attributes."""
    fake = FakeSession()
    monkeypatch.setattr(session, "get", fake.get)
    monkeypatch.setattr(session, "head", fake.head)
    return fake


# --- GFS -------------------------------------------------------------------------------------

@pytest.mark.parametrize("model, url", [
    ("gfs", "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261022/00/atmos/gfs.t00z.pgrb2.0p25.f007"),
    ("aigfs", "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/aigfs.20261022/00/model/atmos/grib2/"
              "aigfs.t00z.sfc.f007.grib2"),
])
def test_gfs_url(model, url):
    assert GFSLoader(model=model).file_url(RUN, 7) == url


def test_aigfs_files_are_cached_separately_from_gfs(fake, tmp_path):
    loader = GFSLoader(data_dir=tmp_path, model="aigfs")
    url = loader.file_url(RUN, 7)
    fake.routes[url + ".idx"] = FakeResponse(body=(FIXTURES / "gfs_f007.idx").read_bytes())
    fake.routes[url] = FakeResponse(body=b"GRIB...7777")
    assert loader.download(RUN, 7, "APCP", "surface") == tmp_path / "aigfs" / "2026102200" / "7h-APCP-surface.grib2"


def test_gfs_index_lengths_come_from_next_offset(fake):
    fake.routes[GFSLoader().file_url(RUN, 7) + ".idx"] = FakeResponse(body=(FIXTURES / "gfs_f007.idx").read_bytes())
    fields = GFSLoader().list_fields(RUN, 7)
    assert fields[1] == {"variable": "UGRD", "level": "10 m above ground", "forecast": "7 hour fcst",
                         "offset": 986998, "length": 1950000 - 986998}
    assert fields[-1]["length"] is None  # last field runs to the end of the file


@pytest.mark.parametrize("variable, level, expected_range", [
    # APCP is listed for 6-7 h and 0-7 h: the run total (0-7 h) must be chosen
    ("APCP", "surface", "bytes=3131000-3399999"),
    # The last field has no next offset, so the range is open-ended
    ("TMP", "2 m above ground", "bytes=3400000-"),
])
def test_gfs_download_requests_the_right_bytes(fake, tmp_path, variable, level, expected_range):
    url = GFSLoader().file_url(RUN, 7)
    fake.routes[url + ".idx"] = FakeResponse(body=(FIXTURES / "gfs_f007.idx").read_bytes())
    fake.routes[url] = FakeResponse(body=b"GRIB...7777")
    out = GFSLoader(data_dir=tmp_path).download(RUN, 7, variable, level)
    assert fake.requests[-1] == (url, {"Range": expected_range})
    assert out.read_bytes() == b"GRIB...7777"


def test_gfs_download_rejects_unknown_field(fake, tmp_path):
    fake.routes[GFSLoader().file_url(RUN, 7) + ".idx"] = FakeResponse(body=(FIXTURES / "gfs_f007.idx").read_bytes())
    with pytest.raises(ValueError, match="No 'UGRD' at '100 m above ground'"):
        GFSLoader(data_dir=tmp_path).download(RUN, 7, "UGRD", "100 m above ground")


# --- ECMWF -----------------------------------------------------------------------------------

@pytest.mark.parametrize("source, host", [
    ("ecmwf", "https://data.ecmwf.int/forecasts"),
    ("aws", "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"),
])
def test_ecmwf_url(source, host):
    loader = ECMWFLoader(model="aifs-single", source=source)
    assert loader.file_url(RUN, 6, "index") == f"{host}/20261022/00z/aifs-single/0p25/oper/20261022000000-6h-oper-fc.index"


def test_ecmwf_download_requests_the_right_bytes(fake, tmp_path):
    loader = ECMWFLoader(data_dir=tmp_path)
    fake.routes[loader.file_url(RUN, 6, "index")] = FakeResponse(body=(FIXTURES / "ecmwf_6h.index").read_bytes())
    fake.routes[loader.file_url(RUN, 6, "grib2")] = FakeResponse(body=b"GRIB...7777")
    out = loader.download(RUN, 6, "tp")
    assert fake.requests[-1] == (loader.file_url(RUN, 6, "grib2"), {"Range": "bytes=15022145-15334068"})
    assert out == tmp_path / "ecmwf" / "ifs" / "2026102200" / "6h-tp.grib2"


def test_ecmwf_download_only_takes_single_level_fields(fake, tmp_path):
    loader = ECMWFLoader(data_dir=tmp_path)
    fake.routes[loader.file_url(RUN, 6, "index")] = FakeResponse(body=(FIXTURES / "ecmwf_6h.index").read_bytes())
    with pytest.raises(ValueError, match="single-level 't'"):
        loader.download(RUN, 6, "t")  # only on pressure levels


# --- Per-file sources (DWD, Canada) ---------------------------------------------------------

def test_icon_eu_url():
    assert ICONEULoader().file_url(RUN, 7, "TOT_PREC") == (
        "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/tot_prec/"
        "icon-eu_europe_regular-lat-lon_single-level_2026102200_007_TOT_PREC.grib2.bz2")


def test_gdps_url():
    assert GDPSLoader().file_url(RUN, 7, "WindU_AGL-10m") == (
        "https://dd.weather.gc.ca/20261022/WXO-DD/model_gdps/15km/00/007/"
        "20261022T00Z_MSC_GDPS_WindU_AGL-10m_LatLon0.15_PT007H.grib2")


def test_bz2_files_are_saved_decompressed(fake, tmp_path):
    loader = ICONEULoader(data_dir=tmp_path)
    fake.routes[loader.file_url(RUN, 7, "U_10M")] = FakeResponse(body=bz2.compress(b"GRIB...7777"))
    out = loader.download(RUN, 7, "U_10M")
    assert out == tmp_path / "icon-eu" / "2026102200" / "7h-U_10M.grib2"
    assert out.read_bytes() == b"GRIB...7777"


def test_truncated_bz2_download_leaves_no_file(fake, tmp_path):
    loader = ICONEULoader(data_dir=tmp_path)
    fake.routes[loader.file_url(RUN, 7, "U_10M")] = FakeResponse(body=bz2.compress(b"GRIB" * 1000)[:-10])
    with pytest.raises(OSError, match="Truncated"):
        loader.download(RUN, 7, "U_10M")
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


# --- Shared behaviour ------------------------------------------------------------------------

ALL_LOADERS = [ECMWFLoader(), GFSLoader(), GFSLoader(model="aigfs"), ICONEULoader(), GDPSLoader(),
               MetOfficeGlobalLoader()]
ALL_IDS = ["ECMWF", "GFS", "AIGFS", "ICON-EU", "GDPS", "Met Office global"]


def _newest_run(run_hours):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(hour=now.hour - now.hour % run_hours, minute=0, second=0, microsecond=0)


@pytest.mark.parametrize("loader", ALL_LOADERS, ids=ALL_IDS)
def test_latest_run_falls_back_to_older_runs(fake, loader):
    newest = _newest_run(loader.run_hours)
    # How the newest run appears in each source's URLs: pretend it isn't published yet
    markers = [f"{newest:%Y%m%d%H}0000-", f".{newest:%Y%m%d}/{newest:%H}/", f"_{newest:%Y%m%d%H}_",
               f"{newest:%Y%m%dT%H}Z_", f"/{newest:%Y%m%dT%H%M}Z/"]
    fake.head_ok = lambda url: not any(m in url for m in markers)
    assert loader.latest_run(step_hours=24) == newest - timedelta(hours=loader.run_hours)


@pytest.mark.parametrize("loader", ALL_LOADERS, ids=ALL_IDS)
def test_latest_run_raises_when_nothing_published(fake, loader):
    fake.head_ok = lambda url: False
    with pytest.raises(RuntimeError, match="No .* run"):
        loader.latest_run(step_hours=24)


@pytest.mark.parametrize("loader", ALL_LOADERS, ids=ALL_IDS)
def test_steps_off_the_schedule_are_refused(loader):
    assert standard_steps(loader, loader.max_step)[-1] == loader.max_step
    with pytest.raises(ValueError, match="only"):
        standard_steps(loader, loader.max_step + loader.step_hours)  # beyond max_step, steps are further apart


def test_steps_switch_to_the_later_spacing():
    loader = MetOfficeGlobalLoader()  # hourly to 54 h, then 3-hourly
    assert standard_steps(loader, 60) == [*range(55), 57, 60]
    assert last_step_by(loader, 59) == 57
    assert last_step_by(loader, 30) == 30


def test_sources_without_later_steps_stop_at_max_step():
    assert last_step_by(GFSLoader(), 200) == 120
    assert last_step_by(ECMWFLoader(model="aifs-single"), 71) == 66


@pytest.mark.parametrize("loader", ALL_LOADERS, ids=ALL_IDS)
def test_every_source_reaches_72_hours(loader):
    assert last_step_by(loader, 72) == 72


def test_download_skips_files_already_on_disk(fake, tmp_path):
    cached = tmp_path / "gfs" / "2026102200" / "7h-APCP-surface.grib2"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    assert GFSLoader(data_dir=tmp_path).download(RUN, 7, "APCP", "surface") == cached
    assert fake.requests == []


def test_interrupted_download_leaves_no_file(fake, tmp_path):
    fake.routes["https://example.com/f"] = FakeResponse(body=b"x" * (1 << 18), fail_after=1 << 16)
    out = tmp_path / "f.grib2"
    with pytest.raises(requests.ConnectionError):
        download_to(out, "https://example.com/f", timeout=1)
    assert list(tmp_path.iterdir()) == []


def test_http_errors_are_raised(fake, tmp_path):
    with pytest.raises(requests.HTTPError):
        download_to(tmp_path / "f.grib2", "https://example.com/missing", timeout=1)
