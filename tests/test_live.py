"""Checks against the live servers, for spotting outages, delays or changed file layouts.

Opt-in (they need the network and depend on today's runs): `pytest -m live`. Each downloads one field
to a temporary folder (a few MB; the Met Office global one is ~6 MB). The Met Office DataHub check needs
MET_OFFICE_API_KEY and uses no data allowance.
"""

from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv

from conftest import SOURCE_NAMES
from loaders import ECMWFLoader, GFSLoader, MetOfficeGlobalLoader, MetOfficeLoader, forecast_sources
from loaders._http import exists
from loaders._per_file import PerFileGribLoader

pytestmark = pytest.mark.live

# Longest expected time from a run's start until its step 6 is published, plus the time until the next run
MAX_RUN_AGE = {
    "Met Office global": timedelta(hours=14),
    "ECMWF IFS": timedelta(hours=16),
    "ECMWF AIFS": timedelta(hours=16),
    "NOAA GFS": timedelta(hours=12),
    "NOAA AIGFS": timedelta(hours=14),
    "DWD ICON-EU": timedelta(hours=9),
    "DWD ICON-D2": timedelta(hours=7),
    "Canada GDPS": timedelta(hours=20),
}
# Fields load_standard reads, for sources with one file per field
PER_FILE_FIELDS = {
    "DWD ICON-EU": ["TOT_PREC", "U_10M", "V_10M"],
    "DWD ICON-D2": ["TOT_PREC", "U_10M", "V_10M"],
    "Canada GDPS": ["Precip-Accum_Sfc", "WindU_AGL-10m", "WindV_AGL-10m"],
}
ARCHIVE_VARIABLES = ["precipitation_accumulation-PT01H", "wind_speed_at_10m", "wind_direction_at_10m"]


@pytest.fixture(params=SOURCE_NAMES)
def source(request, tmp_path):
    return forecast_sources(tmp_path)[request.param]


def test_a_recent_run_is_published(source):
    run = source.latest_run(step_hours=6)
    age = datetime.now(timezone.utc).replace(tzinfo=None) - run
    assert age < MAX_RUN_AGE[source.name], f"latest run {run:%Y-%m-%d %H}Z is {age} old"


def test_the_fields_load_standard_needs_are_published(source):
    run = source.latest_run(step_hours=6)
    if isinstance(source, ECMWFLoader):
        listed = {f["param"] for f in source.list_params(run, 6) if "levelist" not in f}
        assert {"tp", "10u", "10v"} <= listed
    elif isinstance(source, GFSLoader):
        listed = {(f["variable"], f["level"]) for f in source.list_fields(run, 6)}
        assert {("APCP", "surface"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")} <= listed
    elif isinstance(source, PerFileGribLoader):
        for field in PER_FILE_FIELDS[source.name]:
            assert exists(source.file_url(run, 6, field)), f"no {field} file"
    else:
        assert isinstance(source, MetOfficeGlobalLoader)
        assert set(ARCHIVE_VARIABLES) <= set(source.list_variables(run, 6))


def test_a_field_downloads_as_a_valid_file(source):
    run = source.latest_run(step_hours=6)
    if isinstance(source, ECMWFLoader):
        path = source.download(run, 6, "10u")
    elif isinstance(source, GFSLoader):
        path = source.download(run, 6, "UGRD", "10 m above ground")
    elif isinstance(source, PerFileGribLoader):
        path = source.download(run, 6, PER_FILE_FIELDS[source.name][1])
    else:
        path = source.download(run, 6, "wind_speed_at_10m")
        assert path.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"  # NetCDF-4 is an HDF5 file
        return
    data = path.read_bytes()
    # One GRIB message (a whole file, or a byte range) starts with "GRIB" and ends with "7777"
    assert data[:4] == b"GRIB" and data[-4:] == b"7777"


def test_met_office_order_has_the_fields_standardise_needs():
    load_dotenv()
    loader = MetOfficeLoader()
    try:
        orders = loader.list_orders()
    except RuntimeError as e:  # no API key
        pytest.skip(str(e))
    assert orders, "no active DataHub orders"
    file_ids = [f["fileId"] for f in loader.list_latest_files(orders[0]["orderId"])]
    for part in ["rainfall-accumulation-1h", "u-component-of-wind", "v-component-of-wind"]:
        assert any(part in f for f in file_ids), f"order has no {part} file"
