"""HTTP helpers shared by the loaders."""

import bz2
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Retries brief server errors (e.g. S3's occasional 503 "slow down"), with 1 s, 2 s, 4 s backoff
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET", "HEAD"])))


def download_to(out, url, timeout, decompress_bz2=False, **kwargs):
    """Stream a URL to `out`, creating its folder. Extra kwargs are passed to `session.get`.

    With `decompress_bz2`, the download is bzip2-decompressed as it arrives, so `out` holds the plain file.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first so an interrupted download isn't mistaken for a complete one
    tmp = out.with_suffix(".part")
    decompressor = bz2.BZ2Decompressor() if decompress_bz2 else None
    try:
        with session.get(url, stream=True, timeout=timeout, **kwargs) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(decompressor.decompress(chunk) if decompressor else chunk)
        if decompressor and not decompressor.eof:
            raise OSError(f"Truncated bzip2 download from {url}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(out)


def exists(url):
    """Whether a URL can be fetched (by a HEAD request)."""
    return session.head(url, timeout=30).ok


def find_latest_run(is_published, run_hours, max_age, description):
    """Most recent run (naive UTC datetime, on a `run_hours` cycle) for which `is_published(run)` is true,
    looking back up to `max_age`."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    run = now.replace(hour=now.hour - now.hour % run_hours, minute=0, second=0, microsecond=0)
    while now - run <= max_age:
        if is_published(run):
            return run
        run -= timedelta(hours=run_hours)
    raise RuntimeError(f"No {description} in the last {max_age}")
