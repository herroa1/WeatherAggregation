"""Runs the aggregate-forecasts notebook end to end, so loader changes that break it are noticed.

Opt-in: `pytest -m notebook`. It fetches the latest runs, so a new Met Office run counts towards the
DataHub allowance. The notebook is run in memory; its saved outputs aren't changed.
"""

import nbformat
import pytest
from nbclient import NotebookClient

from conftest import ROOT

pytestmark = pytest.mark.notebook

NOTEBOOK = ROOT / "latestForecasts" / "aggregateForecasts.ipynb"


def test_aggregate_forecasts_notebook_runs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
