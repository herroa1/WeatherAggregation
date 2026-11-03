"""Data loaders, one module per data source.

The Met Office UK model is the benchmark the other sources are compared against.
"""

from .dwd import ICONEULoader
from .ecmwf import ECMWFLoader
from .gfs import GFSLoader
from .met_office import MetOfficeGlobalLoader, MetOfficeLoader
from .msc import GDPSLoader

BENCHMARK = MetOfficeLoader.name


def forecast_sources(data_dir="data", ecmwf_source="ecmwf"):
    """Every non-benchmark source, keyed by display name. Each has the attributes and methods listed in
    `loaders.standard`.

    `ecmwf_source="aws"` reads ECMWF from its AWS mirror, which keeps old runs (the ECMWF portal
    only keeps a few days).
    """
    loaders = [
        MetOfficeGlobalLoader(data_dir=data_dir),
        ECMWFLoader(data_dir=data_dir, model="ifs", source=ecmwf_source),
        ECMWFLoader(data_dir=data_dir, model="aifs-single", source=ecmwf_source),
        GFSLoader(data_dir=data_dir, model="gfs"),
        GFSLoader(data_dir=data_dir, model="aigfs"),
        ICONEULoader(data_dir=data_dir),
        GDPSLoader(data_dir=data_dir),
    ]
    return {loader.name: loader for loader in loaders}


__all__ = ["BENCHMARK", "ECMWFLoader", "GDPSLoader", "GFSLoader", "ICONEULoader", "MetOfficeGlobalLoader",
           "MetOfficeLoader", "forecast_sources"]
