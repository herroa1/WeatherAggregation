"""Writes the map outlines the website draws over its maps, from Natural Earth (public domain).

- `uk_coastline.json`: the coastline, for the UK maps
- `london_outline.json`: the London boroughs and the Thames, for the London maps

Each is {"lon": [...], "lat": [...]} with None between lines (as Plotly draws them), plus "river" for
London. They're committed, so builds don't need cartopy's downloads; re-run this
(`python -m website.outlines`) only to change them. Needs cartopy, which downloads Natural Earth on
first use.
"""

import json
from pathlib import Path

import cartopy.io.shapereader as shp
from shapely.geometry import box

from .build import LONDON_EXTENT, UK_EXTENT

HERE = Path(__file__).parent


def _lines(geometries, extent, tolerance, min_length=0.0):
    """Lines (and polygon edges) clipped to extent, as Plotly lon/lat lists with None between lines."""
    lon_min, lon_max, lat_min, lat_max = extent
    clip = box(lon_min, lat_min, lon_max, lat_max)
    lons, lats = [], []
    for geometry in geometries:
        if not geometry.intersects(clip):
            continue
        geometry = geometry.boundary if geometry.geom_type.endswith("Polygon") else geometry
        geometry = geometry.intersection(clip).simplify(tolerance)
        for part in getattr(geometry, "geoms", [geometry]):
            if part.is_empty or part.length < min_length:
                continue
            xs, ys = part.xy
            lons += [round(x, 4) for x in xs] + [None]
            lats += [round(y, 4) for y in ys] + [None]
    return lons, lats


def write():
    coast = shp.Reader(shp.natural_earth("10m", "physical", "coastline")).geometries()
    lon, lat = _lines(coast, UK_EXTENT, tolerance=0.02, min_length=0.05)
    (HERE / "uk_coastline.json").write_text(json.dumps(
        {"lon": lon, "lat": lat, "source": "Natural Earth 1:10m coastline (public domain)"}, separators=(",", ":")))

    regions = shp.Reader(shp.natural_earth("10m", "cultural", "admin_1_states_provinces")).records()
    uk = [r.geometry for r in regions if r.attributes.get("admin") == "United Kingdom"]
    lon, lat = _lines(uk, LONDON_EXTENT, tolerance=0.002)
    rivers = shp.Reader(shp.natural_earth("10m", "physical", "rivers_lake_centerlines")).records()
    river_lon, river_lat = _lines([r.geometry for r in rivers if r.attributes.get("name") == "Thames"],
                                  LONDON_EXTENT, tolerance=0.002)
    (HERE / "london_outline.json").write_text(json.dumps(
        {"lon": lon, "lat": lat, "river": {"lon": river_lon, "lat": river_lat},
         "source": "Natural Earth 1:10m admin-1 boundaries and rivers (public domain)"}, separators=(",", ":")))


if __name__ == "__main__":
    write()
