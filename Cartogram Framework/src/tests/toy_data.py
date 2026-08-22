"""
toy_data.py

A small, fully synthetic test map for CartogramFramework: an n x n grid
of unit squares. It exists to make the framework's mechanics easy to see
and cheap to iterate on, as a companion to your real (China / US) data:

  - Every shared border is an EXACT floating-point match (grid coordinates),
    so neighbouring_pairs() / shared_vertices_of_neighbors() / the
    contiguous-mode shared-vertex coupling all get a clean, unambiguous
    signal -- no near-miss-tolerance edge cases to debug.
  - Target areas follow a checkerboard pattern (every other cell doubles,
    the rest halve), so a correct solve is visually obvious: the map
    should look like a checkerboard of big and small squares, still tiled
    edge-to-edge with no gaps or overlaps. Wrong output is equally obvious
    (a checkerboard that still looks like a uniform grid = area term did
    nothing; visible gaps/overlaps = contiguity term or postprocessing
    broke).
  - 16 regions solves in well under a second, so this is what you want
    for iterating on lambda_* values or on the postprocessing/recommender
    code, before spending minutes re-solving the real 34-province map.

Two ways to use it
-------------------
1. In-memory, bypassing your file-based loader() entirely (recommended
   for quick iteration -- guaranteed to work regardless of your loader's
   file-discovery convention, since it hands CartogramFramework_global
   the same (data, polygons, target_areas, target_positions, names,
   centroid) tuple your loader() normally returns):

    from toy_data import build_toy_grid
    data, polygons, target_areas, target_positions, names, centroid = build_toy_grid()

    # then exactly the same as contiguous.py from here:
    horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)
    new_data = CartogramFramework_global(
        data=data, fixed_points=None, shape="contiguous2",
        target_centers=target_positions,
        horizontal_pairs=horizontal_pairs, vertical_pairs=vertical_pairs,
        neighboring_pairs=neighbouring_pairs(polygons),
        shared_vertices_of_neighbors=shared_vertices_of_neighbors(polygons),
        lambda_shape=0.5, lambda_area=1.0, lambda_contiguous=30.0,
        soft_mean_scale=True, lambda_mean_scale=1e5, t_min=0.01, t_max=8.0,
        lambda_vertex_distance=3.0, vertex_distance_margin=1.5,
        postprocess_disputed_pixels=True, postprocess_gaps=True,
        postprocess_equalize_areas=True,
    )

   Or, if you've already got parameter_recommender.py, even simpler:

    from parameter_recommender import recommend_params
    from toy_data import run_toy_cartogram
    params, trace = recommend_params(cartogram_type="contiguous", neighborhoods_kept=1.0)
    data = run_toy_cartogram(params)   # builds the grid AND solves it

2. On disk, in your real loader()'s file format, if you want to test the
   actual file-based loading path (fuzzy name matching, CSV parsing,
   etc.) rather than bypassing it:

    from toy_data import write_toy_files
    write_toy_files("toy_grid")   # writes toy_grid.geojson + toy_grid.csv
                                    # in the same ADM1_EN/Entity/gdp schema
                                    # as states.geojson/states.csv

   Property/column names match your real China dataset exactly
   (ADM1_EN/ADM0_EN/ADM1_PCODE in the GeoJSON, Entity/Code/Year/gdp in
   the CSV) so if your loader(region, level, year) matches on those
   fields, pointing it at these files should work without modification
   -- just place them wherever your loader's file-discovery convention
   expects region="toygrid", level="cell" (or whatever your loader uses)
   to resolve to.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def _cell_polygon(r: int, c: int, cell_size: float) -> np.ndarray:
    """Unit (or cell_size) square for grid cell (row r, col c), CCW, matching
    the winding convention polygon_areanp() expects (positive area)."""
    x0, y0 = c * cell_size, r * cell_size
    x1, y1 = x0 + cell_size, y0 + cell_size
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def build_toy_grid(
    nrows: int = 4,
    ncols: int = 4,
    cell_size: float = 1.0,
    high: float = 2.0,
    low: float = 0.5,
    pattern: str = "checkerboard",
):
    """
    Build an (nrows x ncols) grid of unit squares with a checkerboard
    area-target pattern, in the same tuple shape your loader() returns:

        data, polygons, target_areas, target_positions, names, centroid

    `high`/`low` are target areas as a MULTIPLE of each cell's own
    original area (cell_size**2) -- e.g. the defaults make every other
    cell want to double, the rest want to halve. `preprocess()` will
    sum-normalise these against the total original area regardless, so
    the exact multiple mostly controls the *contrast* between neighbours,
    not the absolute scale.

    pattern="checkerboard" (default) alternates high/low by (r+c) parity.
    pattern="uniform" gives every cell the same target = its own original
    area (a no-op cartogram -- useful as a "does the solver reproduce the
    identity map" sanity check).
    pattern="gradient" ramps target area smoothly from `low` in one
    corner to `high` in the opposite corner -- useful for checking smooth
    area transitions don't produce shape artefacts.
    """
    cell_area = cell_size ** 2
    names = []
    polygons = []
    target_areas = []
    target_positions = []

    for r in range(nrows):
        for c in range(ncols):
            name = f"R{r}C{c}"
            poly = _cell_polygon(r, c, cell_size)
            centroid_xy = poly.mean(axis=0)

            if pattern == "checkerboard":
                mult = high if (r + c) % 2 == 0 else low
            elif pattern == "uniform":
                mult = 1.0
            elif pattern == "gradient":
                t = ((r / max(nrows - 1, 1)) + (c / max(ncols - 1, 1))) / 2.0
                mult = low + t * (high - low)
            else:
                raise ValueError(f"Unknown pattern {pattern!r}")

            names.append(name)
            polygons.append(poly)
            target_areas.append(cell_area * mult)
            target_positions.append(centroid_xy)

    all_xy = np.vstack(polygons)
    centroid = all_xy.mean(axis=0)

    data = {}
    for name, poly, ta, tc in zip(names, polygons, target_areas, target_positions):
        data[name] = {
            "polygon": poly,
            "area": float(cell_area),
            "original_area": float(cell_area),
            "target_area": float(ta),
            "target_positions": tc,
            "centroid": tc,
        }
    data["__meta__"] = {"source": "toy_data.build_toy_grid", "pattern": pattern}

    return data, polygons, target_areas, target_positions, names, centroid


def run_toy_cartogram(params: dict, nrows: int = 4, ncols: int = 4, **grid_kwargs):
    """
    Build the toy grid and solve it with a given parameter dict (e.g. from
    parameter_recommender.recommend_params()). Needs this project's src.*
    modules on the path, same as build_cartogram() in
    parameter_recommender.py.
    """
    from src.core import (   # noqa: local import, only needed here
        CartogramFramework_global,
        neighbouring_pairs,
        shared_vertices_of_neighbors,
        disjoint_pairs_horizontal_and_vertical_center,
    )

    data, polygons, target_areas, target_positions, names, centroid = build_toy_grid(
        nrows=nrows, ncols=ncols, **grid_kwargs
    )
    horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)

    params = dict(params)  # don't mutate caller's dict
    params.pop("area_scale", None)
    params.pop("area_scale_cap", None)

    return CartogramFramework_global(
        data=data,
        fixed_points=None,
        target_centers=target_positions,
        horizontal_pairs=horizontal_pairs,
        vertical_pairs=vertical_pairs,
        neighboring_pairs=neighbouring_pairs(polygons),
        shared_vertices_of_neighbors=shared_vertices_of_neighbors(polygons),
        **params,
    )


# ---------------------------------------------------------------------------
# File-based variant — same property/column schema as states.geojson /
# states.csv, for testing the real file-based loader() path.
# ---------------------------------------------------------------------------

def write_toy_files(
    path_prefix: str,
    nrows: int = 4,
    ncols: int = 4,
    cell_size: float = 1.0,
    high: float = 2.0,
    low: float = 0.5,
    pattern: str = "checkerboard",
    year: int = 2024,
):
    """
    Write `{path_prefix}.geojson` and `{path_prefix}.csv`, schema-matched
    to states.geojson / states.csv (ADM1_EN/ADM0_EN/ADM1_PCODE properties;
    Entity/Code/Year/gdp columns) so your existing loader() should be able
    to read them the same way it reads the real China data, once placed
    wherever your loader's region/level file-discovery convention expects.

    `gdp` is written as target_area x 1000 (arbitrary units) rather than
    the raw target_area, since real GDP-style value columns are what your
    loader is presumably built to convert into target areas — check
    your loader's area-conversion step if the resulting map doesn't match
    build_toy_grid()'s in-memory target_areas exactly; the two are only
    guaranteed to agree if your loader's value->area mapping is linear
    with no extra normalisation beyond preprocess()'s own sum-matching.
    """
    _, polygons, target_areas, _, names, _ = build_toy_grid(
        nrows=nrows, ncols=ncols, cell_size=cell_size,
        high=high, low=low, pattern=pattern,
    )

    features = []
    for name, poly in zip(names, polygons):
        ring = [[float(x), float(y)] for x, y in poly] + [[float(poly[0][0]), float(poly[0][1])]]
        features.append({
            "type": "Feature",
            "properties": {
                "ADM1_EN": name,
                "ADM1_ZH": name,
                "ADM1_PCODE": f"TOY{name}",
                "ADM0_EN": "ToyGrid",
                "ADM0_ZH": "ToyGrid",
                "ADM0_PCODE": "TOY",
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[ring]],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(f"{path_prefix}.geojson", "w") as f:
        json.dump(geojson, f)

    lines = ["Entity,Code,Year,gdp"]
    for name, ta in zip(names, target_areas):
        lines.append(f"{name},,{year},{ta * 1000:.4f}")
    with open(f"{path_prefix}.csv", "w") as f:
        f.write("\n".join(lines) + "\n")

    return f"{path_prefix}.geojson", f"{path_prefix}.csv"


# ---------------------------------------------------------------------------
# Demo / sanity check — runs standalone (no repo needed) to verify grid
# construction, adjacency, and the checkerboard target-area pattern.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def polygon_area(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    data, polygons, target_areas, target_positions, names, centroid = build_toy_grid()

    print(f"{len(names)} cells, map centroid at {centroid}")
    print(f"{'cell':<6} {'orig_area':>10} {'target_area':>12} {'ratio':>8}")
    for name in names:
        rec = data[name]
        ratio = rec["target_area"] / rec["area"]
        print(f"{name:<6} {rec['area']:>10.2f} {rec['target_area']:>12.2f} {ratio:>8.2f}")

    # Sanity check: every polygon should have positive (CCW) area.
    bad = [n for n, p in zip(names, polygons) if polygon_area(p) <= 0]
    print(f"\nNon-positive-area polygons (should be empty): {bad}")

    # Sanity check: interior cells should each share an edge (2 vertices)
    # with exactly their von Neumann neighbours.
    from collections import defaultdict
    vertex_owners = defaultdict(set)
    for name, poly in zip(names, polygons):
        for pt in poly:
            vertex_owners[(round(pt[0], 8), round(pt[1], 8))].add(name)
    shared_counts = defaultdict(int)
    for owners in vertex_owners.values():
        if len(owners) > 1:
            for a in owners:
                for b in owners:
                    if a < b:
                        shared_counts[(a, b)] += 1
    n_pairs = len(shared_counts)
    n_edges_shared = sum(1 for v in shared_counts.values() if v >= 2)
    print(f"Region pairs sharing >=1 vertex: {n_pairs}")
    print(f"Region pairs sharing a full edge (>=2 vertices): {n_edges_shared}")

    print("\nWriting toy_grid.geojson / toy_grid.csv ...")
    paths = write_toy_files("toy_grid")
    print("Wrote:", paths)
