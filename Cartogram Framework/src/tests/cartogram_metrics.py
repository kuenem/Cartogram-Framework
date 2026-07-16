"""
cartogram_metrics.py

Quantitative evaluation metrics for cartograms, following the definitions in:

  - Alam, Kobourov & Veeramoni (2015), "Quantitative Measures for Cartogram
    Generation Techniques", CGF 34(3) -> cartographic error (statistical accuracy)
  - Nusrat, Alam & Kobourov (2015), "Evaluating Cartogram Effectiveness"
    (arXiv:1504.02218) -> Hamming distance, angular orientation error,
    adjacency error (geographical + topological accuracy)
  - Heilmann, Keim, Panse & Sips (2004), "RecMap: Rectangular Map
    Approximations" -> angular orientation error definition

All polygon inputs are plain (n, 2) numpy arrays of vertex coordinates
(exterior ring only), matching the output of get_polygon_data() /
loader() in your existing pipeline. No dependency on your src/ package,
so this can be dropped in and imported independently.
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry


# --------------------------------------------------------------------------
# 1. Statistical accuracy: cartographic error
# --------------------------------------------------------------------------

def cartographic_error(
    measured_areas: Dict[str, float],
    target_areas: Dict[str, float],
) -> Tuple[float, float, Dict[str, float]]:
    """Average (eps) and maximum (xi) cartographic error.

    eps_v = |o(v) - w(v)| / max(o(v), w(v))   for each region v

    This matches your existing implementation exactly -- kept here so the
    whole evaluation suite can be called from one module. `measured_areas`
    and `target_areas` are dicts keyed by region name/id.
    """
    errors: Dict[str, float] = {}
    for region_id, target in target_areas.items():
        measured = measured_areas.get(region_id)
        if measured is None:
            continue
        denom = max(measured, target)
        errors[region_id] = 0.0 if denom == 0 else abs(measured - target) / denom

    if not errors:
        raise ValueError("No overlapping region ids between measured_areas and target_areas.")

    eps = sum(errors.values()) / len(errors)
    xi = max(errors.values())
    return eps, xi, errors


# --------------------------------------------------------------------------
# 2. Geographical accuracy: shape (Hamming distance) + position (angle)
# --------------------------------------------------------------------------

def _normalize_unit_area(poly_coords: np.ndarray) -> ShapelyPolygon:
    """Center on centroid and rescale to unit area, per Nusrat et al. Sec 4."""
    poly = ShapelyPolygon(poly_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)  # cheap self-intersection fix
    area = poly.area
    if area <= 0:
        raise ValueError("Degenerate polygon with non-positive area.")
    cx, cy = poly.centroid.x, poly.centroid.y
    scale = 1.0 / np.sqrt(area)
    coords = np.asarray(poly.exterior.coords)
    coords = (coords - [cx, cy]) * scale
    return ShapelyPolygon(coords)


def hamming_distance(original_poly: np.ndarray, new_poly: np.ndarray) -> float:
    """Shape-deformation distance for a single region, in [0, 1].

    Both polygons are centered on their own centroid and rescaled to unit
    area, then superimposed. delta = area(symmetric difference) / 2, since
    both normalized shapes have area 1 (so the "area in exactly one of the
    two polygons" is naturally on a [0, 2] scale before dividing by 2).
    """
    p_orig = _normalize_unit_area(original_poly)
    p_new = _normalize_unit_area(new_poly)
    sym_diff_area = p_orig.symmetric_difference(p_new).area
    return sym_diff_area / 2.0


def mean_hamming_distance(
    original_polygons: Sequence[np.ndarray],
    new_polygons: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Average Hamming distance (shape deformation) over all regions."""
    if names is None:
        names = [str(i) for i in range(len(original_polygons))]
    per_region = {}
    for name, p_o, p_n in zip(names, original_polygons, new_polygons):
        per_region[name] = hamming_distance(p_o, p_n)
    avg = sum(per_region.values()) / len(per_region)
    return avg, per_region


def _pair_angle(c_from: np.ndarray, c_to: np.ndarray) -> float:
    """Angle (radians, in (-pi, pi]) of the vector from c_from to c_to."""
    d = c_to - c_from
    return np.arctan2(d[1], d[0])


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two angles, in [0, pi]."""
    diff = abs(a - b) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)


def angular_orientation_error(
    original_centroids: Sequence[np.ndarray],
    new_centroids: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    pairs: Optional[Iterable[Tuple[int, int]]] = None,
    degrees: bool = True,
) -> Tuple[float, Dict[Tuple[str, str], float]]:
    """Angular orientation error theta (Heilmann et al. 2004).

    For every pair of regions (i, j), compares the slope of the centroid-to-
    centroid line in the original map vs. the cartogram, and averages the
    absolute change.

    original_centroids / new_centroids: sequences of (x, y) points, same
    order and same length as `names`. These require the ORIGINAL geojson
    (pre-optimization) centroids -- e.g. the `centroid` list returned by
    get_polygon_data() -- alongside the post-optimization centroids
    (m_i = c_i + delta_i, or simply np.mean(new_polygon, axis=0)).

    pairs: optional iterable of index pairs to restrict the comparison to
    (e.g. only neighbouring_pairs(polygons), which is O(n) instead of the
    full O(n^2) all-pairs comparison -- use this for large maps like
    the world dataset).
    """
    n = len(original_centroids)
    if names is None:
        names = [str(i) for i in range(n)]
    if pairs is None:
        pairs = combinations(range(n), 2)

    orig = [np.asarray(c, dtype=float) for c in original_centroids]
    new = [np.asarray(c, dtype=float) for c in new_centroids]

    per_pair = {}
    for i, j in pairs:
        if i == j:
            continue
        theta_orig = _pair_angle(orig[i], orig[j])
        theta_new = _pair_angle(new[i], new[j])
        d = _angle_diff(theta_orig, theta_new)
        per_pair[(names[i], names[j])] = np.degrees(d) if degrees else d

    if not per_pair:
        raise ValueError("No region pairs to compare -- check `pairs` argument.")

    avg = sum(per_pair.values()) / len(per_pair)
    return avg, per_pair


# --------------------------------------------------------------------------
# 3. Topological accuracy: adjacency error
# --------------------------------------------------------------------------

def build_adjacency(
    polygons: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    tolerance: float = 0.0,
) -> set:
    """Adjacency edge set E, built from actual polygon geometry.

    Two regions are adjacent if their polygons touch or overlap within
    `tolerance` (buffer distance in the same units as the coordinates).
    tolerance=0.0 requires exact touching (shared boundary/vertex), which
    is correct for contiguous cartograms and the original map. For
    Dorling / non-contiguous cartograms you will want tolerance > 0, since
    regions rarely touch exactly -- pick a tolerance relative to typical
    region size (e.g. a small fraction of the median region "radius").

    Returns a set of frozenset({name_i, name_j}) pairs.
    """
    if names is None:
        names = [str(i) for i in range(len(polygons))]

    shapely_polys = []
    for p in polygons:
        poly = ShapelyPolygon(p)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if tolerance > 0:
            poly = poly.buffer(tolerance)
        shapely_polys.append(poly)

    edges = set()
    for (i, pi), (j, pj) in combinations(enumerate(shapely_polys), 2):
        if pi.intersects(pj):
            edges.add(frozenset((names[i], names[j])))
    return edges


def adjacency_error(
    original_polygons: Sequence[np.ndarray],
    new_polygons: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    original_tolerance: float = 0.0,
    new_tolerance: float = 0.0,
) -> Tuple[float, set, set]:
    """Adjacency error tau = 1 - |Ec ∩ Em| / |Ec ∪ Em|.

    Em = adjacency graph of the ORIGINAL map (needs the original geojson
    polygons -- this is the extra argument you were missing).
    Ec = adjacency graph of the cartogram (new_polygons).

    For contiguous cartograms tau should come out at (or very near) 0 by
    construction, since shared vertices are hard-constrained equal --
    this is a good sanity check that your framework is doing what it
    claims. For Dorling / non-contiguous, tune `new_tolerance` upward
    since exact touching is not expected/desired there, and instead
    reflects "close enough to read as adjacent."
    """
    if names is None:
        names = [str(i) for i in range(len(original_polygons))]

    e_m = build_adjacency(original_polygons, names, tolerance=original_tolerance)
    e_c = build_adjacency(new_polygons, names, tolerance=new_tolerance)

    union = e_m | e_c
    if not union:
        return 0.0, e_c, e_m
    tau = 1.0 - len(e_c & e_m) / len(union)
    return tau, e_c, e_m


# --------------------------------------------------------------------------
# 4. Convenience wrapper: run everything at once
# --------------------------------------------------------------------------

def evaluate_cartogram(
    original_polygons: Sequence[np.ndarray],
    new_polygons: Sequence[np.ndarray],
    original_centroids: Sequence[np.ndarray],
    new_centroids: Sequence[np.ndarray],
    measured_areas: Dict[str, float],
    target_areas: Dict[str, float],
    names: Optional[Sequence[str]] = None,
    angle_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    adjacency_original_tolerance: float = 0.0,
    adjacency_new_tolerance: float = 0.0,
) -> Dict[str, object]:
    """Run all four metric families and return one report dict.

    Pass `angle_pairs=neighbouring_pairs(polygons)` (your existing helper)
    for large maps (e.g. world) to avoid the O(n^2) all-pairs cost.
    """
    if names is None:
        names = [str(i) for i in range(len(original_polygons))]

    eps, xi, err_per_region = cartographic_error(measured_areas, target_areas)
    hamming_avg, hamming_per_region = mean_hamming_distance(
        original_polygons, new_polygons, names
    )
    theta_avg, theta_per_pair = angular_orientation_error(
        original_centroids, new_centroids, names, pairs=angle_pairs
    )
    tau, e_c, e_m = adjacency_error(
        original_polygons,
        new_polygons,
        names,
        original_tolerance=adjacency_original_tolerance,
        new_tolerance=adjacency_new_tolerance,
    )

    return {
        "cartographic_error_avg": eps,
        "cartographic_error_max": xi,
        "cartographic_error_per_region": err_per_region,
        "hamming_distance_avg": hamming_avg,
        "hamming_distance_per_region": hamming_per_region,
        "angular_orientation_error_avg_deg": theta_avg,
        "angular_orientation_error_per_pair_deg": theta_per_pair,
        "adjacency_error": tau,
        "adjacency_cartogram_edges": e_c,
        "adjacency_original_edges": e_m,
    }
