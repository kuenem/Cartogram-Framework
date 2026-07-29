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
from collections.abc import Mapping
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely import frechet_distance
from shapely.geometry import LineString, Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover - older shapely fallback
    make_valid = None


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
    if _is_region_record_mapping(original_polygons) and _is_region_record_mapping(new_polygons):
        return evaluate_cartogram_records(  # type: ignore[arg-type]
            original_polygons,
            new_polygons,
            names=names,
            angle_pairs=angle_pairs,
            adjacency_original_tolerance=adjacency_original_tolerance,
            adjacency_new_tolerance=adjacency_new_tolerance,
        )

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


# --------------------------------------------------------------------------
# 5. Dict-native helpers and extended metrics
# --------------------------------------------------------------------------

def _is_region_record_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(v, Mapping) for v in value.values() if v is not None
    )


def _ordered_region_ids(
    data: Mapping[str, Mapping[str, Any]],
    names: Optional[Sequence[str]] = None,
) -> List[str]:
    if names is not None:
        return [str(name) for name in names]
    return [str(name) for name in data.keys() if name != "__meta__"]


def _coerce_region_polygon(record: Mapping[str, Any], key: str, fallback_keys: Sequence[str]) -> np.ndarray:
    for candidate in (key, *fallback_keys):
        if candidate in record and record[candidate] is not None:
            return np.asarray(record[candidate], dtype=float)
    raise KeyError(f"Region record does not contain any polygon key from {([key, *fallback_keys])}")


def _coerce_region_centroid(
    record: Mapping[str, Any],
    centroid_key: str,
    polygon_key: str,
) -> np.ndarray:
    if centroid_key in record and record[centroid_key] is not None:
        return np.asarray(record[centroid_key], dtype=float)
    polygon = _coerce_region_polygon(record, polygon_key, ("new_polygon", "polygon"))
    return np.mean(polygon, axis=0)


def _record_scalar_map(
    data: Mapping[str, Mapping[str, Any]],
    key: str,
    names: Optional[Sequence[str]] = None,
    fallback_keys: Sequence[str] = (),
) -> Dict[str, float]:
    region_ids = _ordered_region_ids(data, names)
    values: Dict[str, float] = {}
    for region_id in region_ids:
        record = data[region_id]
        value = None
        for candidate in (key, *fallback_keys):
            if candidate in record and record[candidate] is not None:
                value = record[candidate]
                break
        if value is None:
            raise KeyError(f"Region '{region_id}' is missing scalar key '{key}'")
        values[region_id] = float(value)
    return values


def _record_polygons(
    data: Mapping[str, Mapping[str, Any]],
    key: str,
    names: Optional[Sequence[str]] = None,
    fallback_keys: Sequence[str] = (),
) -> Tuple[List[str], List[np.ndarray]]:
    region_ids = _ordered_region_ids(data, names)
    polygons: List[np.ndarray] = []
    for region_id in region_ids:
        record = data[region_id]
        polygons.append(_coerce_region_polygon(record, key, fallback_keys))
    return region_ids, polygons


def _record_centroids(
    data: Mapping[str, Mapping[str, Any]],
    centroid_key: str,
    polygon_key: str,
    names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    region_ids = _ordered_region_ids(data, names)
    centroids: List[np.ndarray] = []
    for region_id in region_ids:
        record = data[region_id]
        centroids.append(_coerce_region_centroid(record, centroid_key, polygon_key))
    return region_ids, centroids


def _ensure_valid_polygon(poly: ShapelyPolygon) -> ShapelyPolygon:
    if poly.is_valid:
        return poly
    if make_valid is not None:
        valid = make_valid(poly)
        if valid.geom_type == "Polygon":
            return valid
        if valid.geom_type == "MultiPolygon":
            return max(valid.geoms, key=lambda geom: geom.area)
    return poly.buffer(0)


def _polygon_from_coords(coords: Any) -> ShapelyPolygon:
    poly = ShapelyPolygon(np.asarray(coords, dtype=float))
    return _ensure_valid_polygon(poly)


def _ring_coords(poly: Any) -> np.ndarray:
    polygon = _polygon_from_coords(poly)
    coords = np.asarray(polygon.exterior.coords, dtype=float)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def _polygon_vertex_count(poly: Any) -> int:
    return int(len(_ring_coords(poly)))


def max_relative_area_error(
    measured_areas: Mapping[str, float],
    target_areas: Mapping[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Maximum relative area error E_max = max_i |A_i^cart / A_i^target - 1|."""
    per_region: Dict[str, float] = {}
    for region_id, target in target_areas.items():
        measured = measured_areas.get(region_id)
        if measured is None:
            continue
        if target == 0:
            per_region[region_id] = 0.0 if measured == 0 else float("inf")
        else:
            per_region[region_id] = abs(measured / target - 1.0)

    if not per_region:
        raise ValueError("No overlapping region ids between measured_areas and target_areas.")

    return max(per_region.values()), per_region


def success_rate(
    original_areas: Mapping[str, float],
    measured_areas: Mapping[str, float],
    target_areas: Mapping[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Region-wise success ratio |o(v)-a(v)| / |w(v)-a(v)| from the design doc.

    A value of 1.0 means the target area was hit exactly. If the desired
    change is zero, the score falls back to 1.0 when the region stays at its
    original area and 0.0 otherwise.
    """
    per_region: Dict[str, float] = {}
    for region_id, target in target_areas.items():
        original = original_areas.get(region_id)
        measured = measured_areas.get(region_id)
        if original is None or measured is None:
            continue
        denom = abs(target - original)
        if denom == 0:
            per_region[region_id] = 1.0 if abs(measured - original) == 0 else 0.0
        else:
            per_region[region_id] = abs(measured - original) / denom

    if not per_region:
        raise ValueError("No overlapping region ids between the three area mappings.")

    return sum(per_region.values()) / len(per_region), per_region


def orthogonal_orientation_error(
    original_centroids: Sequence[np.ndarray],
    new_centroids: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    pairs: Optional[Iterable[Tuple[int, int]]] = None,
) -> Tuple[float, Dict[Tuple[str, str], float]]:
    """Fraction of region pairs whose N-S/E-W ordering flips."""
    n = len(original_centroids)
    if names is None:
        names = [str(i) for i in range(n)]
    if pairs is None:
        pairs = combinations(range(n), 2)

    orig = [np.asarray(c, dtype=float) for c in original_centroids]
    new = [np.asarray(c, dtype=float) for c in new_centroids]

    per_pair: Dict[Tuple[str, str], float] = {}
    for i, j in pairs:
        if i == j:
            continue
        dx_orig = np.sign(orig[j][0] - orig[i][0])
        dy_orig = np.sign(orig[j][1] - orig[i][1])
        dx_new = np.sign(new[j][0] - new[i][0])
        dy_new = np.sign(new[j][1] - new[i][1])

        flipped = False
        if dx_orig != 0 and dx_new != 0 and dx_orig != dx_new:
            flipped = True
        if dy_orig != 0 and dy_new != 0 and dy_orig != dy_new:
            flipped = True

        per_pair[(names[i], names[j])] = 1.0 if flipped else 0.0

    if not per_pair:
        raise ValueError("No region pairs to compare -- check `pairs` argument.")

    return sum(per_pair.values()) / len(per_pair), per_pair


def _turning_function(poly: Any, n_samples: int = 256, rotation_invariant: bool = True) -> np.ndarray:
    coords = _ring_coords(poly)
    if len(coords) < 3:
        raise ValueError("Turning-function metric requires polygons with at least 3 vertices.")

    closed = np.vstack([coords, coords[0]])
    edges = np.diff(np.vstack([closed, closed[0]]), axis=0)
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    turning = np.diff(np.unwrap(np.r_[edge_angles, edge_angles[0]]))
    cum_len = np.cumsum(np.r_[0.0, np.linalg.norm(edges, axis=1)])
    if cum_len[-1] == 0:
        raise ValueError("Degenerate polygon with zero perimeter.")

    s = cum_len[:-1] / cum_len[-1]
    angles = np.cumsum(np.r_[0.0, turning[:-1]])
    if rotation_invariant:
        angles = angles - angles[0]

    grid = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    samples = np.interp(grid, np.r_[s, 1.0], np.r_[angles, angles[0] + (0.0 if rotation_invariant else 2 * np.pi)])
    return samples


def turning_angle_distortion(
    original_polygons: Sequence[np.ndarray],
    new_polygons: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    n_samples: int = 256,
    rotation_invariant: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Approximate turning-angle distortion (Ψ / Ψ_M) from polygon rings."""
    if names is None:
        names = [str(i) for i in range(len(original_polygons))]

    per_region: Dict[str, float] = {}
    for name, original_poly, new_poly in zip(names, original_polygons, new_polygons):
        tf_original = _turning_function(original_poly, n_samples=n_samples, rotation_invariant=rotation_invariant)
        tf_new = _turning_function(new_poly, n_samples=n_samples, rotation_invariant=rotation_invariant)
        per_region[name] = float(np.mean(np.abs(tf_original - tf_new)) / (2 * np.pi))

    if not per_region:
        raise ValueError("No polygons supplied for turning-angle distortion.")

    return sum(per_region.values()) / len(per_region), per_region


def _boundary_linestring(poly: Any) -> LineString:
    coords = _ring_coords(poly)
    if len(coords) < 2:
        raise ValueError("Polygon boundary requires at least 2 coordinates.")
    return LineString(np.vstack([coords, coords[0]]))


def frechet_distance_metric(
    original_polygons: Sequence[np.ndarray],
    new_polygons: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Discrete Fréchet distance between polygon exteriors."""
    if names is None:
        names = [str(i) for i in range(len(original_polygons))]

    per_region: Dict[str, float] = {}
    for name, original_poly, new_poly in zip(names, original_polygons, new_polygons):
        per_region[name] = float(frechet_distance(_boundary_linestring(original_poly), _boundary_linestring(new_poly)))

    if not per_region:
        raise ValueError("No polygons supplied for Fréchet distance.")

    return sum(per_region.values()) / len(per_region), per_region


def hausdorff_distance_metric(
    original_polygons: Sequence[np.ndarray],
    new_polygons: Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Hausdorff distance between polygon exteriors."""
    if names is None:
        names = [str(i) for i in range(len(original_polygons))]

    per_region: Dict[str, float] = {}
    for name, original_poly, new_poly in zip(names, original_polygons, new_polygons):
        per_region[name] = float(_boundary_linestring(original_poly).hausdorff_distance(_boundary_linestring(new_poly)))

    if not per_region:
        raise ValueError("No polygons supplied for Hausdorff distance.")

    return sum(per_region.values()) / len(per_region), per_region


def topological_integrity(
    regions: Mapping[str, Mapping[str, Any]] | Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    polygon_key: str = "new_polygon",
    fallback_polygon_keys: Sequence[str] = ("polygon",),
) -> Dict[str, Any]:
    """Overlap / self-intersection summary from the design document.

    For dict input, each region record may provide either `new_polygon` or
    `polygon`. For array input, pass the polygons directly.
    """
    if _is_region_record_mapping(regions):
        region_ids, polygons = _record_polygons(regions, polygon_key, names, fallback_polygon_keys)
    else:
        region_ids = [str(i) for i in range(len(regions))] if names is None else [str(name) for name in names]
        polygons = [np.asarray(poly, dtype=float) for poly in regions]

    fixed = {
        region_id: _polygon_from_coords(poly)
        for region_id, poly in zip(region_ids, polygons)
    }

    self_intersections = 0
    for poly in fixed.values():
        if not poly.is_valid:
            self_intersections += 1

    overlap_count = 0
    overlap_area = 0.0
    total_area = sum(poly.area for poly in fixed.values())

    for (id_a, poly_a), (id_b, poly_b) in combinations(fixed.items(), 2):
        if not poly_a.intersects(poly_b):
            continue
        inter = poly_a.intersection(poly_b)
        if inter.area > 1e-9:
            overlap_count += 1
            overlap_area += inter.area

    return {
        "self_intersections": self_intersections,
        "overlap_count": overlap_count,
        "overlap_area_fraction": overlap_area / total_area if total_area else 0.0,
    }


def build_width_proxy(
    target_areas: Mapping[str, float],
    pairs: Iterable[Tuple[int, int]],
    names: Optional[Sequence[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """Build the pairwise width proxy w_ij = 0.5(√A_i + √A_j)."""
    region_ids = list(target_areas.keys()) if names is None else [str(name) for name in names]
    proxy: Dict[Tuple[str, str], float] = {}
    for i, j in pairs:
        key = (region_ids[i], region_ids[j])
        proxy[key] = 0.5 * (np.sqrt(float(target_areas[region_ids[i]])) + np.sqrt(float(target_areas[region_ids[j]])))
    return proxy


def gap_error(
    regions: Mapping[str, Mapping[str, Any]] | Sequence[np.ndarray],
    adjacency_T: Iterable[Tuple[int, int]],
    width_proxy: Mapping[Tuple[str, str], float],
    names: Optional[Sequence[str]] = None,
    polygon_key: str = "new_polygon",
    fallback_polygon_keys: Sequence[str] = ("polygon",),
    gap_tol_fraction: float = 0.01,
) -> Dict[str, Any]:
    """Post-hoc gap metric from the design doc."""
    adjacency_pairs = list(adjacency_T)
    if _is_region_record_mapping(regions):
        region_ids, polygons = _record_polygons(regions, polygon_key, names, fallback_polygon_keys)
    else:
        region_ids = [str(i) for i in range(len(regions))] if names is None else [str(name) for name in names]
        polygons = [np.asarray(poly, dtype=float) for poly in regions]

    polygon_map = {region_id: _polygon_from_coords(poly) for region_id, poly in zip(region_ids, polygons)}
    violations = []
    for i, j in adjacency_pairs:
        region_i = region_ids[i]
        region_j = region_ids[j]
        key = (region_i, region_j)
        reverse_key = (region_j, region_i)
        width = width_proxy.get(key, width_proxy.get(reverse_key))
        if width is None:
            raise KeyError(f"Missing width proxy for pair {key}")
        dist = polygon_map[region_i].distance(polygon_map[region_j])
        tol = gap_tol_fraction * width
        if dist > tol:
            violations.append((region_i, region_j, dist, dist / width))

    n_pairs = len(adjacency_pairs)
    gap_count = len(violations)
    gap_mean_normalized = (sum(v[3] for v in violations) / n_pairs) if n_pairs else 0.0

    return {
        "gap_count": gap_count,
        "gap_rate": gap_count / n_pairs if n_pairs else 0.0,
        "gap_mean_normalized": gap_mean_normalized,
        "violations": violations,
    }


def polygon_complexity(
    polygons: Mapping[str, Mapping[str, Any]] | Sequence[np.ndarray],
    names: Optional[Sequence[str]] = None,
    polygon_key: str = "new_polygon",
    fallback_polygon_keys: Sequence[str] = ("polygon",),
) -> Dict[str, Any]:
    """Vertex-count summary for the complexity axis in the design doc."""
    if _is_region_record_mapping(polygons):
        region_ids, polygon_list = _record_polygons(polygons, polygon_key, names, fallback_polygon_keys)
    else:
        region_ids = [str(i) for i in range(len(polygons))] if names is None else [str(name) for name in names]
        polygon_list = [np.asarray(poly, dtype=float) for poly in polygons]

    per_region = {region_id: _polygon_vertex_count(poly) for region_id, poly in zip(region_ids, polygon_list)}
    counts = list(per_region.values())
    return {
        "mean_vertices": float(np.mean(counts)) if counts else 0.0,
        "max_vertices": int(max(counts)) if counts else 0,
        "per_region": per_region,
    }


def elapsed_seconds(start_time: float, end_time: float) -> float:
    """Small helper for runtime comparisons in the benchmark suite."""
    return float(end_time - start_time)


def _edges_to_index_pairs(
    edges: set, region_ids: Sequence[str]
) -> List[Tuple[int, int]]:
    """Convert a set of frozenset({name_i, name_j}) to list of (index_i, index_j)."""
    name_to_idx = {name: idx for idx, name in enumerate(region_ids)}
    pairs = []
    for edge in edges:
        a, b = tuple(edge)
        pairs.append((name_to_idx[a], name_to_idx[b]))
    return pairs


def evaluate_cartogram_records(
    original_data: Mapping[str, Mapping[str, Any]],
    cartogram_data: Mapping[str, Mapping[str, Any]],
    names: Optional[Sequence[str]] = None,
    angle_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    adjacency_original_tolerance: float = 0.0,
    adjacency_new_tolerance: float = 0.0,
    original_polygon_key: str = "polygon",
    cartogram_polygon_key: str = "new_polygon",
    original_centroid_key: str = "centroid",
    cartogram_centroid_key: str = "new_centroid",
    target_area_key: str = "target_area",
    original_area_key: str = "area",
    gap_adjacency_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    gap_width_proxy: Optional[Mapping[Tuple[str, str], float]] = None,
    gap_tol_fraction: float = 0.01,
    compute_gap_from_original_adjacency: bool = True,
) -> Dict[str, Any]:
    """Dict-native end-to-end report for the loader/cartogram record format.

    If `compute_gap_from_original_adjacency` is True (default) and `gap_adjacency_pairs`
    is not provided, the adjacency edges of the original map (as computed by
    `adjacency_error`) are used to generate the adjacency pairs for the gap metric.
    """
    region_ids = _ordered_region_ids(original_data, names)

    original_polygons = [
        _coerce_region_polygon(original_data[region_id], original_polygon_key, ("polygon",))
        for region_id in region_ids
    ]
    cartogram_polygons = [
        _coerce_region_polygon(cartogram_data.get(region_id, original_data[region_id]), cartogram_polygon_key, ("new_polygon", "polygon"))
        for region_id in region_ids
    ]
    original_centroids = [
        _coerce_region_centroid(original_data[region_id], original_centroid_key, original_polygon_key)
        for region_id in region_ids
    ]
    cartogram_centroids = [
        _coerce_region_centroid(cartogram_data.get(region_id, original_data[region_id]), cartogram_centroid_key, cartogram_polygon_key)
        for region_id in region_ids
    ]

    original_areas = _record_scalar_map(original_data, original_area_key, region_ids, ("original_area", "area"))
    target_areas = _record_scalar_map(original_data, target_area_key, region_ids, ("target_area", "area"))
    measured_areas = {region_id: float(_polygon_from_coords(poly).area) for region_id, poly in zip(region_ids, cartogram_polygons)}

    eps, xi, err_per_region = cartographic_error(measured_areas, target_areas)
    emax, emax_per_region = max_relative_area_error(measured_areas, target_areas)
    sr_avg, sr_per_region = success_rate(original_areas, measured_areas, target_areas)
    hamming_avg, hamming_per_region = mean_hamming_distance(original_polygons, cartogram_polygons, region_ids)
    theta_avg, theta_per_pair = angular_orientation_error(original_centroids, cartogram_centroids, region_ids, pairs=angle_pairs)
    rho_avg, rho_per_pair = orthogonal_orientation_error(original_centroids, cartogram_centroids, region_ids, pairs=angle_pairs)

    # Compute adjacency – needed for gap error if requested
    adjacency_tau, adjacency_cartogram_edges, adjacency_original_edges = adjacency_error(
        original_polygons,
        cartogram_polygons,
        region_ids,
        original_tolerance=adjacency_original_tolerance,
        new_tolerance=adjacency_new_tolerance,
    )

    # Prepare gap error automatically if requested
    if gap_adjacency_pairs is None and compute_gap_from_original_adjacency:
        if adjacency_original_edges:
            gap_adjacency_pairs = _edges_to_index_pairs(adjacency_original_edges, region_ids)
        else:
            gap_adjacency_pairs = []
    if gap_width_proxy is None and gap_adjacency_pairs is not None:
        gap_width_proxy = build_width_proxy(target_areas, gap_adjacency_pairs, region_ids)

    turning_avg, turning_per_region = turning_angle_distortion(original_polygons, cartogram_polygons, region_ids)
    frechet_avg, frechet_per_region = frechet_distance_metric(original_polygons, cartogram_polygons, region_ids)
    hausdorff_avg, hausdorff_per_region = hausdorff_distance_metric(original_polygons, cartogram_polygons, region_ids)
    topo = topological_integrity(cartogram_data, region_ids, polygon_key=cartogram_polygon_key)

    gap_report = None
    if gap_adjacency_pairs is not None and gap_width_proxy is not None:
        gap_report = gap_error(
            cartogram_data,
            gap_adjacency_pairs,
            gap_width_proxy,
            names=region_ids,
            polygon_key=cartogram_polygon_key,
            gap_tol_fraction=gap_tol_fraction,
        )

    complexity_report = polygon_complexity(cartogram_polygons, region_ids)

    return {
        "cartographic_error_avg": eps,
        "cartographic_error_max": xi,
        "cartographic_error_per_region": err_per_region,
        "max_relative_area_error_avg": emax,
        "max_relative_area_error_per_region": emax_per_region,
        "success_rate_avg": sr_avg,
        "success_rate_per_region": sr_per_region,
        "hamming_distance_avg": hamming_avg,
        "hamming_distance_per_region": hamming_per_region,
        "angular_orientation_error_avg_deg": theta_avg,
        "angular_orientation_error_per_pair_deg": theta_per_pair,
        "orthogonal_orientation_error_avg": rho_avg,
        "orthogonal_orientation_error_per_pair": rho_per_pair,
        "turning_angle_distortion_avg": turning_avg,
        "turning_angle_distortion_per_region": turning_per_region,
        "frechet_distance_avg": frechet_avg,
        "frechet_distance_per_region": frechet_per_region,
        "hausdorff_distance_avg": hausdorff_avg,
        "hausdorff_distance_per_region": hausdorff_per_region,
        "adjacency_error": adjacency_tau,
        "adjacency_cartogram_edges": adjacency_cartogram_edges,
        "adjacency_original_edges": adjacency_original_edges,
        "topological_integrity": topo,
        "gap_error": gap_report,
        "complexity": complexity_report,
    }


# --------------------------------------------------------------------------
# 6. One‑call convenience for your data format
# --------------------------------------------------------------------------

def evaluate_cartogram_from_data(
    data: Mapping[str, Mapping[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Evaluate all metrics from a single data dict that contains both original
    and cartogram fields.

    Expected keys per region:
        - 'original_polygon' (or fallback 'polygon') for original shape
        - 'new_polygon' for cartogram shape
        - 'centroid' for original centroid
        - 'new_centroid' for cartogram centroid
        - 'original_area' for original area
        - 'target_area' for desired area

    Additional keyword arguments are passed through to `evaluate_cartogram_records`.
    """
    return evaluate_cartogram_records(
        original_data=data,
        cartogram_data=data,
        original_polygon_key='original_polygon',
        cartogram_polygon_key='new_polygon',
        original_centroid_key='centroid',
        cartogram_centroid_key='new_centroid',
        target_area_key='target_area',
        original_area_key='original_area',
        **kwargs
    )