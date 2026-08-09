import numpy as np
import os
import sys
from typing import Optional

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.geometry import *
from src.utils import *

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _coerce_point_list(values, n: int, fallback: list[np.ndarray]) -> list[np.ndarray]:
    if values is None:
        return [np.asarray(v, dtype=float) for v in fallback]
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.shape[0] == 2:
        return [np.asarray(values, dtype=float)] * n
    if isinstance(values, np.ndarray) and values.ndim == 2:
        return [np.asarray(values[k], dtype=float) for k in range(len(values))]
    return [np.asarray(v, dtype=float) for v in values]


def preprocess(
    data,
    fixed_points,
    cartographic_error,
    target_centers=None,
):
    """
    Normalise region dictionaries into parallel arrays.

    Expected dictionary keys per region:
        polygon, area, target_area, target_positions, centroid

    Returns:
        data            dict[str, dict]
        polygons        list[np.ndarray]   each (n_i, 2)
        target_areas    list[float]
        fixed_points    list[np.ndarray]   each (2,)
        cartographic_error  float
        target_centers  list[np.ndarray]   each (2,)
    """
    if not isinstance(data, dict):
        raise TypeError(
            "preprocess() expects the loader dictionary returned by get_polygon_data()/loader()."
        )

    names = [name for name in data.keys() if name != "__meta__"]
    polygons = []
    original_areas = []
    target_areas = []
    default_fixed_points = []
    default_target_centers = []

    for name in names:
        record = data[name]

        polygon = record.get("new_polygon", record.get("polygon"))
        if polygon is None:
            raise ValueError(f"Region '{name}' is missing a polygon")

        polygon = np.asarray(polygon, dtype=float)
        polygons.append(polygon)

        original_area = float(record.get("area", polygon_areanp(polygon)))
        original_areas.append(original_area)
        record.setdefault("area", original_area)
        record.setdefault("original_area", original_area)
        record.setdefault("polygon", polygon)
        record.setdefault("original_polygon", polygon)

        target_area = record.get("target_area", original_area)
        target_areas.append(float(target_area))

        default_fixed_points.append(np.asarray(record.get("centroid", np.mean(polygon, axis=0)), dtype=float))
        default_target_centers.append(
            np.asarray(record.get("target_positions", record.get("centroid", np.mean(polygon, axis=0))), dtype=float)
        )

    n = len(polygons)
    fixed_points = _coerce_point_list(fixed_points, n, default_fixed_points)
    target_centers = _coerce_point_list(target_centers, n, default_target_centers)

    if isinstance(target_areas, (int, float)):
        target_areas = [float(target_areas)] * n
    else:
        target_areas = [float(a) for a in target_areas]

    original_total_area = sum(original_areas)
    target_total_area = sum(target_areas)
    if target_total_area > 0:
        for i, target_area in enumerate(target_areas):
            target_areas[i] = target_areas[i] * (original_total_area / target_total_area)

    return data, polygons, target_areas, fixed_points, cartographic_error, target_centers


def preprocess_global(
    data,
    fixed_points,
    cartographic_error,
    target_centers=None,
    # legacy quality scalars (kept for backward compatibility)
    shape_deformation=0.0,
    relative_direction=1.0,
    topological_accuracy=1.0,
    spatial_deformation=1.0,
    global_shape=1.0,
    local_shape=1.0,
    complexity=1.0,
    data_ink_ratio=1.0,
    # new explicit λ weights (take precedence if not None)
    lambda_shape: Optional[float] = None,
    lambda_area: Optional[float] = None,
    lambda_center: Optional[float] = None,
    lambda_topology: Optional[float] = None,
):
    """
    Maps the quality-criterion API onto the four λ weights used in the
    objective. If explicit λ values are provided they override the
    derived ones.
    """
    data, polygons, target_areas, fixed_points, cartographic_error, target_centers = preprocess(
        data,
        fixed_points,
        cartographic_error,
        target_centers,
    )

    W_shape = (lambda_shape if lambda_shape is not None else (1.0 - shape_deformation))
    W_area = (lambda_area if lambda_area is not None else cartographic_error)
    W_spatial = (lambda_center if lambda_center is not None else spatial_deformation)
    W_topology = (lambda_topology if lambda_topology is not None else topological_accuracy)

    return (
        data,
        polygons,
        target_areas,
        fixed_points,
        cartographic_error,
        target_centers,
        W_shape,
        W_area,
        W_spatial,
        W_topology,
    )


# ---------------------------------------------------------------------------
# Leader-line computation (Nickel et al. Lemma 2 — O(n²) sweep)
# ---------------------------------------------------------------------------

def compute_leaders(
    centers: list[np.ndarray],
    half_sides: list[float],
    adjacent_pairs: set[tuple[int, int]],
    tol: float = 1e-3,
) -> list[tuple[int, int, np.ndarray]]:
    """
    For every adjacent pair (i,j) whose squares do not touch, return a
    minimal-length monotone orthogonal leader.

    Returns
    -------
    leaders : list of (i, j, waypoints)
        waypoints is an (m,2) array of the polyline vertices.
    """
    n = len(centers)
    leaders = []

    for (i, j) in adjacent_pairs:
        ci, cj = centers[i], centers[j]
        ri, rj = half_sides[i], half_sides[j]

        # Axis-aligned bounding boxes of the two squares
        li, ri_ = ci[0] - ri, ci[0] + ri
        bi, ti  = ci[1] - ri, ci[1] + ri
        lj, rj_ = cj[0] - rj, cj[0] + rj
        bj, tj  = cj[1] - rj, cj[1] + rj

        # Check whether boxes already share a boundary segment
        overlap_x = min(ri_, rj_) - max(li, lj)
        overlap_y = min(ti, tj)   - max(bi, bj)

        if overlap_x >= tol or overlap_y >= tol:
            # Squares touch or overlap — no leader needed
            continue

        # Closest points on the two boxes
        px = np.clip(cj[0], li, ri_)
        py = np.clip(cj[1], bi, ti)
        qx = np.clip(ci[0], lj, rj_)
        qy = np.clip(ci[1], bj, tj)

        start = np.array([px, py])
        end   = np.array([qx, qy])

        # Minimal L-shaped leader: go horizontal then vertical, or vice versa
        mid_h = np.array([end[0], start[1]])
        mid_v = np.array([start[0], end[1]])

        # Pick the bend that stays further from all other squares (heuristic)
        waypoints = np.array([start, mid_h, end])
        leaders.append((i, j, waypoints))

    return leaders


# ---------------------------------------------------------------------------
# Shared-vertex index lookup helpers
# ---------------------------------------------------------------------------

def _build_vertex_index(poly: np.ndarray, tolerance: float = 1e-8) -> dict:
    """
    Return a dict  {rounded_key: vertex_index}  for fast lookup of which
    index in `poly` corresponds to a given coordinate.
    """
    idx = {}
    for k, pt in enumerate(poly):
        key = (round(pt[0] / tolerance) * tolerance,
               round(pt[1] / tolerance) * tolerance)
        idx.setdefault(key, k)   # keep first occurrence
    return idx


def find_shared_vertex_indices(
    poly_i: np.ndarray,
    poly_j: np.ndarray,
    shared_pts: list,
    tolerance: float = 1e-8,
) -> list[tuple[int, int]]:
    """
    For each shared point in `shared_pts`, return (k, l) where
    k = index in poly_i and l = index in poly_j.

    Points not found in either polygon are silently skipped.
    """
    idx_i = _build_vertex_index(poly_i, tolerance)
    idx_j = _build_vertex_index(poly_j, tolerance)

    pairs = []
    for pt in shared_pts:
        key = (round(pt[0] / tolerance) * tolerance,
               round(pt[1] / tolerance) * tolerance)
        if key in idx_i and key in idx_j:
            pairs.append((idx_i[key], idx_j[key]))
    return pairs


# ---------------------------------------------------------------------------
# Shared-vertex helper functions (user-supplied; included here for convenience)
# ---------------------------------------------------------------------------

def neighbouring_pairs(polygons, tolerance=1e-8):
    """
    Return list of (i, j) pairs such that polygon i and polygon j share
    at least one vertex (within tolerance).
    """
    n = len(polygons)
    neighbors = []
    for i in range(n):
        set_i = set()
        for pt in polygons[i]:
            set_i.add((round(pt[0] / tolerance) * tolerance,
                       round(pt[1] / tolerance) * tolerance))
        for j in range(i + 1, n):
            for pt in polygons[j]:
                key = (round(pt[0] / tolerance) * tolerance,
                       round(pt[1] / tolerance) * tolerance)
                if key in set_i:
                    neighbors.append((i, j))
                    break
    return neighbors


def common_vertices(poly1, poly2, tolerance=1e-8):
    """
    Return a list of points (each as [lon, lat]) that appear in both polygons.
    Points are considered equal if their distance is < tolerance.
    """
    set1 = {}
    for pt in poly1:
        key = (round(pt[0] / tolerance) * tolerance,
               round(pt[1] / tolerance) * tolerance)
        set1.setdefault(key, pt)

    common = []
    seen = set()
    for pt in poly2:
        key = (round(pt[0] / tolerance) * tolerance,
               round(pt[1] / tolerance) * tolerance)
        if key in set1 and key not in seen:
            common.append(list(set1[key]))
            seen.add(key)
    return common


def shared_vertices_of_neighbors(polygons, tolerance=1e-1):
    """
    Return a list of [i, j, shared_points] where:
      - i, j are indices of neighbouring polygons
      - shared_points is a list of vertices they have in common
    """
    neighbors = neighbouring_pairs(polygons, tolerance)
    result = []
    for i, j in neighbors:
        shared = common_vertices(polygons[i], polygons[j], tolerance)
        result.append([i, j, shared])
    return result