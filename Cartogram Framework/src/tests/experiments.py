import os
import sys
import time
import numpy as np
import shapely
from collections import defaultdict
from shapely.geometry import Polygon as ShapelyPolygon

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.cartograms import *
from src.utils.common import *
from src.tests import *
from src.core import *


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_evaluable_regions(data, target_area_key="target_area"):
    dropped = [s for s in data if data[s][target_area_key] == 0]
    if dropped:
        print(f"Excluding {len(dropped)} placeholder region(s) from scoring: {dropped}")
    return {s: v for s, v in data.items() if v[target_area_key] != 0}, dropped

def _vertex_coincident_pairs(polygon_list, tol=1e-6):
    polygon_list = [_to_polygon(p) for p in polygon_list]   # <-- added
    buckets = defaultdict(set)
    for idx, poly in enumerate(polygon_list):
        for (x, y) in poly.exterior.coords:
            key = (round(x / tol), round(y / tol))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    buckets[(key[0] + dx, key[1] + dy)].add(idx)

    pairs = set()
    for members in buckets.values():
        members = list(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(frozenset((members[i], members[j])))
    return pairs


def _touching_pairs(polygon_list, tol=0.0):
    polygon_list = [_to_polygon(p) for p in polygon_list]   # <-- added
    n = len(polygon_list)
    buffered = [p.buffer(tol) if tol else p for p in polygon_list]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if buffered[i].intersects(buffered[j]):
                pairs.add(frozenset((i, j)))
    return pairs


def neighbouring_pairs_(polygon_list, mode="touching", tol=1e-6):
    if mode == "vertex":
        return _vertex_coincident_pairs(polygon_list, tol=tol)
    elif mode == "touching":
        return _touching_pairs(polygon_list, tol=tol)
    raise ValueError(f"Unknown adjacency mode: {mode!r}")

def _to_polygon(obj):

    if hasattr(obj, "exterior"):
        poly = obj
    elif hasattr(obj, "geoms"):
        # already a MultiPolygon/GeometryCollection (e.g. from upstream repair)
        poly = obj
    else:
        poly = ShapelyPolygon(obj)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.geom_type != "Polygon":
        # buffer(0) (or the input itself) produced multiple disjoint parts —
        # keep the largest by area rather than silently losing the shape.
        parts = list(poly.geoms)
        poly = max(parts, key=lambda p: p.area)

    return poly


def _region_polygons(mapping, key):
    return {state: _to_polygon(mapping[state][key]) for state in mapping.keys()}


def _adjacency_pairs(polygons, tol=0.0):
    states = list(polygons.keys())
    pairs = set()
    for i in range(len(states)):
        pi = polygons[states[i]]
        pi_buf = pi.buffer(tol) if tol else pi
        for j in range(i + 1, len(states)):
            pj = polygons[states[j]]
            pj_buf = pj.buffer(tol) if tol else pj
            if pi_buf.intersects(pj_buf):
                pairs.add(frozenset((states[i], states[j])))
    return pairs


def neighbouring_state_pairs(org_data, map_key="polygon", tol=0.0):
    map_polygons = _region_polygons(org_data, map_key)
    return _adjacency_pairs(map_polygons, tol=tol)


# ---------------------------------------------------------------------------
# 4.4.4 Topological Accuracy: Adjacency Error (already present)
# ---------------------------------------------------------------------------

def topological_accuracy(cartogram_polygons, map_polygons, cartogram_mode="touching"):
    if cartogram_mode == "touching":
        Ec = set(neighbouring_pairs(cartogram_polygons))
        Em = set(neighbouring_pairs(map_polygons))
    else:
        Ec = set(neighbouring_pairs_(cartogram_polygons, mode=cartogram_mode))
        Em = set(neighbouring_pairs_(map_polygons, mode="touching"))

    return 1 - len(Ec & Em) / len(Ec | Em)


# Clear alias matching the thesis' own naming (Sec. 4.4.4).
adjacency_error = topological_accuracy


# ---------------------------------------------------------------------------
# 4.4.1 Statistical Accuracy: Cartographic Error
# ---------------------------------------------------------------------------

def cartographic_errors(data, new_area_key="new_area", target_area_key="target_area"):
    errors = []
    for state in data.keys():
        na, ta = data[state][new_area_key], data[state][target_area_key]
        denom = max(na, ta)
        errors.append(abs(na - ta) / denom if denom > 0 else 0.0)    
    return errors


def max_relative_area_error(data, new_area_key="new_area", target_area_key="target_area"):
    errors = []
    for state in data.keys():
        a = data[state][new_area_key]
        a_star = data[state][target_area_key]
        errors.append(abs(a / a_star - 1))
    return errors


def success_rate(data, org_data=None, new_area_key="new_area", target_area_key="target_area",
                  original_area_key="original_area", zero_change_tol=1e-9):
    if org_data is None:
        org_data = data
    rates = []
    for state in data.keys():
        source = data[state] if original_area_key in data[state] else org_data[state]
        a0 = source[original_area_key]
        a = data[state][new_area_key]
        a_star = data[state][target_area_key]
        denom = abs(a_star - a0)
        # print(f"target_area: {a_star}")
        # print(f"original_area: {a0}")
        # print(f"denom: {denom}")
        # print(f"numerator: {abs(a - a0)}")
        if denom < zero_change_tol * max(abs(a0), abs(a_star), 1e-12):
            continue
        rates.append(abs(a - a0) / denom)
    return rates


# ---------------------------------------------------------------------------
# 4.4.2 Geographical Accuracy: Shape Preservation
# ---------------------------------------------------------------------------

def _normalize_unit_area(poly_coords):
    poly = _to_polygon(poly_coords)
    area = poly.area
    if area <= 0:
        raise ValueError("Broken Polygon: negative or zero area")
    cx, cy = poly.centroid.x, poly.centroid.y
    scale = 1.0 / np.sqrt(area)
    coords = np.asarray(poly.exterior.coords)
    coords = (coords - [cx, cy]) * scale
    return ShapelyPolygon(coords)


def hamming_distance(data, org_data, map_key="polygon", cartogram_key="new_polygon"):
    errors = []
    for state in data.keys():
        original_poly = org_data[state][map_key]
        new_poly = data[state][cartogram_key]
        p_orig = _normalize_unit_area(original_poly)
        p_new = _normalize_unit_area(new_poly)
        sym_diff_area = p_orig.symmetric_difference(p_new).area
        errors.append(sym_diff_area / 2.0)
    return errors


def frechet_hausdorff_distances(data, org_data, map_key="polygon", cartogram_key="new_polygon"):
    frechet, hausdorff = [], []
    for state in data.keys():
        p_orig = _to_polygon(org_data[state][map_key])
        p_new = _to_polygon(data[state][cartogram_key])
        # Module-level functions (not geometry methods) for portability across
        # Shapely 2.x point releases.
        frechet.append(shapely.frechet_distance(p_orig, p_new))
        hausdorff.append(shapely.hausdorff_distance(p_orig, p_new))
    return frechet, hausdorff


# ---------------------------------------------------------------------------
# 4.4.3 Geographical Accuracy: Positional Fidelity
# ---------------------------------------------------------------------------

def angular_orientation_error(data, org_data, map_key="polygon", cartogram_key="new_polygon",
                               pairs=None):
    states = list(data.keys())
    orig_centroids = {s: _to_polygon(org_data[s][map_key]).centroid for s in states}
    new_centroids = {s: _to_polygon(data[s][cartogram_key]).centroid for s in states}

    if pairs is None:
        pairs = [frozenset((states[i], states[j]))
                  for i in range(len(states)) for j in range(i + 1, len(states))]

    errors = []
    for pair in pairs:
        i, j = tuple(pair)
        orig_angle = np.arctan2(orig_centroids[j].y - orig_centroids[i].y,
                                 orig_centroids[j].x - orig_centroids[i].x)
        new_angle = np.arctan2(new_centroids[j].y - new_centroids[i].y,
                                new_centroids[j].x - new_centroids[i].x)
        diff = abs(orig_angle - new_angle) % (2 * np.pi)
        diff = min(diff, 2 * np.pi - diff)  # reduce to [0, pi]
        errors.append(diff)
    return errors


def orthogonal_direction_error(data, org_data, map_key="polygon", cartogram_key="new_polygon",
                                pairs=None):
    states = list(data.keys())
    orig_centroids = {s: _to_polygon(org_data[s][map_key]).centroid for s in states}
    new_centroids = {s: _to_polygon(data[s][cartogram_key]).centroid for s in states}

    if pairs is None:
        pairs = [frozenset((states[i], states[j]))
                  for i in range(len(states)) for j in range(i + 1, len(states))]

    if not pairs:
        return 0.0

    flips = 0
    for pair in pairs:
        i, j = tuple(pair)
        dx_orig = orig_centroids[j].x - orig_centroids[i].x
        dy_orig = orig_centroids[j].y - orig_centroids[i].y
        dx_new = new_centroids[j].x - new_centroids[i].x
        dy_new = new_centroids[j].y - new_centroids[i].y
        if np.sign(dx_orig) != np.sign(dx_new) or np.sign(dy_orig) != np.sign(dy_new):
            flips += 1
    return flips / len(pairs)


# ---------------------------------------------------------------------------
# 4.4.5 Topological Integrity: Overlap and Self-Intersection
# ---------------------------------------------------------------------------

def overlap_integrity(data, cartogram_key="new_polygon"):
    states = list(data.keys())
    raw_polys, repaired_polys = {}, {}
    invalid_count = 0
    for s in states:
        raw = data[s][cartogram_key]
        raw_poly = raw if hasattr(raw, "exterior") else ShapelyPolygon(raw)
        if not raw_poly.is_valid:
            invalid_count += 1
        raw_polys[s] = raw_poly
        repaired_polys[s] = raw_poly if raw_poly.is_valid else raw_poly.buffer(0)

    total_overlap_area = 0.0
    overlap_pair_count = 0
    for i in range(len(states)):
        pi = repaired_polys[states[i]]
        for j in range(i + 1, len(states)):
            pj = repaired_polys[states[j]]
            inter_area = pi.intersection(pj).area
            if inter_area > 0:
                overlap_pair_count += 1
                total_overlap_area += inter_area

    total_area = sum(p.area for p in repaired_polys.values())
    overlap_fraction = total_overlap_area / total_area if total_area > 0 else 0.0

    return {
        "invalid_count": invalid_count,
        "overlap_pair_count": overlap_pair_count,
        "overlap_fraction": overlap_fraction,
    }


# ---------------------------------------------------------------------------
# 4.4.6 Topological Integrity: Gap Error
# ---------------------------------------------------------------------------

def gap_error(data, org_data=None, map_key="polygon", cartogram_key="new_polygon", tol=1e-9):
    if org_data is None:
        org_data = data

    map_polygons = _region_polygons(org_data, map_key)
    cartogram_polygons = _region_polygons(data, cartogram_key)
    adjacency = _adjacency_pairs(map_polygons, tol=tol)

    total_gap_area = 0.0
    for pair in adjacency:
        i, j = tuple(pair)
        pi, pj = cartogram_polygons[i], cartogram_polygons[j]
        hull_area = pi.union(pj).convex_hull.area
        gap = hull_area - (pi.area + pj.area)
        if gap > 0:
            total_gap_area += gap

    total_map_area = sum(p.area for p in map_polygons.values())
    gap_fraction = total_gap_area / total_map_area if total_map_area > 0 else 0.0

    return {"gap_area": total_gap_area, "gap_fraction": gap_fraction}


# ---------------------------------------------------------------------------
# 4.4.7 Complexity and Runtime
# ---------------------------------------------------------------------------

def complexity_metrics(data, cartogram_key="new_polygon"):
    counts = []
    for state in data.keys():
        poly = _to_polygon(data[state][cartogram_key])
        counts.append(len(poly.exterior.coords) - 1)  # closed ring: last == first
    return {
        "mean_vertex_count": float(np.mean(counts)),
        "max_vertex_count": int(np.max(counts)),
    }


def time_cartogram_run(solve_fn, *args, **kwargs):
    start = time.perf_counter()
    result = solve_fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def quality_criteria(data, org_data=None, stat_data=None, new_area_key="new_area", map_key="polygon",
                      target_area_key="target_area", cartogram_key="new_polygon",
                      original_area_key="original_area", restrict_to_adjacent=False, adjacency_tol=0.0, cartogram_mode="touching"):
    errors = {}
    if not org_data:
        org_data = data
    if not stat_data:
        stat_data = data

    # 4.4.1 Statistical Accuracy: Cartographic Error
    carto_errors = cartographic_errors(data, new_area_key=new_area_key, target_area_key=target_area_key)
    errors["Mean cartographic error"] = np.mean(carto_errors)
    errors["Max cartographic error"] = np.max(carto_errors)
    # max_rel_errors = max_relative_area_error(data, new_area_key=new_area_key, target_area_key=target_area_key)
    # errors["Emax (max relative area error)"] = np.max(max_rel_errors)

    try:
        sr = success_rate(data, org_data, new_area_key=new_area_key, target_area_key=target_area_key,
                           original_area_key=original_area_key)
        if sr:
            errors["Mean success rate SR(N)"] = np.mean(sr)
            # errors["Max success rate SR(N)"] = np.max(sr)
    except KeyError:
        # original_area not available for this dataset; skip SR(N) rather than fail the whole report.
        pass
    # 4.4.4 Topological Accuracy: Adjacency Error
    cartogram_polygons = [data[stat][cartogram_key] for stat in data.keys()]
    map_polygons = [org_data[stat][map_key] for stat in org_data.keys()]
    errors["Topological accuracy"] = topological_accuracy(cartogram_polygons, map_polygons, cartogram_mode=cartogram_mode)
    # 4.4.2 Geographical Accuracy: Shape Preservation
    hamming = hamming_distance(data, org_data, map_key=map_key, cartogram_key=cartogram_key)
    errors["Mean Hamming distance"] = np.mean(hamming)
    errors["Max Hamming distance"] = np.max(hamming)
    # frechet, hausdorff = frechet_hausdorff_distances(data, org_data, map_key=map_key, cartogram_key=cartogram_key)
    # errors["Mean Frechet distance"] = np.mean(frechet)
    # errors["Max Frechet distance"] = np.max(frechet)
    # errors["Mean Hausdorff distance"] = np.mean(hausdorff)
    # errors["Max Hausdorff distance"] = np.max(hausdorff)
    # 4.4.3 Geographical Accuracy: Positional Fidelity
    adjacency_pairs = None
    if restrict_to_adjacent:
        adjacency_pairs = neighbouring_state_pairs(org_data, map_key=map_key, tol=adjacency_tol)

    angular_errors = angular_orientation_error(data, org_data, map_key=map_key, cartogram_key=cartogram_key,
                                                pairs=adjacency_pairs)
    errors["Mean angular orientation error"] = np.mean(angular_errors)

    errors["Orthogonal direction error"] = orthogonal_direction_error(
        data, org_data, map_key=map_key, cartogram_key=cartogram_key, pairs=adjacency_pairs
    )

    # 4.4.5 Topological Integrity: Overlap and Self-Intersection
    overlap = overlap_integrity(data, cartogram_key=cartogram_key)
    errors["Invalid (self-intersecting) polygon count"] = overlap["invalid_count"]
    # errors["Overlapping pair count"] = overlap["overlap_pair_count"]
    errors["Overlap area fraction"] = overlap["overlap_fraction"]

    # 4.4.6 Topological Integrity: Gap Error
    gaps = gap_error(data, org_data=org_data, map_key=map_key, cartogram_key=cartogram_key)
    errors["Gap area fraction"] = gaps["gap_fraction"]
    # 4.4.7 Complexity (runtime is measured externally, see time_cartogram_run)
    complexity = complexity_metrics(data, cartogram_key=cartogram_key)
    errors["Mean vertex count"] = complexity["mean_vertex_count"]
    # errors["Max vertex count"] = complexity["max_vertex_count"]
    return errors
