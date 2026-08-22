import os
import sys
import time
import numpy as np
import shapely

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.cartograms import *
from src.utils.common import *
from src.tests import *
from src.core import *


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_polygon(obj):
    """Coerce `obj` into a valid Shapely Polygon.
    ...
    """
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
    """Return {state: Polygon} for every state in `mapping`."""
    return {state: _to_polygon(mapping[state][key]) for state in mapping.keys()}


def _adjacency_pairs(polygons, tol=0.0):
    """Build the set T of geographically neighbouring state pairs.

    Two regions are considered adjacent if their polygons intersect
    (optionally after a small buffer `tol`, useful for near-touching
    geometries after floating point noise). Returned as a set of
    frozenset({state_i, state_j}) pairs, restricted to i != j.
    """
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
    """Public helper: adjacency set T over the *original map* geometry.

    Used to restrict positional-fidelity metrics (4.4.3) to O(n) pairs on
    large maps such as the World dataset, instead of the full O(n^2) set.
    """
    map_polygons = _region_polygons(org_data, map_key)
    return _adjacency_pairs(map_polygons, tol=tol)


# ---------------------------------------------------------------------------
# 4.4.4 Topological Accuracy: Adjacency Error (already present)
# ---------------------------------------------------------------------------

def topological_accuracy(cartogram_polygons, map_polygons):
    """Adjacency error tau, eq. (4.7): 1 - |Ec ∩ Em| / |Ec ∪ Em|."""
    Ec = set(neighbouring_pairs(cartogram_polygons))
    Em = set(neighbouring_pairs(map_polygons))

    return 1 - len(Ec & Em) / len(Ec | Em)


# Clear alias matching the thesis' own naming (Sec. 4.4.4).
adjacency_error = topological_accuracy


# ---------------------------------------------------------------------------
# 4.4.1 Statistical Accuracy: Cartographic Error
# ---------------------------------------------------------------------------

def cartographic_errors(data, new_area_key="new_area", target_area_key="target_area"):
    """Per-region cartographic error epsilon_v, eq. (4.1)."""
    errors = []
    for state in data.keys():
        errors.append(abs(data[state][new_area_key] - data[state][target_area_key])/max(data[state][new_area_key], data[state][target_area_key]))
    return errors


def max_relative_area_error(data, new_area_key="new_area", target_area_key="target_area"):
    """E_max, eq. (4.2): unbounded, asymmetric, sensitive to over-scaling."""
    errors = []
    for state in data.keys():
        a = data[state][new_area_key]
        a_star = data[state][target_area_key]
        errors.append(abs(a / a_star - 1))
    return errors


def success_rate(data, org_data=None, new_area_key="new_area", target_area_key="target_area",
                  original_area_key="original_area"):
    """SR(N), eq. (4.3) with the editorial-note correction applied:

        SR_i = |A_i - A_i^(0)| / |A_i^* - A_i^(0)|

    i.e. how much of the intended area change was actually achieved
    (0 = no progress toward the target, 1 = target reached exactly; values
    above 1 indicate overshoot). Regions whose target change is ~0 (A_i^* ~
    A_i^(0)) are skipped to avoid dividing by zero, since "success" is
    undefined when no change was intended.
    """
    if org_data is None:
        org_data = data
    rates = []
    for state in data.keys():
        source = data[state] if original_area_key in data[state] else org_data[state]
        a0 = source[original_area_key]
        a = data[state][new_area_key]
        a_star = data[state][target_area_key]
        denom = abs(a_star - a0)
        if denom == 0:
            continue
        rates.append(abs(a - a0) / denom)
    return rates


# ---------------------------------------------------------------------------
# 4.4.2 Geographical Accuracy: Shape Preservation
# ---------------------------------------------------------------------------

def _normalize_unit_area(poly_coords):
    """Center on centroid and rescale to unit area, per Nusrat et al. Sec 4."""
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
    """Hamming distance delta_v, eq. (4.4). Normalized to [0, 1], poolable
    across datasets of different absolute size."""
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
    """Discrete Frechet and Hausdorff boundary distances (Sec. 4.4.2).

    Unlike the area-based Hamming distance, these compare the map and
    cartogram polygon *boundaries* directly, so they catch localised
    boundary spikes (vertex penetration, runaway scale) that an
    area-aggregated score can dilute. Reported in raw map units, so they
    should be aggregated per dataset rather than pooled across datasets of
    very different absolute size.
    """
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
    """Angular orientation error theta_ij, eq. (4.5).

    `pairs`: optional iterable of frozenset({state_i, state_j}) restricting
    the comparison to adjacent pairs (see `neighbouring_state_pairs`), used
    for large maps to keep the comparison O(n) instead of O(n^2).
    Defaults to the full pairwise set.
    """
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
    """Orthogonal (relative-direction) error rho, eq. (4.6).

    Fraction of region pairs whose cardinal (N/S, E/W) ordering flips
    between map and cartogram - a coarser, more interpretable companion to
    the angular error above.
    """
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
    """Overlap / self-intersection check (Sec. 4.4.5), eq. (4.8).

    Returns a dict with:
      - "invalid_count": number of self-intersecting (invalid) cartogram
        polygons, checked *before* repair via buffer(0)
      - "overlap_pair_count": number of region pairs whose (repaired)
        polygons have a positive-area intersection
      - "overlap_fraction": Phi_overlap, the summed pairwise overlap area
        as a fraction of total map area
    """
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
    """Gap metric (Sec. 4.4.6): previously-adjacent regions pulling apart.

    For every pair adjacent in the *original map*, the gap contribution is
    the area of the convex hull of the pair's union minus the summed area
    of the two (repaired) polygons - a positive value if a visible gap has
    opened between regions that used to share a border, ~0 if they still
    touch or overlap. Total gap area is reported both raw and as a fraction
    of total map area, mirroring `overlap_integrity`.
    """
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
    """Mean/max polygon vertex count per region (Sec. 4.4.7).

    Post-processing steps such as gap-closing and overlap resolution can
    inflate polygon complexity, so this is tracked separately from quality.
    Runtime itself is not computed here since it depends on wall-clock
    timing around the solver call; see `time_cartogram_run` below for a
    thin convenience wrapper.
    """
    counts = []
    for state in data.keys():
        poly = _to_polygon(data[state][cartogram_key])
        counts.append(len(poly.exterior.coords) - 1)  # closed ring: last == first
    return {
        "mean_vertex_count": float(np.mean(counts)),
        "max_vertex_count": int(np.max(counts)),
    }


def time_cartogram_run(solve_fn, *args, **kwargs):
    """Convenience wrapper reporting wall-clock runtime of a solver call,
    matching the per-type / per-dataset runtimes reported in Table 4.1.
    Data loading should happen before this call, per the comparative
    protocol in Sec. 4.3 ("Runtime was measured separately from data
    loading to final polygon output.").
    """
    start = time.perf_counter()
    result = solve_fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def quality_criteria(data, org_data=None, stat_data=None, new_area_key="new_area", map_key="polygon",
                      target_area_key="target_area", cartogram_key="new_polygon",
                      original_area_key="original_area", restrict_to_adjacent=False, adjacency_tol=0.0):
    """Compute the full quality-criteria report from Sec. 4.4.

    `restrict_to_adjacent`: if True, angular/orthogonal positional-fidelity
    metrics are computed only over map-adjacent pairs (the O(n) variant
    Sec. 4.4.3 recommends for large maps such as the World dataset) rather
    than all O(n^2) pairs.
    """
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
    errors["Topological accuracy"] = topological_accuracy(cartogram_polygons, map_polygons)
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
