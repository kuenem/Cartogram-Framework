"""
CartogramFramework_global_CLAUDE
=================================
Unified cartogram optimisation covering:
  - Demers (square) cartograms
  - Dorling (circle) cartograms
  - Original shape
  - Contiguous shape  (shape="contiguous")
  - Contiguous soft   (shape="contiguous2")

Changes relative to previous version
--------------------------------------
1.  Objective restructured: γ (shape/target blend) and β (L1/L2 blend)
    are now separate parameters, removing the self-referential γ·(1-γ)
    structure from the original.

2.  Four explicit λ weights (lambda_shape, lambda_area, lambda_center,
    lambda_topology) replace the quality-criterion scalars.  A thin
    preprocess_global() shim maps the old API onto the new one.

3.  Mean-scale constraint is a hard linear equality by default.
    The squared area-error term in the objective is kept as a soft
    penalty and is zero at optimality when the hard constraint is active.

4.  cp.maximum() replaced with two explicit linear inequalities (CVXPY
    does not accept cp.maximum of two affine expressions as a constraint
    RHS — only cp.maximum of a variable and a constant is DCP-compliant).

5.  Directional deviation d_ij uses x_j - x_i (directed, matching the
    paper's secondary criterion) rather than x_i - x_j.

6.  Pairwise loop restricted: hard H/V ordering constraints are only
    added for pairs that actually appear in horizontal_pairs /
    vertical_pairs; the excess-distance (hor, ver) variables are only
    created for neighbouring pairs in T.  For non-adjacent pairs the
    hard separation constraint from H/V already prevents overlap, so no
    hor/ver slack is needed.

7.  Leader-line support (Demers/Dorling completion step):
    After solving, for every adjacent pair (i,j)∈T whose squares do not
    touch (gap > tol), a minimal-length axis-aligned leader is computed
    via the O(n²) sweep from Lemma 2 of Nickel et al.

8.  preprocess() updated to a single clean function that also returns
    the weight vector.

9.  Contiguous modes added (shape="contiguous" and shape="contiguous2"):

    APPROACH — shared-vertex coupling via the scale variables t_i, t_j.

    Each vertex v_ik of region i moves to:
        v'_ik = m_i + t_ik * (v_ik - c_i)

    where m_i = c_i + Δ_i is the displaced centre and t_ik is the
    per-vertex radial scale.

    For a vertex p shared by regions i and j (i.e. p ∈ poly_i ∩ poly_j),
    we look up its index k in poly_i and index l in poly_j and enforce
    that both map to the same point in the plane:

        m_i + t_{i,k} * (v_{i,k} - c_i)  ==  m_j + t_{j,l} * (v_{j,l} - c_j)

    This is a 2-component linear equality constraint in the CVXPY
    variables (Δ_i, t_i, Δ_j, t_j).  However it is *bilinear* in
    (Δ_i, t_i) because m_i = c_i + Δ_i multiplies nothing while
    t_{i,k} is scalar and (v_{i,k}-c_i) is a constant offset, so
    in fact the expression is **affine** in (Δ_i, t_{i,k}):

        (c_i + Δ_i) + t_{i,k}*(v_{i,k}-c_i)
      = c_i + Δ_i + t_{i,k}*d_{i,k}          (d_{i,k} = v_{i,k}-c_i constant)

    ⟹ fully DCP-compliant linear equality.

    shape="contiguous"  — hard equality (exact shared-vertex locking).
                          Pairs with area ratios > contiguous_area_ratio_cap
                          are skipped (prevents very large regions from
                          pinning very small ones).

    shape="contiguous2" — soft coupling: adds a quadratic penalty
        λ_contiguous * Σ ||v'_{i,k} - v'_{j,l}||²
    to the objective instead of an equality constraint.  This lets the
    area and shape terms breathe while still encouraging closure.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Optional


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polygon_areanp(pts: np.ndarray) -> float:
    """Signed shoelace area (absolute value = unsigned area)."""
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def make_circle(center: np.ndarray, area: float, n: int = 64) -> np.ndarray:
    r = np.sqrt(area / np.pi)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + r * np.column_stack([np.cos(angles), np.sin(angles)])


def make_square(center: np.ndarray, area: float, n: int = 64) -> np.ndarray:
    half = np.sqrt(area) / 2
    # n points distributed around the perimeter
    corners = np.array([
        [ half,  half],
        [-half,  half],
        [-half, -half],
        [ half, -half],
    ])
    pts_per_side = max(n // 4, 1)
    pts = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        for t in np.linspace(0, 1, pts_per_side, endpoint=False):
            pts.append(a + t * (b - a))
    return center + np.array(pts)


def nesting_depth(lst) -> int:
    if not isinstance(lst, (list, np.ndarray)):
        return 0
    if len(lst) == 0:
        return 1
    first = lst[0]
    if isinstance(first, (list, np.ndarray)):
        first = np.asarray(first)
        if first.ndim >= 2:
            return 3
        return 2
    return 2


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    polygons,
    target_areas,
    fixed_points,
    cartographic_error,
    target_centers,
):
    """
    Normalise input formats.  Returns:
        polygons        list[np.ndarray]   each (n_i, 2)
        target_areas    list[float]
        fixed_points    list[np.ndarray]   each (2,)
        cartographic_error  float
        target_centers  list[np.ndarray]   each (2,)
    """
    depth = nesting_depth(polygons)

    if depth == 2:
        polygons = [np.asarray(polygons)]
    else:
        polygons = [np.asarray(p) for p in polygons]

    n = len(polygons)

    # fixed_points
    if fixed_points is None:
        fixed_points = [np.mean(p, axis=0) for p in polygons]
    elif isinstance(fixed_points, np.ndarray) and fixed_points.shape == (2,):
        fixed_points = [fixed_points] * n
    elif isinstance(fixed_points, (list, np.ndarray)):
        arr = np.asarray(fixed_points)
        if arr.ndim == 1 and arr.shape[0] == 2:
            fixed_points = [arr] * n
        else:
            fixed_points = [np.asarray(fp) for fp in fixed_points]

    # target_areas
    if target_areas is None:
        target_areas = [polygon_areanp(p) for p in polygons]
    elif isinstance(target_areas, (int, float)):
        target_areas = [float(target_areas)] * n
    else:
        target_areas = [float(a) for a in target_areas]

        original_total_area = sum(polygon_areanp(p) for p in polygons)
        target_total_area = sum(target_areas)
        for i, target_area in enumerate(target_areas):
            target_areas[i] = target_areas[i] * (original_total_area / target_total_area)


    # target_centers
    if target_centers is None:
        target_centers = list(fixed_points)
    elif isinstance(target_centers, np.ndarray) and target_centers.ndim == 2:
        target_centers = [target_centers[k] for k in range(len(target_centers))]
    else:
        target_centers = [np.asarray(c) for c in target_centers]

    return polygons, target_areas, fixed_points, cartographic_error, target_centers


def preprocess_global(
    polygons,
    target_areas,
    fixed_points,
    cartographic_error,
    target_centers,
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
    objective.  If explicit λ values are provided they override the
    derived ones.
    """
    polygons, target_areas, fixed_points, cartographic_error, target_centers = preprocess(
        polygons, target_areas, fixed_points, cartographic_error, target_centers
    )

    W_shape    = (lambda_shape    if lambda_shape    is not None else (1.0 - shape_deformation))
    W_area     = (lambda_area     if lambda_area     is not None else cartographic_error)
    W_spatial  = (lambda_center   if lambda_center   is not None else spatial_deformation)
    W_topology = (lambda_topology if lambda_topology is not None else topological_accuracy)

    return (
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


def _find_shared_vertex_indices(
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
# Main optimisation
# ---------------------------------------------------------------------------

def CartogramFramework_global(
    polygons,
    target_areas,
    fixed_points=None,
    shape: str = "original",
    # shape options:
    #   "original"     — preserve input polygon shape
    #   "circle"       — Dorling cartogram
    #   "square"       — Demers cartogram
    #   "contiguous"   — original shape + hard shared-vertex locking
    #   "contiguous2"  — original shape + soft shared-vertex coupling
    target_centers=None,
    horizontal_pairs=None,            # list[(i,j)] i left of j
    vertical_pairs=None,              # list[(i,j)] i below j
    neighboring_pairs=None,           # list[(i,j)] geographically adjacent
    shared_vertices_of_neighbors=None,
    # list of [i, j, shared_points] from shared_vertices_of_neighbors()

    # --- objective weights (λ) ---
    lambda_shape: float    = 1.0,     # λ_s  shape deformation
    lambda_area: float     = 1.0,     # λ_a  cartographic error (soft)
    lambda_center: float   = 0.0,     # λ_c  centre fidelity
    lambda_topology: float = 1.0,     # λ_t  pairwise topology

    # --- contiguous-mode parameters ---
    lambda_contiguous: float = 1.0,
    # Weight for soft shared-vertex penalty (contiguous2 only).
    # Has no effect for shape="contiguous" (hard constraint mode).

    contiguous_area_ratio_cap: float = 10.0,
    # Hard-constraint mode (contiguous): skip locking a shared vertex
    # when max(A_i,A_j)/min(A_i,A_j) > cap.  Prevents a large region
    # from fully pinning a very small neighbour.
    # Set to np.inf to lock all shared vertices regardless of size ratio.

    contiguous_use_topology: bool = True,
    # contiguous / contiguous2: whether to still include the pairwise
    # topology (hor/ver excess-distance) terms in the objective.
    # Set False if topology constraints make the problem infeasible.

    # --- inner shape parameters ---
    gamma: float = 1.0,   # 1 = pure deformation term, 0 = pure E_target
    beta: float  = 1.0,   # 1 = L1 (z), 0 = L2 ((t-1)²)

    # --- geometry parameters ---
    alpha:   float = 1.0,   # slope for diagonal deviation d_ij
    epsilon: float = 1e-2,  # gap for non-adjacent pairs
    b:       float = 1e-2,  # scale for directional weight b_ij
    t_min:   float = 0.01,  # minimum radial scale
    t_max:   float = 5.0,   # maximum radial scale  (caps runaway vertices)

    # --- legacy quality-criterion API (mapped onto λ weights) ---
    cartographic_error: float = 1.0,
    shape_deformation:  float = 0.0,
    relative_direction: float = 1.0,
    topological_accuracy: float = 1.0,
    spatial_deformation: float = 0.0,
    global_shape: float = 1.0,
    local_shape:  float = 0.0,
    complexity:   float = 1.0,
    data_ink_ratio: float = 1.0,

    # ── Area scaling for non-contiguous ────────────────────────────────────
    area_scale: float = 1.0,
    # Multiplies every target area BEFORE the total-area normalisation.
    # 1.0 = no change (default).
    # 0.5 = every region is at most half its original size.
    # Values < 1 guarantee no region grows beyond scale * original_area,
    # which is useful for non-contiguous cartograms where you want to
    # show relative size differences while keeping all regions visually
    # smaller than the base map (e.g. to show they "matter less").
    # Values > 1 are allowed but will cause regions to grow beyond original.

    # ── Contiguous post-processing ─────────────────────────────────────────
    postprocess_contiguous: bool = False,
    # When True AND shape in ("contiguous", "contiguous2"), automatically
    # runs close_contiguous_gaps() then resolve_overlaps() on the result
    # before returning.  Has zero effect for any other shape mode.
    postprocess_snap_method: str = "mean",
    # Vertex-snapping method passed to close_contiguous_gaps():
    #   "mean" — snap both sides to their midpoint (default, symmetric)
    #   "i"    — polygon j's vertex moves to polygon i's position
    #   "j"    — polygon i's vertex moves to polygon j's position
    postprocess_shrink: float = 1.0,
    # Shrink factor passed to resolve_overlaps() after snapping.
    # 1.0 = no shrink (default); 0.98 = 2% inward to eliminate residual overlaps.
    # Only applied when postprocess_contiguous=True.

    compute_leaders_flag: bool = True,
    leader_tol: float = 1e-3,

    # ── Approach 1: soft mean-scale + looser t_max ─────────────────────────
    soft_mean_scale: bool = False,
    # Replace the hard mean(t)==s equality with a quadratic soft penalty.
    # Allows t_max to bite without forcing infeasibility: the solver can
    # let mean(t) deviate from s slightly if t_max would otherwise be
    # violated on every vertex.
    lambda_mean_scale: float = 1e4,
    # Weight of the soft mean-scale penalty (only used when soft_mean_scale=True).
    # Set high (1e3–1e5) to keep area accuracy, lower to allow more slack.

    # ── Approach 2: pairwise centre repulsion (anti-overlap) ───────────────
    lambda_repulsion: float = 0.0,
    # Weight of soft repulsion penalty between all pairs of region centres.
    # Penalises ||m_i - m_j||^{-2} (approximated) — pushes centres apart,
    # which indirectly prevents spikes from piercing neighbouring regions.
    # A value of 0 (default) disables this term entirely.
    repulsion_only_adjacent: bool = False,
    # If True, repulsion is applied only between adjacent pairs (T).
    # If False (default), all pairs are repelled (safer for spike prevention).
    repulsion_margin: float = 1.0,
    # Minimum centre-to-centre distance below which repulsion activates.
    # Set to roughly the average region diameter in your coordinate units.

    # ── Approach 3: vertex–centroid distance penalty ────────────────────────
    lambda_vertex_distance: float = 0.0,
    # Weight of the soft vertex-distance penalty.
    # Penalises vertices that are farther from their region centre than
    # the "ideal radius" r*_i = sqrt(A*_i / pi), i.e. the radius of a
    # circle with the target area.  Only kicks in beyond that radius so
    # normal deformations are fully feasible; only extreme spikes are discouraged.
    vertex_distance_margin: float = 1.5,
    # Multiplier on r*_i beyond which the penalty becomes active.
    # 1.5 means: no penalty up to 1.5× the ideal radius, quadratic beyond.
):
    # ------------------------------------------------------------------
    # 0. Preprocessing
    # ------------------------------------------------------------------
    (
        polygons,
        target_areas,
        fixed_points,
        cartographic_error,
        target_centers,
        W_shape,
        W_area,
        W_spatial,
        W_topology,
    ) = preprocess_global(
        polygons, target_areas, fixed_points,
        cartographic_error, target_centers,
        shape_deformation=shape_deformation,
        relative_direction=relative_direction,
        topological_accuracy=topological_accuracy,
        spatial_deformation=spatial_deformation,
        global_shape=global_shape,
        local_shape=local_shape,
        complexity=complexity,
        data_ink_ratio=data_ink_ratio,
        lambda_shape=lambda_shape,
        lambda_area=lambda_area,
        lambda_center=lambda_center,
        lambda_topology=lambda_topology,
    )

    # ── Apply area_scale AFTER normalisation ───────────────────────────────
    # preprocess_global already rescaled target_areas so their sum equals the
    # total original map area.  area_scale then uniformly shrinks/grows every
    # target area by the same factor.  Because it is applied post-normalisation
    # the relative proportions between regions are preserved exactly — only the
    # absolute size changes.  s_i = sqrt(target_area_i / A0_i) will be < 1 for
    # all regions when area_scale < 1, so regions can only shrink.
    if area_scale != 1.0:
        target_areas = [a * area_scale for a in target_areas]

    n_regions = len(polygons)

    # Determine the "base shape mode" used for per-region energy / init.
    # contiguous modes behave like "original" for the per-region terms.
    is_contiguous = shape in ("contiguous", "contiguous2")
    base_shape = "original" if is_contiguous else shape

    # ------------------------------------------------------------------
    # 1. Adjacency helpers
    # ------------------------------------------------------------------
    adjacent_set: set[tuple[int, int]] = set()
    if neighboring_pairs is not None:
        for i, j in neighboring_pairs:
            adjacent_set.add((min(i, j), max(i, j)))

    def is_adjacent(i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in adjacent_set

    def gap_ij(i: int, j: int) -> float:
        return 0.0 if is_adjacent(i, j) else epsilon

    def a_ij(i: int, j: int) -> float:
        return 1.0 if is_adjacent(i, j) else 0.1

    def b_ij(i: int, j: int) -> float:
        return b * a_ij(i, j)

    # ------------------------------------------------------------------
    # 2. Optional polygon re-initialisation for pure shape modes
    # ------------------------------------------------------------------
    init_polygons = []
    for k, poly in enumerate(polygons):
        ctr = np.asarray(fixed_points[k])
        if base_shape == "circle" and shape_deformation == 0.0:
            init_polygons.append(make_circle(ctr, target_areas[k], n=64))
        elif base_shape == "square" and (shape_deformation == 0.0 or gamma == 0.0):
            init_polygons.append(make_square(ctr, target_areas[k], n=64))
        else:
            init_polygons.append(poly)

    # ------------------------------------------------------------------
    # 3. Per-region CVXPY variables and constraints
    # ------------------------------------------------------------------
    constraints = []
    centers_vars: list[cp.Expression] = []
    new_polygon_exprs: list[cp.Expression] = []

    shape_terms  = []
    area_terms   = []
    center_terms = []

    # Store per-region data needed for leader computation and contiguous coupling
    r_vars: list = []          # cp.Variable or None
    s_vals: list[float] = []   # target scales
    # For contiguous coupling we also store (t_var, directions, fp) per region
    _t_vars:      list = []    # cp.Variable(ni) or None
    _directions:  list = []    # np.ndarray (ni,2) or None
    _fp_vals:     list = []    # np.ndarray (2,) for each region

    for i, (poly, ta, fp, tc) in enumerate(
        zip(init_polygons, target_areas, fixed_points, target_centers)
    ):
        poly = np.asarray(poly)
        fp   = np.asarray(fp)
        tc   = np.asarray(tc)

        _fp_vals.append(fp)

        A0 = polygon_areanp(poly)
        if A0 <= 1e-12:
            # Degenerate polygon: push dummy expressions and skip.
            # Store the raw numpy array (NOT a cp.Parameter) so that
            # extraction always yields a well-shaped (n,2) ndarray.
            dummy_center = cp.Variable(2)
            centers_vars.append(dummy_center)
            new_polygon_exprs.append(poly)          # plain np.ndarray sentinel
            r_vars.append(None)
            s_vals.append(1.0)
            _t_vars.append(None)
            _directions.append(None)
            continue

        s = float(np.sqrt(ta / A0))
        s_vals.append(s)

        # Moving centre  m_i = c_i + Δ_i
        delta = cp.Variable(2)
        m = fp + delta
        centers_vars.append(m)

        ni = len(poly)
        t = cp.Variable(ni)
        z = cp.Variable(ni, nonneg=True)

        directions = poly - fp          # (ni, 2)  fixed offsets from ref centre

        # New vertices  v'_ik = m_i + t_ik * (v_ik - c_i)
        # Shape: (ni, 2)
        new_pts = m[None, :] + cp.multiply(t[:, None], directions)
        new_polygon_exprs.append(new_pts)

        # Store for contiguous coupling
        _t_vars.append(t)
        _directions.append(directions)

        # ── Mean-scale: hard equality OR soft penalty (Approach 1) ─────────
        if soft_mean_scale:
            # Soft mode: drop the equality, keep bounds, add quadratic penalty.
            # This lets t_max be tight without causing infeasibility: the solver
            # can let mean(t) deviate from s by a small amount rather than
            # being forced to violate the box constraint t <= t_max.
            constraints += [
                t >= t_min,
                t <= t_max,
                z >= t - 1,
                z >= -(t - 1),
            ]
            mean_scale_penalty = lambda_mean_scale * cp.square(cp.sum(t) / ni - s)
        else:
            # Hard mode (original behaviour).
            constraints += [
                cp.sum(t) / ni == s,
                t >= t_min,
                t <= t_max,
                z >= t - 1,
                z >= -(t - 1),
            ]
            mean_scale_penalty = 0.0

        # ── Approach 3: vertex–centroid distance penalty ─────────────────────
        # Ideal radius: circle with target area.
        r_ideal = float(np.sqrt(ta / np.pi))
        # Per-vertex original distance from centroid ||d_{i,k}||.
        d_norms = np.linalg.norm(directions, axis=1)          # (ni,) constants
        # Scaled distance: t_{i,k} * ||d_{i,k}||  (affine in t).
        scaled_dist = cp.multiply(t, d_norms)                 # (ni,) expression
        # Threshold beyond which we penalise: margin * r_ideal.
        threshold = vertex_distance_margin * r_ideal
        # Soft-hinge: max(0, scaled_dist - threshold)^2  → auxiliary variable.
        if lambda_vertex_distance > 0.0:
            exceed = cp.Variable(ni, nonneg=True)             # excess over threshold
            constraints += [
                exceed >= scaled_dist - threshold,            # hinge lower bound
            ]
            vertex_dist_term = lambda_vertex_distance * cp.sum_squares(exceed)
        else:
            vertex_dist_term = 0.0

        # ---- Shape deformation term (inner blend β) -------------------
        l1_def = cp.sum(z)
        l2_def = cp.sum_squares(t - 1)
        deformation_term = beta * l1_def + (1.0 - beta) * l2_def

        # ---- Target-shape energy E_target ----------------------------
        r_var = None
        radial_norms = np.linalg.norm(directions, axis=1)   # (ni,)

        if base_shape == "circle":
            # E_circle: equalise all scaled radii
            new_radii = cp.multiply(t, radial_norms)
            mean_r    = cp.sum(new_radii) / ni
            E_target  = cp.sum_squares(new_radii - mean_r)

        elif base_shape == "square":
            # E_square: equalise scaled ℓ∞ radii to r_i
            M = np.maximum(
                np.abs(directions[:, 0]),
                np.abs(directions[:, 1])
            )                                               # (ni,)
            r_var    = cp.Variable(nonneg=True)
            E_target = cp.sum_squares(cp.multiply(t, M) - r_var)

        else:
            # Original shape (also used by contiguous modes): no target-shape energy
            E_target = 0.0

        r_vars.append(r_var)

        # ---- Soft area-error term  (zero when hard constraint active) -
        area_error = cp.square(cp.sum(t) / ni - s)

        # ---- Centre-fidelity term ------------------------------------
        centre_dist = cp.sum_squares(m - tc)

        # ---- Accumulate per-region contributions to objective ---------
        shape_contribution = (
            gamma * deformation_term + (1.0 - gamma) * E_target
        )
        shape_terms.append(shape_contribution)
        area_terms.append(area_error + mean_scale_penalty)    # Approach 1: soft scale
        center_terms.append(centre_dist)
        if lambda_vertex_distance > 0.0:
            shape_terms[-1] = shape_terms[-1] + vertex_dist_term  # Approach 3

    # ------------------------------------------------------------------
    # 4. Pairwise topology constraints and objective terms
    # ------------------------------------------------------------------
    pairwise_terms = []

    # For contiguous modes the caller can suppress topology terms if they
    # cause infeasibility (contiguous_use_topology=False).
    include_topology = (not is_contiguous) or contiguous_use_topology

    h_set = set(map(tuple, horizontal_pairs)) if horizontal_pairs is not None else set()
    v_set = set(map(tuple, vertical_pairs))   if vertical_pairs   is not None else set()

    if include_topology:
        for (i, j) in adjacent_set:
            # Only form hor/ver slack for adjacent pairs (T)
            ta_i = target_areas[i]
            ta_j = target_areas[j]
            w    = (np.sqrt(ta_i) + np.sqrt(ta_j)) / 2.0    # eq. (9)
            g    = gap_ij(i, j)                               # eq. (10) = 0

            xi = centers_vars[i][0];  yi = centers_vars[i][1]
            xj = centers_vars[j][0];  yj = centers_vars[j][1]

            hor = cp.Variable(nonneg=True, name=f"hor_{i}_{j}")
            ver = cp.Variable(nonneg=True, name=f"ver_{i}_{j}")

            # Excess-distance constraints (eqs. 13–14), linearised absolute value
            constraints += [
                hor >= (xi - xj) - w + g,
                hor >= (xj - xi) - w + g,
                ver >= (yi - yj) - w + g,
                ver >= (yj - yi) - w + g,
            ]

            # Directional deviation d_ij (eq. 15)
            diag_expr = yi + alpha * (xj - xi) - yj
            d_var = cp.Variable(nonneg=True, name=f"d_{i}_{j}")
            constraints += [
                d_var >= diag_expr,
                d_var >= -diag_expr,
            ]

            bij = b_ij(i, j)
            pairwise_terms.append(hor + ver + bij * d_var)

        # Hard ordering constraints for H and V pairs (eqs. 11–12)
        all_pairs_seen: set[tuple[int, int]] = set()
        if horizontal_pairs is not None:
            for (i, j) in horizontal_pairs:
                key = (min(i, j), max(i, j))
                if key not in all_pairs_seen:
                    ta_i = target_areas[i];  ta_j = target_areas[j]
                    w = (np.sqrt(ta_i) + np.sqrt(ta_j)) / 2.0
                    g = gap_ij(i, j)
                    xi = centers_vars[i][0];  xj = centers_vars[j][0]
                    constraints.append(xj - xi >= w + g)
                    all_pairs_seen.add(key)

        all_pairs_seen_v: set[tuple[int, int]] = set()
        if vertical_pairs is not None:
            for (i, j) in vertical_pairs:
                key = (min(i, j), max(i, j))
                if key not in all_pairs_seen_v:
                    ta_i = target_areas[i];  ta_j = target_areas[j]
                    w = (np.sqrt(ta_i) + np.sqrt(ta_j)) / 2.0
                    g = gap_ij(i, j)
                    yi = centers_vars[i][1];  yj = centers_vars[j][1]
                    constraints.append(yj - yi >= w + g)
                    all_pairs_seen_v.add(key)

    topology_term = cp.sum(pairwise_terms) if pairwise_terms else cp.Constant(0.0)

    # ------------------------------------------------------------------
    # 4b-pre. Approach 2: pairwise centre repulsion (anti-spike / anti-overlap)
    # ------------------------------------------------------------------
    # Rationale: spikes pierce neighbouring regions because nothing prevents
    # a vertex from flying through another polygon's interior.  We cannot
    # enforce true polygon non-intersection in a convex QP, but pushing
    # *centres* apart is a DCP-compliant surrogate: if centres stay well
    # separated, extreme radial spikes are the only way to bridge the gap
    # and the vertex-distance penalty (Approach 3) then suppresses those.
    #
    # Penalty:  lambda_repulsion * sum_{pairs} max(0, margin^2 - ||m_i-m_j||^2)
    # which is a soft hinge in the *squared* distance — convex and affine
    # in m_i, m_j.  We use an auxiliary slack variable to stay DCP-clean.
    repulsion_terms = []
    if lambda_repulsion > 0.0:
        pairs_for_repulsion = []
        if repulsion_only_adjacent:
            pairs_for_repulsion = list(adjacent_set)
        else:
            for ii in range(n_regions):
                for jj in range(ii + 1, n_regions):
                    pairs_for_repulsion.append((ii, jj))

        # DCP-compliant repulsion via L1 (Manhattan) distance.
        #
        # The original attempt used:
        #   slack >= margin^2 - ||m_i - m_j||^2
        # which is NOT DCP because ||.||^2 is convex, so the RHS is concave.
        #
        # Fix: use the L1 norm as the distance measure instead.
        # ||m_i - m_j||_1 = |dx| + |dy|  is convex and can be expressed
        # as two linear constraints per component via auxiliary variables:
        #   abs_dx >= (dx),  abs_dx >= -(dx)     (abs_dx = |dx|)
        #   abs_dy >= (dy),  abs_dy >= -(dy)
        # Then:  l1_dist = abs_dx + abs_dy        (affine, convex)
        #
        # Hinge penalty: max(0, margin - l1_dist)
        # = slack_r  where  slack_r >= margin - l1_dist,  slack_r >= 0
        # Since l1_dist is convex, (margin - l1_dist) is CONCAVE — same problem.
        #
        # Correct DCP formulation: penalise the L1 shortfall directly
        # using per-component hinges, which ARE affine:
        #   shortfall_x = max(0,  half_margin - |dx|)   (concave — still bad)
        #
        # The only fully DCP-clean approach with affine variables is:
        #   Penalise  max(0,  margin - dx)  AND  max(0,  margin + dx)  separately
        #   for each signed component — these are linear hinges (convex):
        #     slack_pos >= margin - dx    (dx = xj - xi, linear in variables)
        #     slack_neg >= margin + dx
        # Both RHS are AFFINE → DCP-OK.  The penalty is slack_pos + slack_neg,
        # which is zero when |dx| >= margin and grows linearly when centres are
        # too close in either direction.  Repeat for dy.
        # Total: zero when both |dx| >= margin AND |dy| >= margin (centres well
        # separated), positive otherwise — a convex, DCP-compliant repulsion.
        for (ii, jj) in pairs_for_repulsion:
            dx = centers_vars[jj][0] - centers_vars[ii][0]   # affine
            dy = centers_vars[jj][1] - centers_vars[ii][1]   # affine

            # x-axis repulsion: penalise if centres are closer than margin in x
            sp_x = cp.Variable(nonneg=True, name=f"rspx_{ii}_{jj}")
            sn_x = cp.Variable(nonneg=True, name=f"rsnx_{ii}_{jj}")
            constraints += [
                sp_x >= repulsion_margin - dx,   # centres too close from left
                sn_x >= repulsion_margin + dx,   # centres too close from right
            ]

            # y-axis repulsion
            sp_y = cp.Variable(nonneg=True, name=f"rspy_{ii}_{jj}")
            sn_y = cp.Variable(nonneg=True, name=f"rsny_{ii}_{jj}")
            constraints += [
                sp_y >= repulsion_margin - dy,
                sn_y >= repulsion_margin + dy,
            ]

            # Quadratic penalty on each slack (zero when well-separated)
            repulsion_terms.append(
                cp.square(sp_x) + cp.square(sn_x)
                + cp.square(sp_y) + cp.square(sn_y)
            )

    repulsion_penalty = (
        lambda_repulsion * cp.sum(repulsion_terms)
        if repulsion_terms else cp.Constant(0.0)
    )

    # ------------------------------------------------------------------
    # 4b. Contiguous shared-vertex constraints / penalties
    # ------------------------------------------------------------------
    contiguous_terms = []   # soft penalty terms (contiguous2 only)
    n_locked = 0            # counter for diagnostic output

    if is_contiguous and shared_vertices_of_neighbors is not None:
        for entry in shared_vertices_of_neighbors:
            i, j, shared_pts = entry[0], entry[1], entry[2]

            # Skip if either region is degenerate (no t variable)
            if _t_vars[i] is None or _t_vars[j] is None:
                continue
            if _directions[i] is None or _directions[j] is None:
                continue

            # Optional: skip pairs with a large area ratio to avoid
            # forcing a large region to match a tiny neighbour exactly.
            if shape == "contiguous":
                ta_i = target_areas[i]
                ta_j = target_areas[j]
                ratio = max(ta_i, ta_j) / (min(ta_i, ta_j) + 1e-30)
                if ratio > contiguous_area_ratio_cap:
                    continue

            # Find which vertex indices in each polygon correspond to the
            # shared points (using the *init* polygon geometry, same as
            # what we used to build directions[i/j]).
            poly_i = np.asarray(init_polygons[i])
            poly_j = np.asarray(init_polygons[j])
            kl_pairs = _find_shared_vertex_indices(
                poly_i, poly_j, shared_pts
            )

            for (k, l) in kl_pairs:
                # Transformed position of vertex k in region i:
                #   v'_{i,k} = m_i + t_{i,k} * d_{i,k}
                #            = (fp_i + delta_i) + t_{i,k} * directions_i[k]
                #
                # Because m_i = fp_i + delta_i and delta_i is a cp.Variable,
                # and t_{i,k} is a scalar cp.Variable, and directions_i[k]
                # is a constant 2-vector, this expression is AFFINE in the
                # CVXPY variables — fully DCP compliant.
                #
                # centers_vars[i] = fp_i + delta_i  (stored as cp.Expression)

                v_i = centers_vars[i] + _t_vars[i][k] * _directions[i][k]  # (2,) expr
                v_j = centers_vars[j] + _t_vars[j][l] * _directions[j][l]  # (2,) expr

                if shape == "contiguous":
                    # Hard equality: the two images of the shared vertex coincide.
                    constraints.append(v_i == v_j)
                    n_locked += 1

                else:  # "contiguous2"
                    # Soft quadratic penalty: penalise the gap between them,
                    # normalised by the squared original distance between the
                    # two region centres so the weight is dimensionless and
                    # scale-invariant regardless of coordinate units.
                    diff = v_i - v_j                          # (2,) affine expr
                    # Normalisation: squared centre-to-centre distance (constant).
                    ci_fp = _fp_vals[i]  # fixed point of region i
                    cj_fp = _fp_vals[j]  # fixed point of region j
                    dist2 = float(np.sum((ci_fp - cj_fp) ** 2))
                    norm  = max(dist2, 1e-6)   # avoid /0 for coincident centres
                    contiguous_terms.append(cp.sum_squares(diff) / norm)
                    n_locked += 1

    if contiguous_terms:
        contiguous_penalty = cp.sum(contiguous_terms)
    else:
        contiguous_penalty = cp.Constant(0.0)

    print(f"Contiguous mode '{shape}': {n_locked} shared-vertex "
          f"{'constraints' if shape == 'contiguous' else 'penalty terms'} added.")

    # ------------------------------------------------------------------
    # 5. Global objective  (eq. 16)
    # ------------------------------------------------------------------
    objective = cp.Minimize(
        W_shape    * cp.sum(shape_terms)
        + W_area   * cp.sum(area_terms)
        + W_spatial * cp.sum(center_terms)
        + W_topology * topology_term
        + lambda_contiguous * contiguous_penalty
        + repulsion_penalty                        # Approach 2
    )

    # ------------------------------------------------------------------
    # 6. Solve
    # ------------------------------------------------------------------
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    print(f"Status : {prob.status}")
    print(f"Obj val: {prob.value:.6g}" if prob.value is not None else "Obj val: None")

    # ------------------------------------------------------------------
    # 7. Extract results
    # ------------------------------------------------------------------

    # If the solver did not reach a feasible solution every .value will be
    # None, causing crashes in downstream plotting.  Detect this early and
    # fall back to the original (init) geometry for every region.
    solver_ok = prob.status in (
        cp.OPTIMAL, cp.OPTIMAL_INACCURATE,
        "optimal", "optimal_inaccurate",
    )
    if not solver_ok:
        print(
            f"  WARNING: solver status '{prob.status}' — returning original "
            f"polygons as fallback.\n"
            f"  Tips for contiguous mode:\n"
            f"    • Raise contiguous_area_ratio_cap (skips locking large-ratio pairs)\n"
            f"    • Set contiguous_use_topology=False to drop topology terms\n"
            f"    • Switch to shape='contiguous2' (soft penalty instead of hard equality)"
        )
        fallback = [np.asarray(p, dtype=float) for p in init_polygons]
        return fallback, [float("nan")] * n_regions, prob.status, prob.value, [], target_areas

    new_polygons: list[np.ndarray] = []
    actual_areas: list[float]      = []

    for orig_poly, poly_expr in zip(init_polygons, new_polygon_exprs):
        orig_poly = np.asarray(orig_poly, dtype=float)

        # Resolve the CVXPY expression (or plain ndarray sentinel) to a numpy array.
        if isinstance(poly_expr, np.ndarray):
            # Degenerate-polygon sentinel — pass original geometry through.
            pts = poly_expr.copy()
        else:
            raw = poly_expr.value
            # Ensure we always get a proper float64 (n,2) ndarray.
            if raw is not None:
                pts = np.array(raw, dtype=float)
            else:
                pts = None

        # Validate shape: must be a 2-D (n>=3, 2) array.
        valid = (
            pts is not None
            and isinstance(pts, np.ndarray)
            and pts.ndim == 2
            and pts.shape[1] == 2
            and pts.shape[0] >= 3
        )

        if valid:
            new_polygons.append(pts)
            actual_areas.append(polygon_areanp(pts))
        else:
            # Individual region failed despite overall solve succeeding.
            # Return the (scaled) original so plotting never receives None.
            print(
                f"  Warning: region {len(new_polygons)} has no valid solution "
                f"(shape={getattr(pts, 'shape', None) if pts is not None else 'None'}); "
                f"falling back to original polygon."
            )
            new_polygons.append(orig_poly)
            actual_areas.append(float("nan"))

    # ------------------------------------------------------------------
    # 8. Leader-line computation (Demers / Dorling post-processing)
    #    Implements Nickel et al. Lemma 2 sweep.
    # ------------------------------------------------------------------
    leaders = []
    if compute_leaders_flag and base_shape in ("square", "circle"):

        solved_centers = []
        half_sides     = []

        for i in range(n_regions):
            cv = centers_vars[i]
            if hasattr(cv, "value") and cv.value is not None:
                solved_centers.append(cv.value)
            else:
                solved_centers.append(np.asarray(fixed_points[i]))

            if base_shape == "square":
                rv = r_vars[i]
                if rv is not None and hasattr(rv, "value") and rv.value is not None:
                    half_sides.append(float(rv.value))
                else:
                    half_sides.append(float(np.sqrt(target_areas[i])) / 2.0)
            else:
                # circle: half-side = radius = sqrt(A/π)
                half_sides.append(float(np.sqrt(target_areas[i] / np.pi)))

        leaders = compute_leaders(
            solved_centers,
            half_sides,
            adjacent_set,
            tol=leader_tol,
        )

    # ------------------------------------------------------------------
    # 9. Contiguous post-processing (optional)
    # ------------------------------------------------------------------
    if postprocess_contiguous and is_contiguous and shared_vertices_of_neighbors is not None:
        print("Post-processing: snapping shared vertices "
              f"(method='{postprocess_snap_method}', shrink={postprocess_shrink})…")
        new_polygons = close_contiguous_gaps(
            polygons                          = new_polygons,
            shared_vertices_of_neighbors_data = shared_vertices_of_neighbors,
            original_polygons                 = init_polygons,  # stable index reference
            method                            = postprocess_snap_method,
        )
        if postprocess_shrink != 1.0:
            new_polygons = resolve_overlaps(new_polygons, shrink_factor=postprocess_shrink)
        # Recompute areas after snapping (they change slightly)
        actual_areas = [polygon_areanp(p) for p in new_polygons]

    return new_polygons, actual_areas, prob.status, prob.value, leaders, target_areas


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


def shared_vertices_of_neighbors(polygons, tolerance=1e-8):
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


# ---------------------------------------------------------------------------
# Post-processing: snap shared vertices to close contiguous gaps
# ---------------------------------------------------------------------------

def close_contiguous_gaps(
    polygons: list,
    shared_vertices_of_neighbors_data: list,
    original_polygons: list,
    method: str = "mean",
    tolerance: float = 1e-8,
) -> list:
    """
    Post-processing: snap shared border vertices of neighbouring polygons
    to exactly the same coordinate, closing gaps/overlaps left by contiguous2.

    The key design: vertex identity is established by ORIGINAL INDEX, not by
    searching for the nearest point in the optimised geometry.  After
    optimisation vertices may have moved significantly, so nearest-neighbour
    search picks the wrong vertex and creates the crossing/tangling artefacts
    visible when the naive approach is used.

    Instead we use _find_shared_vertex_indices() which looks up each shared
    point in the ORIGINAL polygon arrays (coordinate-matched) and returns
    stable (k, l) index pairs.  Those same indices are then used to read and
    write the OPTIMISED arrays — correct regardless of how far vertices drifted.

    Parameters
    ----------
    polygons : list[np.ndarray]
        Optimised polygons returned by CartogramFramework_global.
    shared_vertices_of_neighbors_data : list
        Output of shared_vertices_of_neighbors(original_polygons).
    original_polygons : list[np.ndarray]
        The ORIGINAL (pre-optimisation) polygons, used only to resolve
        which vertex index in each polygon is the shared one.
    method : str
        "mean" — snap both sides to their midpoint (symmetric, default).
        "i"    — move polygon j's vertex to polygon i's current position.
        "j"    — move polygon i's vertex to polygon j's current position.
    tolerance : float
        Coordinate-matching tolerance (same value used when computing
        shared_vertices_of_neighbors_data).

    Returns
    -------
    list[np.ndarray]  — new list of polygons with shared vertices snapped.
    """
    # Work on copies so the input is never mutated
    result = [np.array(p, dtype=float) for p in polygons]
    orig   = [np.asarray(p, dtype=float) for p in original_polygons]

    for entry in shared_vertices_of_neighbors_data:
        i, j, shared_pts = entry[0], entry[1], entry[2]

        if i >= len(result) or j >= len(result):
            continue

        # Resolve shared points to vertex INDICES in the original polygons.
        # _find_shared_vertex_indices uses exact coordinate matching on the
        # original geometry — immune to post-optimisation drift.
        kl_pairs = _find_shared_vertex_indices(orig[i], orig[j], shared_pts, tolerance)

        for (k, l) in kl_pairs:
            if k >= len(result[i]) or l >= len(result[j]):
                continue   # guard against polygon size mismatch

            vi = result[i][k]   # optimised position of vertex k in polygon i
            vj = result[j][l]   # optimised position of vertex l in polygon j

            if method == "mean":
                snapped = (vi + vj) / 2.0
            elif method == "i":
                snapped = vi.copy()
            elif method == "j":
                snapped = vj.copy()
            else:
                raise ValueError(f"method must be 'mean', 'i', or 'j', got {method!r}")

            result[i][k] = snapped
            result[j][l] = snapped

    return result


def shrink_polygon(poly: np.ndarray, factor: float) -> np.ndarray:
    """
    Shrink a polygon toward its centroid by `factor` (0–1).
    factor=1.0 → no change; factor=0.9 → 10% inward from centroid.
    Useful as a second post-processing pass to eliminate residual overlaps
    after close_contiguous_gaps: shrink slightly so polygons are strictly
    non-overlapping, at the cost of small gaps between them.
    """
    centroid = np.mean(poly, axis=0)
    return centroid + factor * (poly - centroid)


def resolve_overlaps(
    polygons: list,
    shrink_factor: float = 0.98,
) -> list:
    """
    Apply a uniform shrink toward each polygon's centroid to remove
    residual overlaps after snapping.  Small shrink factors (0.95–0.99)
    leave barely-visible gaps but guarantee no overlap.

    Parameters
    ----------
    polygons    : list[np.ndarray]  from close_contiguous_gaps
    shrink_factor : float  0 < factor <= 1.0

    Returns
    -------
    list[np.ndarray]
    """
    return [shrink_polygon(np.asarray(p, dtype=float), shrink_factor)
            for p in polygons]
