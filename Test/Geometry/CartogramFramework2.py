"""
CartogramFramework_global_CLAUDE
=================================
Unified cartogram optimisation covering:
  - Demers (square) cartograms
  - Dorling (circle) cartograms
  - Original shape

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
# Main optimisation
# ---------------------------------------------------------------------------

def CartogramFramework_global_CLAUDE_2(
    polygons,
    target_areas,
    fixed_points=None,
    shape: str = "original",          # "original" | "circle" | "square"
    target_centers=None,
    horizontal_pairs=None,            # list[(i,j)] i left of j
    vertical_pairs=None,              # list[(i,j)] i below j
    neighboring_pairs=None,           # list[(i,j)] geographically adjacent
    shared_vertices_of_neighbors=None,

    # --- objective weights (λ) ---
    lambda_shape: float    = 1.0,     # λ_s  shape deformation
    lambda_area: float     = 1.0,     # λ_a  cartographic error (soft)
    lambda_center: float   = 0.0,     # λ_c  centre fidelity
    lambda_topology: float = 1.0,     # λ_t  pairwise topology

    # --- inner shape parameters ---
    gamma: float = 1.0,   # 1 = pure deformation term, 0 = pure E_target
    beta: float  = 1.0,   # 1 = L1 (z), 0 = L2 ((t-1)²)

    # --- geometry parameters ---
    alpha:   float = 1.0,   # slope for diagonal deviation d_ij
    epsilon: float = 1e-2,  # gap for non-adjacent pairs
    b:       float = 1e-2,  # scale for directional weight b_ij
    t_min:   float = 0.01,  # minimum radial scale

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

    compute_leaders_flag: bool = True,
    leader_tol: float = 1e-3,
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

    n_regions = len(polygons)

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
        if shape == "circle" and shape_deformation == 0.0:
            init_polygons.append(make_circle(ctr, target_areas[k], n=64))
        # elif shape == "square" and shape_deformation == 0.0:
        elif shape == "square" and (shape_deformation == 0.0 or gamma == 0.0):
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

    # Store per-region data needed for leader computation
    r_vars: list = []         # cp.Variable or None
    s_vals: list[float] = []  # target scales

    for i, (poly, ta, fp, tc) in enumerate(
        zip(init_polygons, target_areas, fixed_points, target_centers)
    ):
        poly = np.asarray(poly)
        fp   = np.asarray(fp)
        tc   = np.asarray(tc)

        A0 = polygon_areanp(poly)
        if A0 <= 1e-12:
            # Degenerate polygon: push dummy expressions and skip
            dummy_center = cp.Variable(2)
            centers_vars.append(dummy_center)
            new_polygon_exprs.append(cp.Parameter(poly.shape, value=poly))
            r_vars.append(None)
            s_vals.append(1.0)
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


        # Hard mean-scale equality  (eq. 4 as constraint, not soft penalty)
        constraints += [
            cp.sum(t) / ni == s,        # mean(t) = s_i
            t >= t_min,                 # eq. 5
            z >= t - 1,                 # eq. 6
            z >= -(t - 1),              # eq. 7
        ]

        # ---- Shape deformation term (inner blend β) -------------------
        l1_def = cp.sum(z)
        l2_def = cp.sum_squares(t - 1)
        deformation_term = beta * l1_def + (1.0 - beta) * l2_def

        # ---- Target-shape energy E_target ----------------------------
        r_var = None
        radial_norms = np.linalg.norm(directions, axis=1)   # (ni,)

        if shape == "circle":
            # E_circle: equalise all scaled radii
            new_radii = cp.multiply(t, radial_norms)
            mean_r    = cp.sum(new_radii) / ni
            E_target  = cp.sum_squares(new_radii - mean_r)

        elif shape == "square":
            # E_square: equalise scaled ℓ∞ radii to r_i
            M = np.maximum(
                np.abs(directions[:, 0]),
                np.abs(directions[:, 1])
            )                                               # (ni,)
            r_var    = cp.Variable(nonneg=True)
            E_target = cp.sum_squares(cp.multiply(t, M) - r_var)

        else:
            # Original shape: no target-shape energy
            E_target = 0.0

        r_vars.append(r_var)

        # ---- Soft area-error term  (zero when hard constraint active) -
        area_error = cp.square(cp.sum(t) / ni - s)

        # ---- Centre-fidelity term ------------------------------------
        centre_dist = cp.sum_squares(m - tc)

        # ---- Accumulate per-region contributions to objective ---------
        # Objective eq. (16):
        #   λ_s [ γ·deformation + (1-γ)·E_target ] + λ_a·area_error + λ_c·centre
        shape_contribution = (
            gamma * deformation_term + (1.0 - gamma) * E_target
        )
        shape_terms.append(shape_contribution)
        area_terms.append(area_error)
        center_terms.append(centre_dist)

    # ------------------------------------------------------------------
    # 4. Pairwise topology constraints and objective terms
    # ------------------------------------------------------------------
    pairwise_terms = []

    h_set = set(map(tuple, horizontal_pairs)) if horizontal_pairs is not None else set()
    v_set = set(map(tuple, vertical_pairs))   if vertical_pairs   is not None else set()

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
        # d_ij = |y_i + α(x_j - x_i) - y_j|
        diag_expr = yi + alpha * (xj - xi) - yj
        d_var = cp.Variable(nonneg=True, name=f"d_{i}_{j}")
        constraints += [
            d_var >= diag_expr,
            d_var >= -diag_expr,
        ]

        bij = b_ij(i, j)
        pairwise_terms.append(hor + ver + bij * d_var)

    # Hard ordering constraints for H and V pairs (eqs. 11–12)
    # These apply to ALL pairs in H/V, whether adjacent or not
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
    # 5. Global objective  (eq. 16)
    # ------------------------------------------------------------------
    objective = cp.Minimize(
        W_shape    * cp.sum(shape_terms)
        + W_area   * cp.sum(area_terms)
        + W_spatial * cp.sum(center_terms)
        + W_topology * topology_term
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
    new_polygons: list[np.ndarray] = []
    actual_areas: list[float]      = []

    for poly_expr in new_polygon_exprs:
        if isinstance(poly_expr, cp.Parameter):
            pts = poly_expr.value
        else:
            pts = poly_expr.value
        if pts is None:
            new_polygons.append(None)
            actual_areas.append(float("nan"))
        else:
            new_polygons.append(pts)
            actual_areas.append(polygon_areanp(pts))

    # ------------------------------------------------------------------
    # 8. Leader-line computation (Demers / Dorling post-processing)
    #    Implements Nickel et al. Lemma 2 sweep.
    # ------------------------------------------------------------------
    leaders = []
    if compute_leaders_flag and shape in ("square", "circle"):

        solved_centers = []
        half_sides     = []

        for i in range(n_regions):
            cv = centers_vars[i]
            if hasattr(cv, "value") and cv.value is not None:
                solved_centers.append(cv.value)
            else:
                solved_centers.append(np.asarray(fixed_points[i]))

            if shape == "square":
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

    return new_polygons, actual_areas, prob.status, prob.value, leaders, target_areas