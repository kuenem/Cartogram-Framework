"""
CartogramFramework_global
=================================
Unified cartogram optimisation covering:
  - Demers (square) cartograms
  - Dorling (circle) cartograms
  - Original shape
  - Contiguous shape  (shape="contiguous")
  - Contiguous soft   (shape="contiguous2")
"""

from __future__ import annotations
from copy import deepcopy

import numpy as np
import cvxpy as cp
from typing import Optional
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.geometry import *
from src.utils import *
from src.core import *
from src.core.preprocessing_data import *

# ---------------------------------------------------------------------------
# Main optimisation
# ---------------------------------------------------------------------------

def CartogramFramework_global(
    data,
    # polygons,
    # target_areas,
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
    # runs close_contiguous_gaps(). Has zero effect for any other shape mode.
    postprocess_snap_method: str = "mean",
    # Vertex-snapping method passed to close_contiguous_gaps():
    #   "mean" — snap both sides to their midpoint (default, symmetric)
    #   "i"    — polygon j's vertex moves to polygon i's position
    #   "j"    — polygon i's vertex moves to polygon j's position
    postprocess_disputed_pixels: bool = False,

    compute_leaders_flag: bool = True,
    leader_tol: float = 1e-3,

    # ── Approach 1: soft mean-scale + looser t_max ─────────────────────────
    soft_mean_scale: bool = True,
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
    ) = preprocess_global(
        data,
        fixed_points,
        cartographic_error,
        target_centers,
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

    data = deepcopy(data)  # avoid mutating the original input data

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

    region_names = [name for name in data.keys() if name != "__meta__"]

    # ------------------------------------------------------------------
    # 2. Optional polygon re-initialisation for pure shape modes
    # ------------------------------------------------------------------
    init_polygons = []
    for k, name in enumerate(region_names):
        ctr = np.asarray(fixed_points[k])
        if base_shape == "circle" and shape_deformation == 0.0:
            init_polygons.append(make_circle(ctr, target_areas[k], n=64))
        elif base_shape == "square" and (shape_deformation == 0.0 or gamma == 0.0):
            init_polygons.append(make_square(ctr, target_areas[k], n=64))
        else:
            init_polygons.append(data[name]["polygon"])

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
            # constraints += [
            #     hor >= (xi - xj) - w + g,
            #     hor >= (xj - xi) - w + g,
            #     ver >= (yi - yj) - w + g,
            #     ver >= (yj - yi) - w + g,
            # ]

            slack = cp.Variable(nonneg=True)
            constraints += [slack >= (w + g) - (xj - xi)]     # only a lower bound -> always feasible

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
            kl_pairs = find_shared_vertex_indices(
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

        for idx, name in enumerate(region_names):
            record = data[name]
            original_polygon = np.asarray(init_polygons[idx], dtype=float)
            record.setdefault("original_polygon", original_polygon)
            record.setdefault("original_area", float(polygon_areanp(original_polygon)))
            record.setdefault("polygon", original_polygon)
            record.setdefault("area", float(polygon_areanp(original_polygon)))
            record["new_polygon"] = fallback[idx]
            record["new_area"] = float(polygon_areanp(fallback[idx]))
            record["target_area"] = float(target_areas[idx])
            record["new_centroid"] = np.mean(fallback[idx], axis=0)
            record["status"] = prob.status
            record["objective_value"] = prob.value

        # Reserved key for solve-level (not per-region) info. Dunder-style
        # so it can never collide with an actual region name.
        data["__meta__"] = {
            "status": prob.status,
            "objective_value": prob.value,
            "leaders": [],
            "target_areas": target_areas,
            "n_regions": n_regions,
        }
        return data

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
              f"(method='{postprocess_snap_method}')")
        new_polygons = close_contiguous_gaps(
            polygons                          = new_polygons,
            shared_vertices_of_neighbors_data = shared_vertices_of_neighbors,
            original_polygons                 = init_polygons,
            method                            = postprocess_snap_method,
        )

        actual_areas = [polygon_areanp(p) for p in new_polygons]

    if postprocess_disputed_pixels:
        new_polygons = resolve_disputed_pixels_by_proximity(
            polygons = new_polygons,
            resolution = 500,
            region_order = None,
            bounds = None,
            simplify_tolerance = 0.0,
            return_raster = False,
        )

        actual_areas = [polygon_areanp(p) for p in new_polygons]

    # overlap_area_lost is always 0.0: the overlap-resolution post-processing
        # pass (resolve_polygon_overlaps) was removed as unused dead code — it was
        # never enabled by any call in exploration.ipynb. The field is kept in the
        # per-region record for downstream compatibility (e.g. plotting code that
        # expects the key to exist).
        overlap_area_lost = [0.0] * n_regions

 # ------------------------------------------------------------------
    # 10. Populate the data dict — single source of truth for results.
    #     Per-region results go on each record; solve-level results
    #     (status, objective, leaders) go under the reserved "__meta__" key.
    # ------------------------------------------------------------------
    for idx, name in enumerate(region_names):
        record = data[name]
        original_polygon = np.asarray(init_polygons[idx], dtype=float)
        solved_polygon = np.asarray(new_polygons[idx], dtype=float)
        record.setdefault("original_polygon", original_polygon)
        record.setdefault("original_area", float(polygon_areanp(original_polygon)))
        record.setdefault("polygon", original_polygon)
        record.setdefault("area", float(polygon_areanp(original_polygon)))
        record.setdefault("target_positions", np.asarray(target_centers[idx], dtype=float))
        record.setdefault("centroid", np.mean(original_polygon, axis=0))
        record["new_polygon"] = solved_polygon
        record["new_area"] = float(actual_areas[idx])
        record["target_area"] = float(target_areas[idx])
        record["new_centroid"] = np.mean(solved_polygon, axis=0)
        # record["overlap_area_lost"] = float(overlap_area_lost[idx])
        record["status"] = prob.status
        record["objective_value"] = prob.value
        record["constraints"] = constraints
        record["objective"] = objective
        print(constraints)
        print(objective)

    # data["__meta__"] = {
    #     "status": prob.status,
    #     "objective_value": prob.value,
    #     "leaders": leaders,
    #     "target_areas": target_areas,
    #     "n_regions": n_regions,
    # }

    return data


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
    # Work on copies so the input is never mutated
    result = [np.array(p, dtype=float) for p in polygons]
    orig   = [np.asarray(p, dtype=float) for p in original_polygons]

    for entry in shared_vertices_of_neighbors_data:
        i, j, shared_pts = entry[0], entry[1], entry[2]

        if i >= len(result) or j >= len(result):
            continue

        # Resolve shared points to vertex INDICES in the original polygons.
        # find_shared_vertex_indices uses exact coordinate matching on the
        # original geometry — immune to post-optimisation drift.
        kl_pairs = find_shared_vertex_indices(orig[i], orig[j], shared_pts, tolerance)

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


# ---------------------------------------------------------------------------
# Post-processing: assign disputed raster pixels to the nearest
# undisputed region (does NOT move any vertices)
# ---------------------------------------------------------------------------

def resolve_disputed_pixels_by_proximity(
    polygons: list,
    resolution: int = 500,
    region_order: Optional[list] = None,
    bounds: Optional[tuple] = None,
    simplify_tolerance: float = 0.0,
    return_raster: bool = False,
):
    from matplotlib.path import Path
    from scipy.ndimage import distance_transform_edt
 
    n_regions = len(polygons)
    if n_regions == 0:
        raise ValueError("polygons must contain at least one region")
 
    polys = [np.asarray(p, dtype=float) for p in polygons]
 
    if region_order is None:
        region_order = list(range(n_regions))
    priority = {r: rank for rank, r in enumerate(region_order)}
    for r in range(n_regions):
        priority.setdefault(r, n_regions + r)  # any region missing from
        # region_order falls back to input order, after the explicit list
 
    # ---- combined bounding box --------------------------------------------
    if bounds is None:
        all_xy = np.vstack(polys)
        xmin, ymin = all_xy.min(axis=0)
        xmax, ymax = all_xy.max(axis=0)
        pad_x = (xmax - xmin) * 0.01 or 1e-6
        pad_y = (ymax - ymin) * 0.01 or 1e-6
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
    else:
        xmin, ymin, xmax, ymax = bounds
 
    width, height = xmax - xmin, ymax - ymin
    if width >= height:
        nx = max(int(resolution), 2)
        pixel_size = width / nx
        ny = max(int(round(height / pixel_size)), 2)
    else:
        ny = max(int(resolution), 2)
        pixel_size = height / ny
        nx = max(int(round(width / pixel_size)), 2)
 
    # pixel-centre coordinates
    xs = xmin + (np.arange(nx) + 0.5) * pixel_size
    ys = ymin + (np.arange(ny) + 0.5) * pixel_size
    grid_x, grid_y = np.meshgrid(xs, ys)          # shape (ny, nx)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
 
    # ---- rasterise each region ---------------------------------------------
    masks = np.zeros((n_regions, ny, nx), dtype=bool)
    for r, poly in enumerate(polys):
        inside = Path(poly).contains_points(points, radius=1e-9)
        masks[r] = inside.reshape(ny, nx)
 
    coverage_count = masks.sum(axis=0)
    undisputed_masks = masks & (coverage_count == 1)[None, :, :]
 
    # ---- distance (in pixels) from every cell to each region's undisputed body
    dist_to_undisputed = np.full((n_regions, ny, nx), np.inf)
    for r in range(n_regions):
        if undisputed_masks[r].any():
            dist_to_undisputed[r] = distance_transform_edt(~undisputed_masks[r])
        # else: stays +inf -- region r has no solid undisputed body to
        # measure distance from, so it can only win via plain priority.
 
    # Regions that don't even claim a given pixel are never candidates there.
    dist_masked = np.where(masks, dist_to_undisputed, np.inf)   # (n_regions, ny, nx)
 
    # Reorder the region axis by tie-break priority (highest priority first).
    # np.argmin returns the FIRST index achieving the minimum, so on an exact
    # distance tie this naturally picks the highest-priority region -- no
    # separate tie-break pass needed.
    order = sorted(range(n_regions), key=lambda r: priority[r])
    dist_ordered = dist_masked[order]
    winner_in_order = np.argmin(dist_ordered, axis=0)
    winner = np.asarray(order)[winner_in_order]
 
    label_grid = np.where(coverage_count > 0, winner, -1).astype(int)
 
    # ---- trace region boundaries back out of the resolved raster ----------
    from skimage.measure import find_contours, approximate_polygon
 
    # Pad with a sentinel border so every region's mask is fully enclosed
    # (no contour touches the array edge, so all traced contours are closed
    # loops we can turn straight into polygons).
    padded = np.full((label_grid.shape[0] + 2, label_grid.shape[1] + 2), -999, dtype=int)
    padded[1:-1, 1:-1] = label_grid
 
    result_polygons = []
    n_empty_norasterize = 0
    n_empty_swallowed = 0
    for r in range(n_regions):
        mask = (padded == r).astype(float)
 
        if not mask.any():
            result_polygons.append(polys[r].copy())
            if masks[r].any():
                # It DID cover some raster pixels, but lost every one of
                # them to higher-priority neighbours (no undisputed core of
                # its own to win ties/distance with) -- fully swallowed.
                n_empty_swallowed += 1
            else:
                # It never rasterised at all: smaller than a pixel at this
                # resolution relative to the bounding box.
                n_empty_norasterize += 1
            continue
 
        contours = find_contours(mask, level=0.5)
        if not contours:
            result_polygons.append(polys[r].copy())
            n_empty_swallowed += 1
            continue
 
        # A region can trace into several disconnected loops (e.g. if it
        # got fully cut in two by neighbours); keep the largest by area as
        # the region's boundary, matching the one-polygon-per-region
        # convention used throughout this module.
        best = None
        best_area = -1.0
        for c in contours:
            # c is (row, col) in PADDED pixel-index space, subpixel accurate
            row, col = c[:, 0], c[:, 1]
            x = xmin + (col - 1 + 0.5) * pixel_size
            y = ymin + (row - 1 + 0.5) * pixel_size
            poly_xy = np.column_stack([x, y])
            area = abs(polygon_areanp(poly_xy))
            if area > best_area:
                best_area = area
                best = poly_xy
 
        # find_contours closes the loop by repeating the first point --
        # drop the duplicate to match the open-ring convention used for
        # `polygons` elsewhere in this module.
        if len(best) > 1 and np.allclose(best[0], best[-1]):
            best = best[:-1]
 
        if simplify_tolerance > 0 and len(best) > 3:
            best = approximate_polygon(best, tolerance=simplify_tolerance)
            if len(best) > 1 and np.allclose(best[0], best[-1]):
                best = best[:-1]
 
        result_polygons.append(best)
 
    if n_empty_norasterize or n_empty_swallowed:
        import warnings
        warnings.warn(
            f"resolve_disputed_pixels_by_proximity: {n_empty_norasterize} region(s) "
            f"never rasterised at resolution={resolution} (too small relative to the "
            f"bounding box -- try raising `resolution`), and {n_empty_swallowed} "
            f"region(s) lost every pixel to higher-priority neighbours (fully "
            f"swallowed -- check `region_order` / overlap amount). All of these were "
            f"kept as their ORIGINAL (pre-resolution) polygon instead of being "
            f"dropped, so region count and `polygon_areanp(p)` stay well-defined for "
            f"every entry, but they were NOT actually disputed-pixel-resolved.",
            stacklevel=2,
        )
 
    if return_raster:
        raster_info = {
            "label_grid": label_grid,
            "coverage_count": coverage_count,
            "extent": (xmin, ymin, xmax, ymax),
            "pixel_size": pixel_size,
        }
        return result_polygons, raster_info
 
    return result_polygons