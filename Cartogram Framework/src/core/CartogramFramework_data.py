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
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours, approximate_polygon

import numpy as np
import cvxpy as cp
from typing import Optional
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from matplotlib.path import Path

from src.geometry import *
from src.utils import *
from src.core import *
from src.core.preprocessing_data import *

_ANCHOR_EPS = 1e-4

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
    #   "contiguous"   — original shape + shared-vertex coupling (soft)
    #   "contiguous2"  — same as "contiguous" (kept as an accepted alias)

    target_centers=None,
    horizontal_pairs=None,            # list[(i,j)] i left of j
    vertical_pairs=None,              # list[(i,j)] i below j
    neighboring_pairs=None,           # list[(i,j)] geographically adjacent
    shared_vertices_of_neighbors=None,
    # list of [i, j, shared_points] from shared_vertices_of_neighbors()
    shared_vertex_tolerance: float = 1e-8,
    lambda_shape: Optional[float]    = None,  # λ_s -> W_shape,    else (1 - shape_deformation)
    lambda_area: Optional[float]     = None,  # λ_a -> W_area,     else cartographic_error
    lambda_center: Optional[float]   = None,  # λ_c -> W_spatial,  else spatial_deformation
    lambda_topology: Optional[float] = None,  # λ_t -> W_topology, else topological_accuracy

    # --- contiguous-mode parameters ---
    lambda_contiguous: float = 1.0,
    contiguous_area_ratio_cap: float = 10.0,
    contiguous_use_topology: bool = True,
    lambda_order: float = 1e6,
    # --- inner shape parameters ---
    gamma: float = 1.0,   
    beta: float  = 1.0, 
    # --- geometry parameters ---
    alpha:   float = 1.0,   # slope for diagonal deviation d_ij
    epsilon: float = 1e-2,  # gap for non-adjacent pairs
    b:       float = 1e-2,  # scale for directional weight b_ij
    t_min:   float = 0.01,  # minimum radial scale
    t_max:   float = 5.0,   # maximum radial scale  (caps runaway vertices)
    # --- quality-criterion -------------------------------------
    cartographic_error: float = 1.0,    # <-> lambda_area
    spatial_deformation: float = 0.0,   # <-> lambda_center
    topological_accuracy: float = 1.0,  # <-> lambda_topology
    shape_deformation: float = 0.0,
    relative_direction: float = 1.0,
    global_shape: float = 1.0,
    local_shape: float = 0.0,
    complexity: float = 1.0,
    data_ink_ratio: float = 1.0,

    # ── Area scaling for non-contiguous ────────────────────────────────────
    area_scale: float = 1.0,
    
    compute_leaders_flag: bool = True,
    leader_tol: float = 1e-3,

    # ── Approach 1: soft mean-scale (Fix 1a — always applied) ──────────────
    lambda_mean_scale: float = 1e4,
    # ── Approach 2: pairwise centre repulsion (anti-overlap) ───────────────
    lambda_repulsion: float = 0.0,
   
    repulsion_only_adjacent: bool = False,
    repulsion_margin: float = 1.0,
    repulsion_margin_mode: str = "fixed",

    # ── Approach 4: tangential vertex freedom + vertex-level anti-overlap ──
    tangential_freedom: bool = False,
    
    tangential_margin: float = 0.5,
    lambda_tangential: float = 0.0,
    lambda_vertex_repulsion: float = 0.0,
    vertex_repulsion_only_adjacent: bool = True,
    vertex_repulsion_margin: float = 0.0,
    vertex_repulsion_nearest_k: Optional[int] = None,
    
    vertex_repulsion_boundary_k: Optional[int] = None,
    
    max_vertex_repulsion_pairs: int = 20000,

    # ── Approach 3: vertex–centroid distance penalty ────────────────────────
    lambda_vertex_distance: float = 0.0,

    vertex_distance_margin: float = 1.5,

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
    
    if area_scale != 1.0:
        target_areas = [a * area_scale for a in target_areas]

    n_regions = len(polygons)

    # The only remaining possible source of infeasibility after Fix 1: a
    # per-vertex box bound with an empty range. Validate explicitly and
    # raise a clear error rather than letting the solver report an opaque
    # "infeasible" status.
    if t_min > t_max:
        raise ValueError(
            f"t_min ({t_min}) must be <= t_max ({t_max}); as given, the "
            f"per-vertex scale box [t_min, t_max] is empty and no solution "
            f"can exist."
        )

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
            init_polygons.append(make_square(ctr, target_areas[k], n=4))
        else:
            init_polygons.append(data[name]["polygon"])

    # ------------------------------------------------------------------
    # 2b. Fix 4a: corrected repulsion helper — single one-sided hinge on a
    #     FIXED separating axis/side chosen from the pair's ORIGINAL
    #     relative position, instead of two independent (buggy) hinges.
    #     True direction-agnostic "keep >= margin apart" is non-convex, so
    #     this convex surrogate commits in advance to whichever axis has
    #     the larger original gap and only enforces separation there.
    # ------------------------------------------------------------------
    def _one_sided_repulsion(p_ref_i, p_ref_j, expr_i, expr_j, margin, tag=""):
        gap = np.asarray(p_ref_j, dtype=float) - np.asarray(p_ref_i, dtype=float)
        axis = 0 if abs(gap[0]) >= abs(gap[1]) else 1
        sign = 1.0 if gap[axis] >= 0 else -1.0
        d = expr_j[axis] - expr_i[axis]              # affine in cp variables
        slack = cp.Variable(nonneg=True, name=f"rep_{tag}")
        constraints.append(slack >= margin - sign * d)
        return cp.square(slack)

    # ------------------------------------------------------------------
    # 2c. Precompute which vertex indices are "locked" (shared with a
    #     neighbouring region) vs "free" (facing the void/open background)
    #     per region. Needed before building per-vertex variables below,
    #     since only free vertices get the extra tangential freedom (u).
    # ------------------------------------------------------------------
    locked_indices: list = [set() for _ in region_names]
    _n_shared_pts_total = 0
    _n_shared_pts_matched = 0
    if shared_vertices_of_neighbors is not None:
        for entry in shared_vertices_of_neighbors:
            si, sj, shared_pts = entry[0], entry[1], entry[2]
            _n_shared_pts_total += len(shared_pts)
            poly_si = np.asarray(init_polygons[si])
            poly_sj = np.asarray(init_polygons[sj])
            matched = find_shared_vertex_indices(poly_si, poly_sj, shared_pts, shared_vertex_tolerance)
            _n_shared_pts_matched += len(matched)
            for (k, l) in matched:
                locked_indices[si].add(k)
                locked_indices[sj].add(l)
        if _n_shared_pts_total > 0 and _n_shared_pts_matched < 0.9 * _n_shared_pts_total:
            if False:
                print(
                    f"  WARNING: only matched {_n_shared_pts_matched}/{_n_shared_pts_total} "
                    f"shared-point entries to actual vertex indices (shared_vertex_tolerance="
                    f"{shared_vertex_tolerance}). A low match rate usually means "
                    f"shared_vertex_tolerance is too tight for your data's coordinate "
                    f"precision -- unmatched vertices are silently treated as FREE "
                    f"(void-facing) instead of locked, which breaks both contiguous "
                    f"coupling and correct tangential-freedom targeting. Try loosening "
                    f"shared_vertex_tolerance (and the tolerance used when building "
                    f"shared_vertices_of_neighbors itself)."
                )


    # ------------------------------------------------------------------
    # 3. Per-region CVXPY variables and constraints
    # ------------------------------------------------------------------
    constraints = []
    centers_vars: list[cp.Expression] = []
    new_polygon_exprs: list[cp.Expression] = []

    shape_terms  = []
    area_terms   = []
    center_terms = []
    anchor_terms = []   # Fix 2: always-on, ungated identity-tiebreak penalty
    tangential_terms = []  # Fix 4: lambda_tangential-weighted regularisation on u

    # Store per-region data needed for leader computation and contiguous coupling
    r_vars: list = []          # cp.Variable or None
    s_vals: list[float] = []   # target scales
    # For contiguous coupling we also store (t_var, directions, fp) per region
    _t_vars:      list = []    # cp.Variable(ni) or None
    _u_vars:      list = []    # cp.Variable(ni) or None  (Fix 4: tangential freedom)
    _directions:  list = []    # np.ndarray (ni,2) or None
    _tangents:    list = []    # np.ndarray (ni,2) or None  (Fix 4: 90°-rotated axis)
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
            _u_vars.append(None)
            _directions.append(None)
            _tangents.append(None)
            # Anchor the otherwise-unconstrained dummy centre to fp so it
            # doesn't drift arbitrarily (Fix 2, degenerate-polygon case).
            anchor_terms.append(cp.sum_squares(dummy_center - fp))
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
        # Tangential axis: `directions` rotated 90° (fixed, from the
        # ORIGINAL geometry only — keeps the position expression below
        # affine in the cp variables and fully DCP-compliant).
        tangents = np.stack([-directions[:, 1], directions[:, 0]], axis=1)

        
        u = cp.Variable(ni)
        if tangential_freedom:
            prev_d = np.linalg.norm(poly - np.roll(poly, 1, axis=0), axis=1)
            next_d = np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)
            local_scale = 0.5 * (prev_d + next_d)          # (ni,) constants
            u_bound = tangential_margin * local_scale       # (ni,) box half-width
            for k in range(ni):
                if k in locked_indices[i]:
                    constraints.append(u[k] == 0)
                else:
                    constraints += [u[k] >= -u_bound[k], u[k] <= u_bound[k]]
            anchor_terms.append(cp.sum_squares(u))           # Fix 2: pulls u -> 0 by default
            if lambda_tangential > 0.0:
                tangential_terms.append(cp.sum_squares(u))
        else:
            constraints.append(u == 0)

        # New vertices  v'_ik = m_i + t_ik * (v_ik - c_i) + u_ik * tangent_ik
        new_pts = (
            m[None, :]
            + cp.multiply(t[:, None], directions)
            + cp.multiply(u[:, None], tangents)
        )
        new_polygon_exprs.append(new_pts)

        # Store for contiguous coupling
        _t_vars.append(t)
        _u_vars.append(u)
        _directions.append(directions)
        _tangents.append(tangents)

       
        constraints += [
            t >= t_min,
            t <= t_max,
            z >= t - 1,
            z >= -(t - 1),
        ]
        mean_scale_penalty = lambda_mean_scale * cp.square(cp.sum(t) / ni - s)

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

   
        anchor_terms.append(cp.sum_squares(t - 1) + cp.sum_squares(delta))

    # ------------------------------------------------------------------
    # 4. Pairwise topology constraints and objective terms
    # ------------------------------------------------------------------
    pairwise_terms = []
    order_terms    = []   # Fix 1b: soft hinge penalties for hor/ver ordering

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

  
        all_pairs_seen: set[tuple[int, int]] = set()
        if horizontal_pairs is not None:
            for (i, j) in horizontal_pairs:
                key = (min(i, j), max(i, j))
                if key not in all_pairs_seen:
                    ta_i = target_areas[i];  ta_j = target_areas[j]
                    w = (np.sqrt(ta_i) + np.sqrt(ta_j)) / 2.0
                    g = gap_ij(i, j)
                    xi = centers_vars[i][0];  xj = centers_vars[j][0]
                    slack_h = cp.Variable(nonneg=True, name=f"ordh_{i}_{j}")
                    constraints.append(slack_h >= (w + g) - (xj - xi))
                    order_terms.append(cp.square(slack_h))
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
                    slack_v = cp.Variable(nonneg=True, name=f"ordv_{i}_{j}")
                    constraints.append(slack_v >= (w + g) - (yj - yi))
                    order_terms.append(cp.square(slack_v))
                    all_pairs_seen_v.add(key)

    topology_term = cp.sum(pairwise_terms) if pairwise_terms else cp.Constant(0.0)
    order_penalty = (
        lambda_order * cp.sum(order_terms) if order_terms else cp.Constant(0.0)
    )

    # ------------------------------------------------------------------
    # 4b-pre. Approach 2: pairwise centre repulsion (anti-spike / anti-overlap)
    # ------------------------------------------------------------------
 
    repulsion_terms = []
    if lambda_repulsion > 0.0:
        pairs_for_repulsion = []
        if repulsion_only_adjacent:
            pairs_for_repulsion = list(adjacent_set)
        else:
            for ii in range(n_regions):
                for jj in range(ii + 1, n_regions):
                    pairs_for_repulsion.append((ii, jj))

        if repulsion_margin_mode not in ("fixed", "circle_radius"):
            raise ValueError(
                f"repulsion_margin_mode must be 'fixed' or 'circle_radius', "
                f"got {repulsion_margin_mode!r}"
            )

        for (ii, jj) in pairs_for_repulsion:
            if centers_vars[ii] is None or centers_vars[jj] is None:
                continue
            if repulsion_margin_mode == "circle_radius":
                r_i = float(np.sqrt(target_areas[ii] / np.pi))
                r_j = float(np.sqrt(target_areas[jj] / np.pi))
                pair_margin = r_i + r_j
            else:
                pair_margin = repulsion_margin
            repulsion_terms.append(
                _one_sided_repulsion(
                    _fp_vals[ii], _fp_vals[jj],
                    centers_vars[ii], centers_vars[jj],
                    pair_margin, tag=f"c_{ii}_{jj}",
                )
            )

    repulsion_penalty = (
        lambda_repulsion * cp.sum(repulsion_terms)
        if repulsion_terms else cp.Constant(0.0)
    )

    # ------------------------------------------------------------------
    # 4b-pre-2. Approach 4: vertex-level anti-overlap repulsion
    # ------------------------------------------------------------------

    vertex_repulsion_terms = []
    if lambda_vertex_repulsion > 0.0:
        if vertex_repulsion_only_adjacent:
            region_pairs_for_vr = list(adjacent_set)
        else:
            region_pairs_for_vr = [
                (ii, jj) for ii in range(n_regions) for jj in range(ii + 1, n_regions)
            ]

        # Precompute free-vertex index lists and their ORIGINAL positions
        # once per region (cheap, fixed numpy — not part of the QP), and
        # figure out which (i,k)-(j,l) pairs we're actually going to use
        # BEFORE creating a single cp.Variable, so we can size-check first.
        free_idx  = {}   # region idx -> list[int] of free vertex indices
        free_pos  = {}   # region idx -> (m,2) ndarray of original positions
        for ii, jj in region_pairs_for_vr:
            for r in (ii, jj):
                if r in free_idx or _t_vars[r] is None:
                    continue
                fidx = [k for k in range(len(_directions[r])) if k not in locked_indices[r]]
                free_idx[r] = fidx
                free_pos[r] = _fp_vals[r] + _directions[r][fidx] if fidx else np.zeros((0, 2))

        # Build the actual pair list: first narrow each side down to only
        # the vertices actually near THIS SPECIFIC neighbour's shared
        # border (vertex_repulsion_boundary_k), then apply nearest-k
        # pairing among that narrowed set (vertex_repulsion_nearest_k).
        vr_pairs = []   # list of (ii, k, jj, l)
        for (ii, jj) in region_pairs_for_vr:
            if _t_vars[ii] is None or _t_vars[jj] is None:
                continue
            fi, fj = free_idx.get(ii, []), free_idx.get(jj, [])
            if not fi or not fj:
                continue
            pos_i, pos_j = free_pos[ii], free_pos[jj]

            # Boundary narrowing: keep only each side's vertices closest to
            # the OTHER region's centroid — the ones actually near this
            # specific shared border, dropping ones on the far side of a
            # large country that this neighbour was never going to touch.
            if vertex_repulsion_boundary_k is not None:
                ctr_j = _fp_vals[jj]
                bi = min(vertex_repulsion_boundary_k, len(fi))
                keep_i = np.argsort(np.sum((pos_i - ctr_j) ** 2, axis=1))[:bi]
                fi_use, pos_i_use = [fi[a] for a in keep_i], pos_i[keep_i]

                ctr_i = _fp_vals[ii]
                bj = min(vertex_repulsion_boundary_k, len(fj))
                keep_j = np.argsort(np.sum((pos_j - ctr_i) ** 2, axis=1))[:bj]
                fj_use, pos_j_use = [fj[b] for b in keep_j], pos_j[keep_j]
            else:
                fi_use, pos_i_use = fi, pos_i
                fj_use, pos_j_use = fj, pos_j

            if vertex_repulsion_nearest_k is None:
                for k in fi_use:
                    for l in fj_use:
                        vr_pairs.append((ii, k, jj, l))
            else:
                # For each (narrowed) free vertex of i, keep only its k
                # nearest (narrowed) free vertices of j (by original
                # position) — the ones that can plausibly overlap.
                kk = min(vertex_repulsion_nearest_k, len(fj_use))
                for a, k in enumerate(fi_use):
                    d2 = np.sum((pos_j_use - pos_i_use[a]) ** 2, axis=1)
                    nearest = np.argsort(d2)[:kk]
                    for b in nearest:
                        vr_pairs.append((ii, k, jj, fj_use[b]))

        # Fail fast with a clear message instead of silently building a QP
        # large enough to hang or crash the kernel/notebook.
        if len(vr_pairs) > max_vertex_repulsion_pairs:
            raise ValueError(
                f"lambda_vertex_repulsion would add {len(vr_pairs)} free-vertex "
                f"pairs (> max_vertex_repulsion_pairs={max_vertex_repulsion_pairs}), "
                f"which is very likely to hang or crash your kernel/notebook rather "
                f"than solve. Try one or more of:\n"
                f"    • set vertex_repulsion_boundary_k (e.g. 10-20) to first narrow "
                f"each region down to only the vertices near THIS neighbour's shared "
                f"border\n"
                f"    • set vertex_repulsion_nearest_k (e.g. 3-8) to only pair each "
                f"free vertex with its nearest few on the other region\n"
                f"    • keep vertex_repulsion_only_adjacent=True (already set)\n"
                f"    • simplify your polygons (fewer vertices per region) before "
                f"calling this\n"
                f"    • or, if you're confident your machine can handle it, raise "
                f"max_vertex_repulsion_pairs explicitly"
            )

        pair_terms: dict = {}
        for (ii, k, jj, l) in vr_pairs:
            v_ik = (
                centers_vars[ii]
                + _t_vars[ii][k] * _directions[ii][k]
                + _u_vars[ii][k] * _tangents[ii][k]
            )
            p_ik = _fp_vals[ii] + _directions[ii][k]   # original position (numpy)
            v_jl = (
                centers_vars[jj]
                + _t_vars[jj][l] * _directions[jj][l]
                + _u_vars[jj][l] * _tangents[jj][l]
            )
            p_jl = _fp_vals[jj] + _directions[jj][l]   # original position (numpy)
            term = _one_sided_repulsion(
                p_ik, p_jl, v_ik, v_jl,
                vertex_repulsion_margin, tag=f"v_{ii}_{k}_{jj}_{l}",
            )
            pair_terms.setdefault((ii, jj), []).append(term)

        for (ii, jj), terms in pair_terms.items():
            vertex_repulsion_terms.append(cp.sum(terms) / len(terms))   # mean, not sum

    vertex_repulsion_penalty = (
        lambda_vertex_repulsion * cp.sum(vertex_repulsion_terms)
        if vertex_repulsion_terms else cp.Constant(0.0)
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
                poly_i, poly_j, shared_pts, shared_vertex_tolerance
            )

            for (k, l) in kl_pairs:

                v_i = (
                    centers_vars[i]
                    + _t_vars[i][k] * _directions[i][k]
                    + _u_vars[i][k] * _tangents[i][k]
                )  # (2,) expr — u is pinned to 0 here (locked/shared vertex)
                v_j = (
                    centers_vars[j]
                    + _t_vars[j][l] * _directions[j][l]
                    + _u_vars[j][l] * _tangents[j][l]
                )  # (2,) expr

                diff = v_i - v_j                          # (2,) affine expr
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

    if False:
        print(f"Contiguous mode '{shape}': {n_locked} shared-vertex "
            f"{'constraints' if shape == 'contiguous' else 'penalty terms'} added.")

    # ------------------------------------------------------------------
    # 5. Global objective  (eq. 16)
    # ------------------------------------------------------------------
    # anchor_penalty (Fix 2) is deliberately added OUTSIDE every W_*/lambda_*
    # factor, multiplied only by the hardcoded _ANCHOR_EPS. That's what makes
    # it survive "all weights off" and pin the unique minimiser to t=1,
    # delta=0 (i.e. the input geometry) in that case, while being negligible
    # (1e-9 scale) whenever any real weight is doing actual work.
    anchor_penalty = cp.sum(anchor_terms) if anchor_terms else cp.Constant(0.0)

    tangential_penalty = cp.sum(tangential_terms) if tangential_terms else cp.Constant(0.0)

    objective = cp.Minimize(
        W_shape    * cp.sum(shape_terms)
        + W_area * cp.sum(area_terms)
        + W_spatial * cp.sum(center_terms)
        + W_topology * topology_term
        + order_penalty                             # Fix 1b: soft hor/ver ordering
        + lambda_contiguous * contiguous_penalty
        + repulsion_penalty                         # Approach 2 (Fix 4a: corrected)
        + vertex_repulsion_penalty                  # Approach 4: vertex-level anti-overlap
        + tangential_penalty                        # Fix 4: lambda_tangential regularisation
        + _ANCHOR_EPS * anchor_penalty               # Fix 2: always-on identity tiebreak
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
            f"  Note: after Fix 1, every constraint is a box bound or a "
            f"one-sided slack (never a hard equality), so this status should "
            f"only occur from solver numerical/timeout issues, not structural "
            f"infeasibility. If it persists, try:\n"
            f"    • Lowering lambda_order / lambda_contiguous / lambda_mean_scale "
            f"(very large weights can make the QP numerically stiff)\n"
            f"    • Loosening t_min/t_max"
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
    qp_areas = [polygon_areanp(p) for p in new_polygons]

    overlap_area_lost = [0.0] * n_regions

    # ------------------------------------------------------------------
    # 9. Populate the data dict — single source of truth for results.
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
        record["qp_area"] = float(qp_areas[idx])
        record["target_area"] = float(target_areas[idx])
        record["new_centroid"] = np.mean(solved_polygon, axis=0)
        record["overlap_area_lost"] = float(overlap_area_lost[idx])
        record["status"] = prob.status
        record["objective_value"] = prob.value
        record["constraints"] = constraints
        record["objective"] = objective


    # data["__meta__"] = {
    #     "status": prob.status,
    #     "objective_value": prob.value,
    #     "leaders": leaders,
    #     "target_areas": target_areas,
    #     "n_regions": n_regions,
    # }

    return data