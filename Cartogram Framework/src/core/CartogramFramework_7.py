"""
CartogramFramework_global
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
import os
import sys
from scipy.spatial import KDTree
from skimage.measure import find_contours
from matplotlib.path import Path

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.geometry import *
from src.utils import *
from src.core import *
from src.core.preprocessing import *

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

    contiguous_skip_hard_ordering_for_adjacent: bool = True,
    # contiguous / contiguous2 only. The hard H/V ordering constraints
    # (xj-xi >= w+g, yj-yi >= w+g) enforce a MINIMUM centre-to-centre
    # separation based on combined region "widths" — a sensible hard
    # floor for non-touching regions (Demers/Dorling, or non-adjacent
    # pairs even in contiguous mode), but for a pair that is ALSO in
    # neighboring_pairs (T), gap_ij already returns g=0 and the two
    # regions' shared vertices are separately being pulled into exact
    # coincidence by the contiguous locking constraint — meaning this
    # hard minimum-separation constraint is fighting the hard locking
    # constraint for centre positions that "touching, possibly oddly-
    # shaped" regions frequently cannot satisfy simultaneously. This is
    # the actual, structural cause of most contiguous+topology
    # infeasibility (NOT something tangential freedom can fix, since
    # both conflicting constraints live on centres_vars, not on
    # per-vertex t/u at all).
    #
    # True (default): skip the hard H/V ordering constraint specifically
    # for pairs that are ALSO adjacent (in neighboring_pairs) — their
    # relative ordering is already implicitly preserved by the shared
    # boundary itself (two regions locked together at a shared edge
    # cannot swap left/right or top/bottom without a self-intersection,
    # which t_min/t_max and the mean-scale constraint already guard
    # against). The hard ordering constraint still applies in full to
    # every NON-adjacent pair, where nothing else is holding order.
    # False: reverts to the original behaviour (hard ordering applied to
    # every listed H/V pair regardless of adjacency).

    soft_hv_ordering: bool = False,
    # When True, converts the REMAINING hard H/V ordering constraints
    # (i.e. the ones NOT already skipped by
    # contiguous_skip_hard_ordering_for_adjacent — so, non-adjacent
    # pairs) from hard inequalities into a soft squared-hinge penalty in
    # the objective instead. Guarantees this constraint category can
    # never, by itself, make the problem infeasible.
    #
    # A hard H/V ordering system is a set of "difference constraints"
    # (x_j - x_i >= c_ij); such a system is infeasible if and only if
    # its constraint graph contains a cycle whose accumulated constants
    # cannot be satisfied simultaneously — see check_hv_ordering_feasibility()
    # below, which detects this directly and tells you exactly which
    # regions are involved. That is a property of horizontal_pairs /
    # vertical_pairs themselves (how you generated them), not of
    # anything else in this framework. Turning soft_hv_ordering on papers
    # over that (the solve will succeed, but will silently violate
    # whichever orderings are cheapest to violate) rather than fixing
    # the actual inconsistency — prefer running the diagnostic first.
    lambda_hv_ordering: float = 1.0,
    # Weight on the soft H/V ordering penalty (only used if
    # soft_hv_ordering=True).

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

    # ── Approach 4: tangential vertex freedom + boundary anchoring ─────────
    # Every vertex currently only slides along ONE fixed ray (direction
    # d_i,k = original vertex minus region centre) via its radial scale
    # t_i,k. When a shared-vertex lock (contiguous mode) needs to place
    # that vertex somewhere NOT on that ray, the only way to get close is
    # a large excursion along the wrong-direction ray — a spike. Adding a
    # second, fixed, PERPENDICULAR direction e_i,k gives every vertex a
    # genuine 2-D reach while staying perfectly affine (still DCP-safe):
    #
    #   v'_i,k = m_i + t_i,k * d_i,k + u_i,k * e_i,k,   e_i,k = rot90(d_i,k)
    #
    # This new freedom is only SAFE for vertices that have a neighbour
    # constraining them. Vertices with no shared-vertex partner ("boundary"
    # vertices — facing the sea/void, not another region) get an extra
    # soft penalty pulling them back toward pure uniform target-scaling
    # (t_i,k = s_i, u_i,k = 0) — i.e. away from spiking into empty space,
    # NOT away from moving: a whole region shrinking/growing uniformly
    # (all its vertices at t = s_i) costs this term nothing at all,
    # however large lambda_anchor is set. Only non-uniform, single-vertex
    # distortion at an unconstrained boundary vertex is discouraged.
    lambda_angle: float = 0.05,
    # Weight on the tangential-deformation energy E^angle (parallel to the
    # existing radial deformation term, but for u instead of t). Kept
    # small-but-nonzero by default (not exactly 0) so u stays well-posed
    # even for vertices where nothing else in the objective touches it —
    # otherwise those u's would be free floats within their box bound.
    lambda_anchor: float = 0.0,
    # Weight on the boundary-anchoring energy E^anchor. 0 (default) =
    # feature fully disabled, identical to the framework before this
    # addition. Only applies to vertices with NO shared-vertex partner.
    # Does not resist a region's own uniform area-driven resizing — only
    # resists an individual boundary vertex distorting non-uniformly
    # relative to the rest of its own region.
    u_max: float = 0.15,
    # Symmetric box bound on the tangential offset u_i,k (bounds are
    # [-u_max, u_max]). Start conservative: too large risks a vertex
    # sliding sideways past its own polygon neighbour (a new, local
    # self-intersection failure mode that didn't exist before this
    # feature). 0.15 means at most ~15% of the vertex's own d_i,k length.
    beta_u: Optional[float] = None,
    # L1/L2 blend for the tangential term, same meaning as `beta` but for
    # u instead of t. None (default) reuses `beta`. L1 (=1.0) tends to
    # keep most u_i,k exactly 0 — only vertices that actually need
    # tangential movement to satisfy a shared-vertex lock "spend" any.


    postprocess_resolve_overlaps: bool = False,
    data=None,
    # When True, run the overlap-resolution pass BEFORE hole-filling.
    # Each overlapping region is detected and clipped so the overlapping
    # portion is split along the midline and given to the polygon whose
    # centroid is nearest — preserving total area.  The area "lost" by
    # each polygon during this pass is recorded and used as a priority
    # weight in the subsequent hole-filling step (polygons that lost more
    # to overlap get proportionally more of any adjacent hole area).
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
    # 2b. Boundary-vertex detection (for tangential freedom / anchoring)
    # ------------------------------------------------------------------
    # A vertex is "boundary" (facing the sea/void) if it does NOT appear
    # in any shared-vertex pair with any neighbour. These are exactly the
    # vertices with no locking requirement to justify tangential/radial
    # deviation from uniform scaling — so they're the ones that get the
    # extra anchoring penalty below. Uses the same coordinate-matching
    # logic as Section 4b's contiguous coupling, run once up front here.
    boundary_mask = [np.ones(len(p), dtype=bool) for p in init_polygons]
    if shared_vertices_of_neighbors is not None:
        for entry in shared_vertices_of_neighbors:
            b_i, b_j, b_shared_pts = entry[0], entry[1], entry[2]
            if b_i >= len(init_polygons) or b_j >= len(init_polygons):
                continue
            b_poly_i = np.asarray(init_polygons[b_i])
            b_poly_j = np.asarray(init_polygons[b_j])
            b_kl_pairs = find_shared_vertex_indices(b_poly_i, b_poly_j, b_shared_pts)
            for (b_k, b_l) in b_kl_pairs:
                if b_k < len(boundary_mask[b_i]):
                    boundary_mask[b_i][b_k] = False
                if b_l < len(boundary_mask[b_j]):
                    boundary_mask[b_j][b_l] = False

    beta_u_eff = beta if beta_u is None else beta_u

    # ------------------------------------------------------------------
    # 3. Per-region CVXPY variables and constraints
    # ------------------------------------------------------------------
    constraints = []
    centers_vars: list[cp.Expression] = []
    new_polygon_exprs: list[cp.Expression] = []

    shape_terms  = []
    area_terms   = []
    center_terms = []
    angle_terms  = []   # tangential deformation energy E^angle_i (Approach 4)
    anchor_terms = []   # boundary-anchoring energy E^anchor_i (Approach 4)

    # Store per-region data needed for leader computation and contiguous coupling
    r_vars: list = []          # cp.Variable or None
    s_vals: list[float] = []   # target scales
    # For contiguous coupling we also store (t_var, directions, fp) per region
    _t_vars:      list = []    # cp.Variable(ni) or None
    _directions:  list = []    # np.ndarray (ni,2) or None
    _u_vars:      list = []    # cp.Variable(ni) or None  (tangential scale)
    _e_directions: list = []   # np.ndarray (ni,2) or None (tangential direction)
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
            _u_vars.append(None)
            _e_directions.append(None)
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

        # ── Approach 4: tangential scale u_i,k along the perpendicular
        # direction e_i,k = rot90(d_i,k). Constant like directions, so
        # this stays affine. rot90(x,y) = (-y, x).
        e_directions = np.stack(
            [-directions[:, 1], directions[:, 0]], axis=1
        )                                 # (ni, 2), same norm as directions
        u = cp.Variable(ni)
        w = cp.Variable(ni, nonneg=True)   # L1 aux: w >= |u|
        constraints += [
            u >= -u_max,
            u <= u_max,
            w >= u,
            w >= -u,
        ]

        # New vertices  v'_ik = m_i + t_ik*(v_ik - c_i) + u_ik*e_ik
        # Shape: (ni, 2)
        new_pts = (
            m[None, :]
            + cp.multiply(t[:, None], directions)
            + cp.multiply(u[:, None], e_directions)
        )
        new_polygon_exprs.append(new_pts)

        # Store for contiguous coupling
        _t_vars.append(t)
        _directions.append(directions)
        _u_vars.append(u)
        _e_directions.append(e_directions)

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

        # ---- Approach 4: tangential deformation energy E^angle_i -------
        angle_term = beta_u_eff * cp.sum(w) + (1.0 - beta_u_eff) * cp.sum_squares(u)
        angle_terms.append(angle_term)

        # ---- Approach 4: boundary-anchoring energy E^anchor_i ----------
        # Only over vertices with no shared-vertex partner. Penalises
        # deviation from the pure homothetic image (t=s_i, u=0); costs
        # nothing for a region scaling uniformly, regardless of lambda_anchor.
        b_mask = boundary_mask[i]
        if lambda_anchor > 0.0 and np.any(b_mask):
            b_idx = np.where(b_mask)[0]                      # integer indices (safer for cvxpy indexing)
            d_norms_sq_i = np.sum(directions ** 2, axis=1)   # (ni,) constants
            anchor_term = cp.sum(
                cp.multiply(
                    d_norms_sq_i[b_idx],
                    cp.square(t[b_idx] - s) + cp.square(u[b_idx]),
                )
            )
        else:
            anchor_term = cp.Constant(0.0)
        anchor_terms.append(anchor_term)

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
    hv_ordering_terms = []   # populated only if include_topology and soft_hv_ordering

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

        # Hard ordering constraints for H and V pairs (eqs. 11–12),
        # or soft squared-hinge penalties if soft_hv_ordering=True.
        skip_adjacent_ordering = is_contiguous and contiguous_skip_hard_ordering_for_adjacent
        n_ordering_skipped = 0
        all_pairs_seen: set[tuple[int, int]] = set()
        if horizontal_pairs is not None:
            for (i, j) in horizontal_pairs:
                key = (min(i, j), max(i, j))
                if key not in all_pairs_seen:
                    if skip_adjacent_ordering and is_adjacent(i, j):
                        all_pairs_seen.add(key)
                        n_ordering_skipped += 1
                        continue
                    ta_i = target_areas[i];  ta_j = target_areas[j]
                    w = (np.sqrt(ta_i) + np.sqrt(ta_j)) / 2.0
                    g = gap_ij(i, j)
                    xi = centers_vars[i][0];  xj = centers_vars[j][0]
                    if soft_hv_ordering:
                        hv_ordering_terms.append(cp.square(cp.pos((w + g) - (xj - xi))))
                    else:
                        constraints.append(xj - xi >= w + g)
                    all_pairs_seen.add(key)

        all_pairs_seen_v: set[tuple[int, int]] = set()
        if vertical_pairs is not None:
            for (i, j) in vertical_pairs:
                key = (min(i, j), max(i, j))
                if key not in all_pairs_seen_v:
                    if skip_adjacent_ordering and is_adjacent(i, j):
                        all_pairs_seen_v.add(key)
                        n_ordering_skipped += 1
                        continue
                    ta_i = target_areas[i];  ta_j = target_areas[j]
                    w = (np.sqrt(ta_i) + np.sqrt(ta_j)) / 2.0
                    g = gap_ij(i, j)
                    yi = centers_vars[i][1];  yj = centers_vars[j][1]
                    if soft_hv_ordering:
                        hv_ordering_terms.append(cp.square(cp.pos((w + g) - (yj - yi))))
                    else:
                        constraints.append(yj - yi >= w + g)
                    all_pairs_seen_v.add(key)

        if skip_adjacent_ordering and n_ordering_skipped > 0:
            print(f"  contiguous_skip_hard_ordering_for_adjacent: skipped "
                  f"{n_ordering_skipped} hard H/V constraint(s) for pairs "
                  f"that are also locked via shared-vertex contiguity.")
        if soft_hv_ordering and hv_ordering_terms:
            print(f"  soft_hv_ordering: {len(hv_ordering_terms)} H/V ordering "
                  f"constraint(s) converted to soft penalties (weight={lambda_hv_ordering}).")

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
            if _u_vars[i] is None or _u_vars[j] is None:
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

                v_i = (
                    centers_vars[i]
                    + _t_vars[i][k] * _directions[i][k]
                    + _u_vars[i][k] * _e_directions[i][k]
                )  # (2,) expr
                v_j = (
                    centers_vars[j]
                    + _t_vars[j][l] * _directions[j][l]
                    + _u_vars[j][l] * _e_directions[j][l]
                )  # (2,) expr

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
        + lambda_angle  * (cp.sum(angle_terms) if angle_terms else cp.Constant(0.0))   # Approach 4
        + lambda_anchor * (cp.sum(anchor_terms) if anchor_terms else cp.Constant(0.0)) # Approach 4
        + (lambda_hv_ordering * cp.sum(hv_ordering_terms) if (soft_hv_ordering and hv_ordering_terms) else cp.Constant(0.0))
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

        if isinstance(data, dict):
            for idx, name in enumerate(data.keys()):
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
              f"(method='{postprocess_snap_method}')")
        new_polygons = close_contiguous_gaps(
            polygons                          = new_polygons,
            shared_vertices_of_neighbors_data = shared_vertices_of_neighbors,
            original_polygons                 = init_polygons,
            method                            = postprocess_snap_method,
        )

        actual_areas = [polygon_areanp(p) for p in new_polygons]

    if isinstance(data, dict):
        for idx, name in enumerate(data.keys()):
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
            record["status"] = prob.status
            record["objective_value"] = prob.value

    # ------------------------------------------------------------------
    # 10. Gap / overlap post-processing — unified pipeline
    #
    #  Order matters:
    #    a) resolve_overlaps  (if requested) — split overlap areas, record losses
    # ------------------------------------------------------------------
    overlap_area_lost = [0.0] * n_regions  # per-region area lost to overlap clipping

    if postprocess_resolve_overlaps:
        print("Post-processing: resolving polygon overlaps …")
        new_polygons, overlap_area_lost = resolve_polygon_overlaps(new_polygons)
        actual_areas = [polygon_areanp(p) for p in new_polygons]
        total_lost = sum(overlap_area_lost)
        if total_lost > 0:
            print(f"  Total area redistributed from overlaps: {total_lost:.4g}")


    return new_polygons, actual_areas, prob.status, prob.value, leaders, target_areas


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

    Instead we use find_shared_vertex_indices() which looks up each shared
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


# def resolve_overlaps(
#     polygons: list,
#     shrink_factor: float = 0.98,
# ) -> list:
#     """
#     Apply a uniform shrink toward each polygon's centroid to remove
#     residual overlaps after snapping.  Small shrink factors (0.95–0.99)
#     leave barely-visible gaps but guarantee no overlap.

#     Parameters
#     ----------
#     polygons    : list[np.ndarray]  from close_contiguous_gaps
#     shrink_factor : float  0 < factor <= 1.0

#     Returns
#     -------
#     list[np.ndarray]
#     """
#     return [shrink_polygon(np.asarray(p, dtype=float), shrink_factor)
#             for p in polygons]


def resolve_overlaps_raster(polygons, resolution=0.0):
    """
    Resolve overlapping polygons by rasterisation and marching-squares extraction.
    Returns (new_polygons, area_lost) where area_lost approximates the area
    that was 'taken' from each polygon due to overlap.
    """
    polys = [np.asarray(p, dtype=float) for p in polygons]
    n = len(polys)

    # ---- 1. Compute bounding box and grid ----
    all_pts = np.vstack(polys)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    diag = np.hypot(x_max - x_min, y_max - y_min)
    if resolution <= 0:
        resolution = max(diag / 200.0, 1e-10)

    n_cols = max(int(np.ceil((x_max - x_min) / resolution)) + 1, 2)
    n_rows = max(int(np.ceil((y_max - y_min) / resolution)) + 1, 2)
    col_coords = x_min + (np.arange(n_cols) + 0.5) * resolution
    row_coords = y_min + (np.arange(n_rows) + 0.5) * resolution
    xx, yy = np.meshgrid(col_coords, row_coords)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    # ---- 2. Build binary mask per polygon (point-in-polygon) ----
    masks = []
    for p in polys:
        path = Path(p)
        inside = path.contains_points(grid_pts).reshape(n_rows, n_cols)
        masks.append(inside)

    # ---- 3. Initial assignment: each cell gets first owner, then resolve overlaps ----
    owner = np.full((n_rows, n_cols), -1, dtype=int)
    # First pass: assign cells to first polygon found
    for k, mask in enumerate(masks):
        owner[mask & (owner == -1)] = k

    # For cells covered by multiple polygons, we need to assign to nearest centroid
    # Find all cells that are covered by more than one polygon
    overlap_count = np.sum(masks, axis=0)  # sum over polygons -> (n_rows, n_cols)
    overlap_mask = overlap_count > 1

    if np.any(overlap_mask):
        # Centroids of original polygons
        centroids = np.array([np.mean(p, axis=0) for p in polys])
        tree = KDTree(centroids)

        # Get coordinates of overlapping cells
        rr, cc = np.where(overlap_mask)
        cell_pts = np.column_stack([col_coords[cc], row_coords[rr]])
        # Query nearest centroid for each overlapping cell
        _, idx = tree.query(cell_pts)
        # Reassign
        for r, c, new_owner in zip(rr, cc, idx):
            owner[r, c] = new_owner

    # ---- 4. Compute area_lost for each polygon ----
    # area_lost = number of cells originally covered by polygon k but now assigned to someone else
    area_lost = [0.0] * n
    cell_area = resolution ** 2
    for k in range(n):
        # Originally covered by k
        orig = masks[k]
        # Now assigned to k
        now = (owner == k)
        lost_cells = orig & ~now
        area_lost[k] = lost_cells.sum() * cell_area

    # ---- 5. Extract polygon boundaries using marching squares ----
    new_polys = []
    for k in range(n):
        mask = (owner == k)
        if not np.any(mask):
            # Polygon completely disappeared -> tiny polygon around centroid
            centroid = np.mean(polys[k], axis=0)
            tiny = centroid + 1e-6 * np.array([
                [-1, -1], [1, -1], [1, 1], [-1, 1]
            ])
            new_polys.append(tiny)
            continue

        # find contours at level 0.5 (binary threshold)
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            # fallback: take all cells and build convex hull (but we want to avoid expansion)
            # Instead, we take the convex hull of the cell centres with a small shrink
            rr, cc = np.where(mask)
            pts = np.column_stack([col_coords[cc], row_coords[rr]])
            if len(pts) < 3:
                # too few points, use centroid
                new_polys.append(np.mean(pts, axis=0)[None, :] + 1e-6*np.random.randn(3,2))
            else:
                # Use convex hull but then shrink inward to avoid overlap
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                # shrink by 5% toward centroid to guarantee no overlap
                centroid = np.mean(hull_pts, axis=0)
                shrunk = centroid + 0.95 * (hull_pts - centroid)
                new_polys.append(shrunk)
        else:
            # Take the longest contour (outer boundary)
            largest = max(contours, key=len)
            # Convert row/col indices to coordinates
            coords = np.column_stack([
                col_coords[largest[:, 1].astype(int)],
                row_coords[largest[:, 0].astype(int)]
            ])
            # Close the polygon if not already
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
            new_polys.append(coords)

    return new_polys, area_lost


# ---------------------------------------------------------------------------
# Post-processing: unified gap/hole closing and overlap resolution
# ---------------------------------------------------------------------------

def _rasterise_polygons(polys, resolution):
    """
    Shared rasterisation used by both postprocess_gaps and resolve_polygon_overlaps.

    Returns
    -------
    covered   : (n_rows, n_cols) bool   — True where ANY polygon covers the cell
    owner     : (n_rows, n_cols) int    — index of polygon covering the cell,
                                          -1 if uncovered, n if contested overlap
    col_coords: (n_cols,) float
    row_coords: (n_rows,) float
    resolution: float                   — possibly auto-computed
    """
    from scipy.ndimage import label as nd_label

    all_pts      = np.vstack(polys)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    diag         = float(np.hypot(x_max - x_min, y_max - y_min))
    if resolution <= 0.0:
        resolution = max(diag / 200.0, 1e-10)

    n_cols = max(int(np.ceil((x_max - x_min) / resolution)) + 1, 2)
    n_rows = max(int(np.ceil((y_max - y_min) / resolution)) + 1, 2)

    covered    = np.zeros((n_rows, n_cols), dtype=bool)
    owner      = np.full((n_rows, n_cols), -1, dtype=np.int32)
    col_coords = x_min + (np.arange(n_cols) + 0.5) * resolution
    row_coords = y_min + (np.arange(n_rows) + 0.5) * resolution

    OVERLAP = len(polys)   # sentinel value meaning "owned by 2+ polygons"

    try:
        from matplotlib.path import Path as MplPath
        for k, poly in enumerate(polys):
            path = MplPath(poly)
            px_min, py_min = poly.min(axis=0)
            px_max, py_max = poly.max(axis=0)
            ci0 = max(int((px_min - x_min) / resolution) - 1, 0)
            ci1 = min(int((px_max - x_min) / resolution) + 2, n_cols)
            ri0 = max(int((py_min - y_min) / resolution) - 1, 0)
            ri1 = min(int((py_max - y_min) / resolution) + 2, n_rows)
            cc, rr   = np.meshgrid(col_coords[ci0:ci1], row_coords[ri0:ri1])
            test_pts = np.column_stack([cc.ravel(), rr.ravel()])
            inside   = path.contains_points(test_pts).reshape(rr.shape)
            sub_owner = owner[ri0:ri1, ci0:ci1]
            # Cells already owned by another polygon → mark as overlap
            sub_owner[inside & (sub_owner >= 0) & (sub_owner != k)] = OVERLAP
            # Cells not yet owned → assign to k
            sub_owner[inside & (sub_owner < 0)] = k
            covered[ri0:ri1, ci0:ci1] |= inside
    except ImportError:
        def _pip(px, py, poly):
            inside = False
            n_v = len(poly)
            j   = n_v - 1
            for i in range(n_v):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > py) != (yj > py)) and (
                    px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi
                ):
                    inside = not inside
                j = i
            return inside
        for ri in range(n_rows):
            for ci in range(n_cols):
                px, py = col_coords[ci], row_coords[ri]
                for k, poly in enumerate(polys):
                    if _pip(px, py, poly):
                        if owner[ri, ci] < 0:
                            owner[ri, ci] = k
                        elif owner[ri, ci] != k:
                            owner[ri, ci] = OVERLAP
                        covered[ri, ci] = True

    return covered, owner, col_coords, row_coords, resolution


def resolve_polygon_overlaps(polygons: list, resolution_factor: float = 500.0) -> tuple:
    """
    Remove pairwise polygon overlaps cleanly, without creating holes at
    non-overlapping borders.

    Core principle
    --------------
    The key insight is to distinguish three cases per polygon:

    NO OVERLAP (zero contested raster cells for polygon k)
        Return the ORIGINAL polygon vertices unchanged — no marching
        squares, no shrink, no holes.  This is the common case for
        densely packed maps like US states where the optimizer already
        did a good job.

    PARTIAL OVERLAP (contested cells exist, no containment)
        Nearest-centroid cell reassignment draws the correct midline.
        find_contours traces the new boundary.  A minimal 0.5-cell inward
        nudge corrects the marching-squares half-cell offset.
        Non-overlapping border sections are snapped back to the nearest
        original vertex (within 2 cells) to preserve clean borders.

    CONTAINMENT (>=90% of small is inside large)
        Both polygons returned unchanged.  Simple polygons cannot have
        holes, so the large polygon's outline still visually wraps around
        the small one.  Draw order (large first, small on top) handles
        the visual result.

    Parameters
    ----------
    polygons          : list[np.ndarray]
    resolution_factor : float
        Grid resolution = bbox_diagonal / resolution_factor.
        500 (default) is good for world and US maps.

    Returns
    -------
    new_polygons : list[np.ndarray]
    area_lost    : list[float]
    """
    try:
        from skimage.measure import find_contours
        from matplotlib.path import Path as MplPath
    except ImportError as exc:
        print(f"  resolve_polygon_overlaps requires skimage + matplotlib ({exc}). Skipping.")
        return list(polygons), [0.0] * len(polygons)

    polys = [np.asarray(p, dtype=float) for p in polygons]
    n     = len(polys)

    def _area(pts):
        x, y = pts[:,0], pts[:,1]
        return float(0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))))

    poly_areas = [_area(p) for p in polys]

    # ── 1. Build raster ────────────────────────────────────────────────────
    all_pts       = np.vstack(polys)
    x_min, y_min  = all_pts.min(axis=0)
    x_max, y_max  = all_pts.max(axis=0)
    diag          = float(np.hypot(x_max - x_min, y_max - y_min))
    resolution    = max(diag / resolution_factor, 1e-10)

    n_cols = max(int(np.ceil((x_max - x_min) / resolution)) + 2, 3)
    n_rows = max(int(np.ceil((y_max - y_min) / resolution)) + 2, 3)
    col_coords = x_min + (np.arange(n_cols) + 0.5) * resolution
    row_coords = y_min + (np.arange(n_rows) + 0.5) * resolution

    xx, yy    = np.meshgrid(col_coords, row_coords)
    grid_pts  = np.column_stack([xx.ravel(), yy.ravel()])

    masks = []
    for poly in polys:
        path   = MplPath(poly)
        inside = path.contains_points(grid_pts).reshape(n_rows, n_cols)
        masks.append(inside)

    masks_arr   = np.array(masks, dtype=np.uint8)
    overlap_cnt = masks_arr.sum(axis=0)   # 0=empty, 1=clean, >=2=contested
    cell_area   = resolution ** 2

    # ── 2. Containment detection ──────────────────────────────────────────
    contain_threshold = 0.90
    contained_by = [-1] * n
    for i in range(n):
        if poly_areas[i] < 1e-12:
            continue
        for j in range(n):
            if i == j:
                continue
            shared = float((masks_arr[i].astype(bool) & masks_arr[j].astype(bool)).sum()) * cell_area
            if shared / poly_areas[i] >= contain_threshold:
                contained_by[i] = j
                print(f"  Containment: polygon {i} ({poly_areas[i]:.4g}) is "
                      f"{shared/poly_areas[i]*100:.0f}% inside polygon {j} "
                      f"({poly_areas[j]:.4g}) — both unchanged")
                break

    # ── 3. Count contested cells per polygon ──────────────────────────────
    contested_cells_k = np.array([
        int((masks_arr[k].astype(bool) & (overlap_cnt > 1)).sum())
        for k in range(n)
    ])

    # ── 4. Nearest-centroid reassignment for contested cells ──────────────
    centroids = [np.mean(p, axis=0) for p in polys]
    owner     = np.full((n_rows, n_cols), -1, dtype=np.int32)

    for k in range(n):
        uncontested = masks_arr[k].astype(bool) & (overlap_cnt == 1)
        owner[uncontested] = k

    ri_c, ci_c = np.where(overlap_cnt > 1)
    for ri, ci in zip(ri_c, ci_c):
        owners_here = [k for k in range(n) if masks_arr[k, ri, ci]]
        if not owners_here:
            continue
        pt    = np.array([col_coords[ci], row_coords[ri]])
        dists = [np.linalg.norm(pt - centroids[k]) for k in owners_here]
        owner[ri, ci] = owners_here[int(np.argmin(dists))]

    # ── 5. area_lost ──────────────────────────────────────────────────────
    area_lost = [0.0] * n
    for k in range(n):
        lost = masks_arr[k].astype(bool) & (owner != k)
        area_lost[k] = float(lost.sum() * cell_area)

    # ── 6. Build output polygons ──────────────────────────────────────────
    result = list(polys)

    for k in range(n):
        # CONTAINMENT: both polygons unchanged
        if contained_by[k] >= 0:
            result[k] = polys[k].copy()
            continue
        if any(contained_by[i] == k for i in range(n)):
            result[k] = polys[k].copy()
            continue

        # NO OVERLAP: return original — this is the key fix for US holes
        if contested_cells_k[k] == 0:
            result[k] = polys[k].copy()
            continue

        # PARTIAL OVERLAP: trace boundary from reassigned raster
        mask_k = (owner == k)
        if not mask_k.any():
            c = np.mean(polys[k], axis=0)
            result[k] = c + 1e-4 * np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
            continue

        contours = find_contours(mask_k.astype(float), 0.5)
        if not contours:
            result[k] = polys[k].copy()
            continue

        largest = max(contours, key=len)
        # find_contours returns (row, col) fractional indices
        coords = np.column_stack([
            x_min + largest[:, 1] * resolution,
            y_min + largest[:, 0] * resolution,
        ])

        # Minimal 0.5-cell inward nudge — corrects marching-squares offset only
        centroid  = np.mean(coords, axis=0)
        diff      = coords - centroid
        dists_r   = np.linalg.norm(diff, axis=1, keepdims=True)
        dists_r   = np.maximum(dists_r, 1e-12)
        nudge     = 0.5 * resolution
        coords    = centroid + (diff / dists_r) * np.maximum(dists_r - nudge, 0.0)

        # Snap contour vertices near original vertices back to exact original
        # position, but only where those vertices were in non-contested cells.
        # This preserves clean borders that had no overlap.
        snap_dist  = 2.0 * resolution
        orig_verts = polys[k]
        for vi in range(len(coords)):
            d_orig = np.linalg.norm(orig_verts - coords[vi], axis=1)
            ni     = int(np.argmin(d_orig))
            if d_orig[ni] < snap_dist:
                ov     = orig_verts[ni]
                ci_o   = int(np.clip((ov[0]-x_min)/resolution, 0, n_cols-1))
                ri_o   = int(np.clip((ov[1]-y_min)/resolution, 0, n_rows-1))
                if overlap_cnt[ri_o, ci_o] <= 1:
                    coords[vi] = ov

        if len(coords) >= 3:
            result[k] = coords
            print(f"  Polygon {k}: redrawn ({contested_cells_k[k]} contested cells, "
                  f"lost {area_lost[k]:.4g})")
        else:
            result[k] = polys[k].copy()

    total_changed = sum(1 for k in range(n) if contested_cells_k[k] > 0
                        and contained_by[k] < 0
                        and not any(contained_by[i]==k for i in range(n)))
    print(f"  resolve_polygon_overlaps: {total_changed}/{n} polygons had overlaps "
          f"and were redrawn; {n-total_changed} returned unchanged.")

    return result, area_lost



# =============================================================================
# DIAGNOSTIC: H/V ORDERING SELF-CONSISTENCY CHECK
# =============================================================================
#
# Standalone — no CVXPY, no polygons, no solve. Only needs the same
# horizontal_pairs / vertical_pairs / target_areas you already pass into
# CartogramFramework_global. Answers one specific question directly:
# "even before touching contiguous locking or anything else, CAN these
# ordering pairs ever be satisfied simultaneously?"
#
# The hard H/V ordering constraints are a system of "difference
# constraints" of the form x_j - x_i >= c_ij. Such a system is infeasible
# if and only if its constraint graph contains a cycle whose accumulated
# constants sum to something requiring a variable to be greater than
# itself (e.g. A must be left of B, B left of C, and C left of A — no
# assignment of x-coordinates can satisfy all three). This is a property
# of how horizontal_pairs/vertical_pairs were generated, independent of
# contiguous locking, t_min/t_max, lambda weights, or anything else in
# the optimisation — no amount of parameter tuning fixes it if present;
# only correcting the pair-generation logic does.
# =============================================================================

def check_hv_ordering_feasibility(
    horizontal_pairs,
    vertical_pairs,
    target_areas,
    neighboring_pairs=None,
    skip_adjacent=True,
    epsilon=1.0,
):
    """
    Pre-flight check for whether the hard H/V ordering constraints
        x_j - x_i >= (sqrt(A_i)+sqrt(A_j))/2 + gap_ij   for (i,j) in horizontal_pairs
        y_j - y_i >= (sqrt(A_i)+sqrt(A_j))/2 + gap_ij   for (i,j) in vertical_pairs
    (gap_ij = 0 if (i,j) is adjacent, else epsilon)
    can ever be satisfied simultaneously — the exact system
    CartogramFramework_global builds when soft_hv_ordering=False, with
    the same contiguous_skip_hard_ordering_for_adjacent-style skip
    applied if skip_adjacent=True (match this to whatever you're passing
    the solver, so the check reflects what it actually sees).

    Uses Bellman-Ford negative-cycle detection, the standard technique
    for difference-constraint systems: for x_j - x_i >= c, add a graph
    edge j -> i with weight -c; the system is feasible iff the graph has
    no negative-weight cycle.

    Returns
    -------
    dict with keys 'horizontal' and 'vertical'. Each is either None (no
    inconsistency found — this half of the system is fine) or a list of
    region indices tracing an offending cycle, e.g. [3, 17, 42, 3]
    meaning region 3 must be left of 17, 17 left of 42, and 42 left of 3
    — simultaneously impossible.
    """
    adjacent_set = set()
    if neighboring_pairs is not None:
        for i, j in neighboring_pairs:
            adjacent_set.add((min(i, j), max(i, j)))

    def is_adj(i, j):
        return (min(i, j), max(i, j)) in adjacent_set

    def find_cycle(pairs):
        if not pairs:
            return None
        edges = []   # (u, v, weight) meaning an edge u -> v
        nodes = set()
        for (i, j) in pairs:
            if skip_adjacent and is_adj(i, j):
                continue
            w = (np.sqrt(target_areas[i]) + np.sqrt(target_areas[j])) / 2.0
            g = 0.0 if is_adj(i, j) else epsilon
            c = w + g
            edges.append((j, i, -c))   # x_j - x_i >= c  ->  edge j->i, weight -c
            nodes.add(i); nodes.add(j)
        if not edges:
            return None

        nodes = list(nodes)
        n = len(nodes)
        dist = {node: 0.0 for node in nodes}   # virtual source at 0 to every node
        pred = {node: None for node in nodes}

        for _ in range(n):
            relaxed_edge = None
            for (u, v, w) in edges:
                if dist[u] + w < dist[v] - 1e-9:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    relaxed_edge = v
            if relaxed_edge is None:
                return None   # converged — no negative cycle, system is feasible

        # nth-iteration relaxation still happening => negative cycle exists.
        # Walk n predecessor-steps back from any still-relaxing node to
        # guarantee landing strictly inside the cycle, then trace it out.
        x = relaxed_edge
        for _ in range(n):
            x = pred[x]
        cycle = [x]
        y = pred[x]
        while y != x:
            cycle.append(y)
            y = pred[y]
        cycle.append(x)
        cycle.reverse()
        return cycle

    return {
        "horizontal": find_cycle(horizontal_pairs),
        "vertical": find_cycle(vertical_pairs),
    }
