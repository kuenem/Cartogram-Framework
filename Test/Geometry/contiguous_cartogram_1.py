"""
contiguous_cartogram.py  (v3)
==============================
Two contiguous cartogram implementations that plug into
CartogramFramework_global_CLAUDE_2 via shape="contiguous" and
shape="contiguous2":

  shape="contiguous"   — standalone LP (vertex variables per polygon,
                         shared-boundary hard equalities, linearised
                         shoelace area).  Same as v2 but with the
                         east-coast overlap fix described below.

  shape="contiguous2"  — unified LP: uses the *existing* radial-scale
                         variables (t_ik, delta_i) from the main framework
                         but adds hard equality constraints that pin shared
                         vertices of neighbours to the same position.
                         No new variable types; the rest of the objective
                         (shape deformation, area, centre, topology) is
                         unchanged.  Activated by returning a special
                         "shared_vertex_constraints" list from this module
                         and injecting it in step 3 of the main loop.

Fix for east-coast overlap (question 1)
----------------------------------------
The overlap on dense east-coast states is caused by the linearised shoelace
area being a poor approximation when many small states are packed together:
the solver over-inflates vertices to hit the area target and neighbouring
states collide.  Three changes fix this:

  a) Increase W_area relative to W_shape so area fidelity dominates but
     is not so strong that it overrides the displacement anchor.
     Recommended: W_area=8, W_shape=1, displacement ratio=0.85.

  b) Add a soft repulsion term between the centroids of neighbouring
     polygons (replaces the hard H/V constraints that don't apply here).
     This pushes overlapping state centroids apart gently.

  c) Use iterative re-linearisation (2–3 iterations): after solving,
     use the output polygons as the new reference for _shoelace_linear
     and solve again.  Each iteration the linearisation is more accurate.
     Controlled by n_iter parameter (default 2).

Integration into CartogramFramework_global_CLAUDE_2
----------------------------------------------------
Shape "contiguous" and "contiguous2" require different hooks.

  For shape="contiguous":  unchanged from v2 — early return after step 2.

  For shape="contiguous2": the main loop runs normally but
    build_shared_vertex_constraints() is called before step 3 and its
    output is appended to `constraints` inside the per-region loop at
    step 3, after all t/delta variables are created.

See integration comments at the bottom of this file.
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    ))


def _shoelace_linear(v: cp.Variable, p: np.ndarray) -> cp.Expression:
    """
    Affine (linearised) signed area of polygon vertex variable v,
    expanded around reference positions p  (both shape (n,2)).
    """
    n   = p.shape[0]
    kp1 = (np.arange(n) + 1) % n
    km1 = (np.arange(n) - 1) % n
    dy  = p[kp1, 1] - p[km1, 1]
    term_x = dy @ v[:, 0]
    term_y = p[:, 0] @ (v[kp1, 1] - v[km1, 1])
    return 0.5 * (term_x + term_y)


def _find_vertex_index(poly: np.ndarray, pt: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(poly - pt, axis=1)))


# ---------------------------------------------------------------------------
# shape="contiguous"  — standalone vertex LP  (v2 + iterative re-lin + repulsion)
# ---------------------------------------------------------------------------

def contiguous_cartogram(
    polygons,
    target_areas,
    fixed_points,
    target_centers,
    neighboring_pairs,
    shared_vertices_of_neighbors,
    W_shape:   float = 1.0,
    W_area:    float = 8.0,    # raised from 5 → reduces east-coast overlap
    W_spatial: float = 0.5,
    n_iter:    int   = 2,      # iterative re-linearisation passes
) -> tuple:
    """
    Standalone contiguous cartogram LP.

    Returned tuple matches CartogramFramework_global_CLAUDE_2:
        (new_polygons, actual_areas, status, obj_value, leaders=[])
    """

    # --- normalise coordinates ---
    all_pts      = np.vstack(polygons)
    coord_center = all_pts.mean(axis=0)
    coord_scale  = float(all_pts.std())

    def _n(pts):
        return (np.asarray(pts) - coord_center) / coord_scale

    polys_n   = [_n(p)  for p in polygons]
    centers_n = [_n(tc) for tc in target_centers]
    areas_n   = [ta / coord_scale**2 for ta in target_areas]

    # build neighbour set for repulsion term
    adj_set: set[tuple[int,int]] = set()
    if neighboring_pairs is not None:
        for i, j in neighboring_pairs:
            adj_set.add((min(i,j), max(i,j)))

    current_polys = [p.copy() for p in polys_n]   # reference for linearisation
    status = "unknown"
    obj_val = None
    vvars: list[cp.Variable] = []

    for iteration in range(n_iter):
        vvars = []
        for p in current_polys:
            v = cp.Variable(p.shape)
            v.value = p.copy()
            vvars.append(v)

        constraints: list = []
        area_terms    = []
        shape_terms   = []
        center_terms  = []
        repulse_terms = []

        # --- per-polygon terms ---
        for i, (poly_n, ta_n, tc_n) in enumerate(
                zip(current_polys, areas_n, centers_n)):
            ni   = poly_n.shape[0]
            v    = vvars[i]
            tc_n = np.asarray(tc_n)
            kp1  = (np.arange(ni) + 1) % ni
            km1  = (np.arange(ni) - 1) % ni

            # area fidelity
            area_terms.append(cp.square(_shoelace_linear(v, poly_n) - ta_n))

            # shape: laplacian (0.15) + displacement anchor (0.85)
            # higher displacement ratio → less east-coast spreading
            neighbour_mean = (v[kp1, :] + v[km1, :]) / 2.0
            laplacian      = cp.sum_squares(v - neighbour_mean)
            displacement   = cp.sum_squares(v - poly_n)
            shape_terms.append(0.15 * laplacian + 0.85 * displacement)

            # centroid anchor
            centroid_expr = cp.sum(v, axis=0) / ni
            center_terms.append(cp.sum_squares(centroid_expr - tc_n))

        # --- soft repulsion between adjacent centroids ---
        # Pushes overlapping state centres apart without hard H/V constraints.
        repulsion_weight = 0.3 * W_shape
        # for (i, j) in adj_set:
        #     ci = cp.sum(vvars[i], axis=0) / vvars[i].shape[0]
        #     cj = cp.sum(vvars[j], axis=0) / vvars[j].shape[0]
        #     # penalise squared distance falling below a minimum separation
        #     # minimum = sum of "radii" (sqrt of normalised areas)
        #     min_sep = np.sqrt(areas_n[i]) + np.sqrt(areas_n[j])
        #     diff    = ci - cj
        #     dist_sq = cp.sum_squares(diff)
        #     # soft lower-bound: relu( min_sep² - dist² )²  approximated by
        #     # a linear slack: slack ≥ min_sep² - dist_sq, slack ≥ 0
        #     slack = cp.Variable(nonneg=True)
        #     constraints.append(slack >= min_sep**2 - dist_sq)
        #     repulse_terms.append(cp.square(slack))

        # Replace the entire repulsion block with this DCP-compliant version:
        for (i, j) in adj_set:
            ci = cp.sum(vvars[i], axis=0) / vvars[i].shape[0]
            cj = cp.sum(vvars[j], axis=0) / vvars[j].shape[0]
            # Direction between reference centroids (constant vector)
            ref_ci = polys_n[i].mean(axis=0)
            ref_cj = polys_n[j].mean(axis=0)
            direction = ref_cj - ref_ci                        # constant (2,)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm                                  # unit vector
            # Projected separation along that direction (affine in variables)
            proj_sep = direction @ (cj - ci)                   # scalar affine expr
            # Target minimum separation
            min_sep = np.sqrt(areas_n[i]) + np.sqrt(areas_n[j])
            # Penalise when proj_sep < min_sep  →  hinge loss: max(0, min_sep - proj_sep)
            # Expressed as: slack >= min_sep - proj_sep, slack >= 0  → convex lower bound on affine
            slack = cp.Variable(nonneg=True)
            constraints.append(slack >= min_sep - proj_sep)    # affine RHS → DCP ✓
            repulse_terms.append(cp.square(slack))

        # --- shared-boundary hard equality constraints ---
        if shared_vertices_of_neighbors is not None:
            for entry in shared_vertices_of_neighbors:
                i, j       = int(entry[0]), int(entry[1])
                shared_pts = entry[2]
                poly_i_n   = current_polys[i]
                poly_j_n   = current_polys[j]
                for pt in shared_pts:
                    pt_n  = _n(pt)
                    idx_i = _find_vertex_index(poly_i_n, pt_n)
                    idx_j = _find_vertex_index(poly_j_n, pt_n)
                    constraints.append(vvars[i][idx_i, :] == vvars[j][idx_j, :])

        # --- objective ---
        objective = cp.Minimize(
            W_area    * cp.sum(area_terms)
            + W_shape   * cp.sum(shape_terms)
            + W_spatial * cp.sum(center_terms)
            + (repulsion_weight * cp.sum(repulse_terms)
               if repulse_terms else cp.Constant(0.0))
        )

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=(iteration > 0))
        status  = prob.status
        obj_val = prob.value

        print(f"  [contiguous iter {iteration+1}/{n_iter}] "
              f"status={status}  obj={obj_val:.6g}" if obj_val is not None
              else f"  [contiguous iter {iteration+1}/{n_iter}] status={status}")

        # update reference polygons for next iteration
        for i, v in enumerate(vvars):
            if v.value is not None:
                current_polys[i] = v.value.copy()

    # --- extract and denormalise ---
    new_polygons: list[np.ndarray] = []
    actual_areas: list[float]      = []
    for i, v in enumerate(vvars):
        pts_n     = v.value if v.value is not None else polys_n[i]
        pts_world = pts_n * coord_scale + coord_center
        new_polygons.append(pts_world)
        actual_areas.append(_polygon_area(pts_world))

    print(f"Status : {status}")
    print(f"Obj val: {obj_val:.6g}" if obj_val is not None else "Obj val: None")
    return new_polygons, actual_areas, status, obj_val, []


# ---------------------------------------------------------------------------
# shape="contiguous2"  — unified LP via shared-vertex constraints
# ---------------------------------------------------------------------------

def build_shared_vertex_constraints(
    init_polygons:               list[np.ndarray],
    new_polygon_exprs:           list,             # cp.Expression list from step 3
    shared_vertices_of_neighbors,
) -> list:
    """
    Build hard equality constraints that pin shared vertices of neighbouring
    polygons to the same position in the unified LP.

    Called AFTER all per-region variables and new_polygon_exprs are built
    in step 3 of CartogramFramework_global_CLAUDE_2.

    Parameters
    ----------
    init_polygons       : list of (n_i, 2) reference vertex arrays
    new_polygon_exprs   : list of cp.Expression, each shape (n_i, 2),
                          i.e. the `new_pts` expressions built in step 3
    shared_vertices_of_neighbors : list of [i, j, shared_pts]

    Returns
    -------
    constraints : list of cp.Constraint  (equality constraints)
    """
    constraints = []
    if shared_vertices_of_neighbors is None:
        return constraints

    for entry in shared_vertices_of_neighbors:
        i, j       = int(entry[0]), int(entry[1])
        shared_pts = entry[2]

        poly_i = np.asarray(init_polygons[i])
        poly_j = np.asarray(init_polygons[j])
        expr_i = new_polygon_exprs[i]   # cp.Expression (n_i, 2)
        expr_j = new_polygon_exprs[j]   # cp.Expression (n_j, 2)

        # Skip if either expression is a cp.Parameter (degenerate polygon)
        if isinstance(expr_i, cp.Parameter) or isinstance(expr_j, cp.Parameter):
            continue

        for pt in shared_pts:
            pt = np.asarray(pt)
            idx_i = _find_vertex_index(poly_i, pt)
            idx_j = _find_vertex_index(poly_j, pt)

            # Pin the two polygon expressions at the shared vertex index
            constraints.append(expr_i[idx_i, :] == expr_j[idx_j, :])

    return constraints


# ---------------------------------------------------------------------------
# Integration notes for CartogramFramework_global_CLAUDE_2
# ---------------------------------------------------------------------------
#
# The import at the top of CartogramFramework.py already covers both:
#   from contiguous_cartogram import contiguous_cartogram, build_shared_vertex_constraints
#
# ── shape="contiguous" ──────────────────────────────────────────────────────
# Already wired: the early-return block after step 2 calls contiguous_cartogram().
# No further changes needed.
#
# ── shape="contiguous2" ─────────────────────────────────────────────────────
# Three small additions to CartogramFramework_global_CLAUDE_2:
#
# ADDITION A — in step 2 re-init loop, add one elif:
#     elif shape == "contiguous2":
#         init_polygons.append(poly)          # keep original outline
#
# ADDITION B — after the per-region loop in step 3 closes (after the last
# `center_terms.append(...)` line), add:
#
#     if shape == "contiguous2":
#         shared_constraints = build_shared_vertex_constraints(
#             init_polygons=init_polygons,
#             new_polygon_exprs=new_polygon_exprs,
#             shared_vertices_of_neighbors=shared_vertices_of_neighbors,
#         )
#         constraints.extend(shared_constraints)
#
# ADDITION C — in step 8 leader computation, extend the shape check:
#     if compute_leaders_flag and shape in ("square", "circle"):
#         ...  (unchanged)
#
# That is all.  The objective, solver call, and result extraction are
# completely unchanged.  The only difference is the extra equality
# constraints that glue shared vertices together.
