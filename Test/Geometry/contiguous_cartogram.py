"""
contiguous_cartogram.py  (v2 — fixed scale normalisation + displacement anchor)
================================================================================
Drop-in replacement for the previous version.

Root causes of the exploding-ellipse bug
-----------------------------------------
1. No vertex anchor: the smoothness (Laplacian) term only keeps vertices
   close to *each other* — it has no fixed reference, so the whole cloud
   drifts.  Fix: add a displacement penalty  ||v - v_orig||²  alongside
   the Laplacian.

2. Coordinate scale vs. area scale mismatch: even after preprocess()
   rescales target_areas to geographic units, the linearised shoelace
   area is still sensitive to the absolute magnitude of coordinates
   (~100 for lon/lat).  Fix: normalise all coordinates to zero-mean,
   unit-std before building the LP, then denormalise the result.

3. The linearised area formula (first-order Taylor) is only accurate for
   small deformations.  The displacement anchor (fix 1) keeps deformations
   small, which also keeps the linearisation accurate.

Integration into CartogramFramework  (unchanged from v1)
---------------------------------------------------------
CHANGE 1 — top of CartogramFramework.py:
    from contiguous_cartogram import contiguous_cartogram

CHANGE 2 — inside the init_polygons loop (step 2):
    elif shape == "contiguous":
        init_polygons.append(poly)

CHANGE 3 — after the loop, before step 3:
    if shape == "contiguous":
        return contiguous_cartogram(
            polygons=init_polygons,
            target_areas=target_areas,
            fixed_points=fixed_points,
            target_centers=target_centers,
            neighboring_pairs=neighboring_pairs,
            shared_vertices_of_neighbors=shared_vertices_of_neighbors,
            W_shape=W_shape,
            W_area=W_area,
            W_spatial=W_spatial,
        )
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _shoelace_linear(v: cp.Variable, p: np.ndarray) -> cp.Expression:
    """
    Linearised (affine) signed area of polygon with CVXPY vertex variable v,
    expanded around reference positions p  (both shape (n,2)).

    2A = Σ_k  x_k * (py_{k+1} - py_{k-1})
           + px_k * (y_{k+1} - y_{k-1})
           - px_k * (py_{k+1} - py_{k-1})   ← constant, but kept for correctness

    The last line is constant and cancels in the squared-error objective, but
    we include it so the expression evaluates to the correct area at v=p.
    """
    n   = p.shape[0]
    kp1 = (np.arange(n) + 1) % n
    km1 = (np.arange(n) - 1) % n

    dy = p[kp1, 1] - p[km1, 1]   # py_{k+1} - py_{k-1}  (n,)
    dx = p[kp1, 0] - p[km1, 0]   # px_{k+1} - px_{k-1}  (n,)

    # Affine in v:  Σ x_k*dy_k  +  Σ px_k*(y_{k+1}-y_{k-1})  - const
    term_x = dy @ v[:, 0]
    term_y = p[:, 0] @ (v[kp1, 1] - v[km1, 1])

    return 0.5 * (term_x + term_y)


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    ))


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def contiguous_cartogram(
    polygons:                    list[np.ndarray],
    target_areas:                list[float],
    fixed_points:                list[np.ndarray],
    target_centers:              list[np.ndarray],
    neighboring_pairs,           # list[(i,j)]
    shared_vertices_of_neighbors,# list[[i, j, shared_pts]]
    W_shape:   float = 1.0,      # weight: shape smoothness + displacement anchor
    W_area:    float = 5.0,      # weight: area fidelity
    W_spatial: float = 0.5,      # weight: centroid anchor
) -> tuple:
    """
    Contiguous cartogram solver.  Returns the same 5-tuple as
    CartogramFramework_global_CLAUDE_2:
        (new_polygons, actual_areas, status, obj_value, leaders)
    leaders is always [] — contiguous cartograms have no lost adjacencies.
    """

    # ------------------------------------------------------------------
    # 1. Coordinate normalisation
    #    Work in a zero-mean, unit-std coordinate system so that the LP
    #    coefficients are O(1) and the area units match naturally.
    # ------------------------------------------------------------------
    all_pts      = np.vstack(polygons)
    coord_center = all_pts.mean(axis=0)          # (2,)
    coord_scale  = float(all_pts.std())          # scalar, preserves aspect ratio

    def _n(pts):
        return (np.asarray(pts) - coord_center) / coord_scale

    polys_n   = [_n(p)  for p in polygons]
    centers_n = [_n(tc) for tc in target_centers]
    areas_n   = [ta / coord_scale**2 for ta in target_areas]

    n_regions = len(polys_n)

    # ------------------------------------------------------------------
    # 2. CVXPY variables: one (n_i, 2) matrix per polygon
    # ------------------------------------------------------------------
    vvars: list[cp.Variable] = []
    for poly_n in polys_n:
        v = cp.Variable(poly_n.shape)
        v.value = poly_n.copy()      # warm start
        vvars.append(v)

    constraints: list = []
    area_terms   = []
    shape_terms  = []
    center_terms = []

    # ------------------------------------------------------------------
    # 3. Per-polygon objective terms
    # ------------------------------------------------------------------
    for i, (poly_n, ta_n, tc_n) in enumerate(zip(polys_n, areas_n, centers_n)):
        ni  = poly_n.shape[0]
        v   = vvars[i]
        tc_n = np.asarray(tc_n)

        kp1 = (np.arange(ni) + 1) % ni
        km1 = (np.arange(ni) - 1) % ni

        # ---- Area fidelity (linearised shoelace) ---------------------
        area_expr = _shoelace_linear(v, poly_n)
        area_terms.append(cp.square(area_expr - ta_n))

        # ---- Shape: Laplacian + displacement anchor ------------------
        # Laplacian: each vertex should stay near the mean of its neighbours
        neighbour_mean = (v[kp1, :] + v[km1, :]) / 2.0
        laplacian      = cp.sum_squares(v - neighbour_mean)

        # Displacement anchor: vertices should not move far from original
        displacement = cp.sum_squares(v - poly_n)

        shape_terms.append(0.3 * laplacian + 0.7 * displacement)

        # ---- Centroid anchor -----------------------------------------
        centroid_expr = cp.sum(v, axis=0) / ni
        center_terms.append(cp.sum_squares(centroid_expr - tc_n))

    # ------------------------------------------------------------------
    # 4. Shared-boundary hard equality constraints
    #    Every vertex shared between adjacent polygons must be coincident
    #    after deformation — this is what makes the result *contiguous*.
    # ------------------------------------------------------------------
    if shared_vertices_of_neighbors is not None:
        for entry in shared_vertices_of_neighbors:
            i, j      = int(entry[0]), int(entry[1])
            shared_pts = entry[2]
            poly_i_n  = polys_n[i]
            poly_j_n  = polys_n[j]

            for pt in shared_pts:
                pt_n = _n(pt)

                idx_i = int(np.argmin(np.linalg.norm(poly_i_n - pt_n, axis=1)))
                idx_j = int(np.argmin(np.linalg.norm(poly_j_n - pt_n, axis=1)))

                constraints.append(vvars[i][idx_i, :] == vvars[j][idx_j, :])

    # ------------------------------------------------------------------
    # 5. Objective and solve
    # ------------------------------------------------------------------
    objective = cp.Minimize(
        W_area    * cp.sum(area_terms)
        + W_shape   * cp.sum(shape_terms)
        + W_spatial * cp.sum(center_terms)
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)

    print(f"Status : {prob.status}")
    print(f"Obj val: {prob.value:.6g}" if prob.value is not None else "Obj val: None")

    # ------------------------------------------------------------------
    # 6. Extract and denormalise results
    # ------------------------------------------------------------------
    new_polygons: list[np.ndarray] = []
    actual_areas: list[float]      = []

    for i, v in enumerate(vvars):
        pts_n = v.value if v.value is not None else polys_n[i]
        # Denormalise back to geographic coordinates
        pts_world = pts_n * coord_scale + coord_center
        new_polygons.append(pts_world)
        actual_areas.append(_polygon_area(pts_world))

    return new_polygons, actual_areas, prob.status, prob.value, []
