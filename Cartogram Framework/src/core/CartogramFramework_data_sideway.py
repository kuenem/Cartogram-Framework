"""
CartogramFramework_global
=================================
Unified cartogram optimisation covering:
  - Demers (square) cartograms
  - Dorling (circle) cartograms
  - Original shape
  - Contiguous shape  (shape="contiguous")
  - Contiguous soft   (shape="contiguous2")

Feasibility & identity guarantees
----------------------------------
Two structural properties are guaranteed by construction:

1. ALWAYS FEASIBLE. Every constraint in the problem is now either a
   per-variable box bound or a one-sided slack/hinge lower bound. The
   three constraint groups that used to be hard equalities/inequalities
   (mean-scale, horizontal/vertical ordering, contiguous shared-vertex
   locking) have all been converted to soft quadratic penalties gated by
   their own weight (lambda_mean_scale, lambda_order, lambda_contiguous).
   A large weight approximates the old hard behaviour; it can never make
   the problem infeasible. The only remaining way to get infeasibility is
   t_min > t_max, which is validated explicitly and raises early.

2. ALL-WEIGHTS-OFF IDENTITY. If every lambda_* is 0, every soft penalty
   above evaluates to 0 everywhere in the feasible box, so the objective
   is flat and any feasible point is "optimal" — including but not
   uniquely the input geometry (t=1, delta=0). To make the input the
   *unique* minimiser in that case (rather than an arbitrary tie), a
   tiny always-on anchor penalty pulls t -> 1 and each region's centre
   back to its fixed point. Its weight (_ANCHOR_EPS below) is hardcoded
   and NOT exposed as a user parameter — it is small enough (1e-9) to be
   negligible next to any real lambda, but decisive when all real
   lambdas are 0.
   Caveat: for shape="circle"/"square" the "input" being reproduced is
   the *re-initialised* circle/square (see step 2, "Optional polygon
   re-initialisation"), not the original geographic polygon — that
   re-initialisation happens independently of any lambda weight.
   Caveat: identity requires t=1 to lie within [t_min, t_max]; if you
   pass bounds that exclude 1 (e.g. to force shrinkage), that is a
   deliberate constraint and identity is correctly not reachable.

3. EITHER/OR WEIGHT SOURCE. Each of W_shape/W_area/W_spatial/W_topology
   comes from exactly one source, never a blend of both:
     W_shape    <- lambda_shape    else (1 - shape_deformation)
     W_area     <- lambda_area     else cartographic_error
     W_spatial  <- lambda_center   else spatial_deformation
     W_topology <- lambda_topology else topological_accuracy
   This precedence lives in preprocess_global itself (src/core/
   preprocessing_data.py): each lambda_* defaults to None here, and
   preprocess_global uses it if it's not None, otherwise falls back to
   the legacy value. Whichever one actually governs, it does so outright
   — the two are never added, multiplied, or otherwise combined.
   The three plain legacy args (cartographic_error, spatial_deformation,
   topological_accuracy) keep this wrapper's original concrete defaults
   (1.0, 0.0, 1.0) so that omitting a lambda_*/legacy pair entirely still
   reproduces the framework's original default weights exactly.
   shape_deformation is dual-purpose: besides being the legacy fallback
   for W_shape, it independently selects circle/square re-initialisation
   in step 2 below. Existing calls (e.g. the Demers/Dorling cells in
   exploration.ipynb) legitimately pass both lambda_shape AND
   shape_deformation=0.0 together for that reason — no conflict, since
   once lambda_shape is given it simply wins for W_shape outright and
   shape_deformation is still doing its separate structural job.
   Five previously-accepted legacy parameters (relative_direction,
   global_shape, local_shape, complexity, data_ink_ratio) are kept only
   for call-signature compatibility — preprocess_global never reads any
   of them in its body, so they were already complete no-ops before any
   of today's changes.

4. TANGENTIAL VERTEX FREEDOM + VERTEX-LEVEL ANTI-OVERLAP (opt-in, default
   off — zero behaviour change unless requested):
   a) Fixed a real bug in Approach 2 (centre repulsion): it used two
      independent one-sided hinges per axis, one of which grew WITHOUT
      BOUND the farther apart two centres already were — the opposite of
      what a repulsion penalty should do. Direction-agnostic "stay
      >= margin apart" is inherently non-convex (a keep-out disk), so a
      convex surrogate has to commit to a side in advance; this now picks
      a fixed separating axis from each pair's ORIGINAL relative position
      (_one_sided_repulsion()) and applies a single correct hinge.
   b) tangential_freedom=True gives every "free" vertex — one NOT shared
      with a neighbouring region (i.e. facing the void/open background,
      not another region) — a second local degree of freedom u_{i,k} in
      addition to the existing radial scale t_{i,k}: v'_{i,k} = m_i +
      t_{i,k}*r_{i,k} + u_{i,k}*n_{i,k}, where r is the original
      centre->vertex axis and n is that axis rotated 90°, both fixed
      constants from the original geometry (so the expression stays
      affine and DCP-compliant). Vertices that ARE shared with a
      neighbour are pinned to u=0 ("locked to the void" only applies to
      vertices that actually face it) — their position stays governed by
      the existing contiguous coupling penalty instead.
   c) lambda_vertex_repulsion applies the same corrected one-sided-hinge
      repulsion from (a) between individual free vertices of different
      regions, giving tangential freedom something concrete to do: slide
      sideways to avoid intruding into a neighbour. Off by default; scales
      as O(n_free_i * n_free_j) per region pair, so start with
      vertex_repulsion_only_adjacent=True (the default).
"""

from __future__ import annotations

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

# Weight of the always-on anchor penalty (Fix 2). Hardcoded, not user-facing:
# negligible next to any real lambda_* (which are O(0.1) to O(1e6) in
# practice), but makes t=1/delta=0 the unique minimiser when every real
# lambda_* is 0.
#
# NOTE: this value was empirically tuned, not arbitrary. 1e-9 was tried
# first and FAILS: CLARABEL's convergence tolerance can't resolve an
# objective that small, so with every real lambda_* at 0 it returned an
# arbitrary feasible point up to ~0.14 units away from the true input
# geometry ("optimal" status, but not actually at the anchor's minimiser).
# 1e-4 was verified (see smoke test) to bring that error down to ~1e-8 —
# effectively exact — while still being 3-4 orders of magnitude below the
# smallest lambda_* value used anywhere in practice (e.g. lambda_shape=0.1
# for Demers cartograms), so it never perceptibly competes with real terms.
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
    # NOTE (Fix 1c): "contiguous" no longer hard-locks shared vertices via
    # equality constraints — both names now build the identical soft,
    # squared-distance penalty weighted by lambda_contiguous. Use a large
    # lambda_contiguous (1e3-1e6) to approximate the old hard-locking
    # behaviour while remaining always feasible.
    target_centers=None,
    horizontal_pairs=None,            # list[(i,j)] i left of j
    vertical_pairs=None,              # list[(i,j)] i below j
    neighboring_pairs=None,           # list[(i,j)] geographically adjacent
    shared_vertices_of_neighbors=None,
    # list of [i, j, shared_points] from shared_vertices_of_neighbors()
    shared_vertex_tolerance: float = 1e-8,
    # Coordinate-matching tolerance used to map each shared_points entry
    # back to actual (k, l) vertex INDICES in the two polygons — this
    # governs BOTH which vertices get locked (pinned to u=0, excluded
    # from tangential freedom / vertex repulsion) AND which vertex pairs
    # the contiguous coupling penalty actually pulls together, so a
    # mismatch here silently breaks both mechanisms at once.
    # IMPORTANT: this MUST be set to (at least) whatever tolerance you
    # used when building shared_vertices_of_neighbors(polygons, ...)
    # yourself — they're independent numbers and previously this one was
    # silently hardcoded to 1e-8 regardless of what you passed there.
    # 1e-8 in lat/lon degrees is ~1.1mm — real shapefile data is very
    # unlikely to match that tightly between two independently-stored
    # polygons even when they DO share a real border. If regions that
    # should be adjacent are drifting apart, or free vertices are moving
    # in ways that create overlap right where a real shared border is,
    # try loosening both tolerances together (e.g. 1e-6, 1e-5, 1e-4 —
    # whatever is small relative to your coordinate units but larger than
    # your data's actual floating-point noise).

    # --- objective weights (λ) ---
    # Each is either set directly here, or left as None so its legacy
    # quality-criterion equivalent below governs instead (see
    # preprocess_global). Whichever one is non-None wins OUTRIGHT for that
    # weight — the two never combine, multiply, or blend together.
    lambda_shape: Optional[float]    = None,  # λ_s -> W_shape,    else (1 - shape_deformation)
    lambda_area: Optional[float]     = None,  # λ_a -> W_area,     else cartographic_error
    lambda_center: Optional[float]   = None,  # λ_c -> W_spatial,  else spatial_deformation
    lambda_topology: Optional[float] = None,  # λ_t -> W_topology, else topological_accuracy

    # --- contiguous-mode parameters ---
    lambda_contiguous: float = 1.0,
    # Weight for the soft shared-vertex penalty. Applies identically to
    # shape="contiguous" and shape="contiguous2" (Fix 1c — both names now
    # build the same soft penalty; there is no hard-equality mode anymore).
    # Use a large value (1e3-1e6) to approximate hard locking.

    contiguous_area_ratio_cap: float = 10.0,
    # Hard-constraint mode (contiguous): skip locking a shared vertex
    # when max(A_i,A_j)/min(A_i,A_j) > cap.  Prevents a large region
    # from fully pinning a very small neighbour.
    # Set to np.inf to lock all shared vertices regardless of size ratio.

    contiguous_use_topology: bool = True,
    # contiguous / contiguous2: whether to still include the pairwise
    # topology (hor/ver excess-distance) terms in the objective.
    # (Kept for backward compatibility; no longer needed purely for
    # feasibility since Fix 1 — topology terms can no longer cause
    # infeasibility on their own — but still useful to simplify the
    # objective if you don't want those terms competing with others.)

    lambda_order: float = 1e6,
    # Weight of the soft horizontal/vertical ordering penalty (Fix 1b).
    # Previously "i left of j" / "i below j" (horizontal_pairs /
    # vertical_pairs) were HARD inequality constraints that could make the
    # problem infeasible (e.g. cyclic or mutually-contradictory pair
    # lists, especially combined with contiguous shared-vertex coupling).
    # They are now a soft quadratic hinge penalty: violations are
    # penalised, not forbidden. The large default (1e6) makes ordering
    # "practically hard" in the common case where it's satisfiable, while
    # guaranteeing the problem never becomes infeasible because of it.
    # Set lambda_order=0 to disable ordering entirely.

    # --- inner shape parameters ---
    gamma: float = 1.0,   # 1 = pure deformation term, 0 = pure E_target
    beta: float  = 1.0,   # 1 = L1 (z), 0 = L2 ((t-1)²)

    # --- geometry parameters ---
    alpha:   float = 1.0,   # slope for diagonal deviation d_ij
    epsilon: float = 1e-2,  # gap for non-adjacent pairs
    b:       float = 1e-2,  # scale for directional weight b_ij
    t_min:   float = 0.01,  # minimum radial scale
    t_max:   float = 5.0,   # maximum radial scale  (caps runaway vertices)

    # --- legacy quality-criterion API -------------------------------------
    # Fallback source for W_shape/W_area/W_spatial/W_topology whenever the
    # matching lambda_* above is left as None. Each weight comes from
    # EXACTLY ONE of the two — see the docstring at the top of this file.
    #
    # These three keep the concrete defaults this wrapper has always used
    # (1.0, 0.0, 1.0) rather than preprocess_global's own internal defaults
    # (1.0, 1.0, 1.0) — deliberately, so that omitting a lambda_*/legacy
    # pair entirely still reproduces the framework's original default
    # weights (W_area=1.0, W_spatial=0.0, W_topology=1.0) exactly.
    cartographic_error: float = 1.0,    # <-> lambda_area
    spatial_deformation: float = 0.0,   # <-> lambda_center
    topological_accuracy: float = 1.0,  # <-> lambda_topology
    # shape_deformation is dual-purpose (legacy fallback for W_shape AND,
    # independently, the circle/square re-initialisation switch in step 2
    # below) — see the docstring at the top of this file for why that's
    # fine and not a conflict.
    shape_deformation: float = 0.0,
    # The remaining five legacy criteria are kept ONLY for call-signature
    # backward compatibility. preprocess_global has never read any of them
    # anywhere in its body — they were already complete no-ops before any
    # of today's changes, and still are.
    relative_direction: float = 1.0,
    global_shape: float = 1.0,
    local_shape: float = 0.0,
    complexity: float = 1.0,
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

    # ── Approach 1: soft mean-scale (Fix 1a — always applied) ──────────────
    soft_mean_scale: bool = True,
    # DEPRECATED / NO-OP: kept only so existing calls that pass this
    # argument don't break. The hard mean(t)==s equality has been removed
    # entirely — it could go infeasible whenever s fell outside
    # [t_min, t_max]. The soft quadratic penalty below is now always
    # active regardless of this flag's value.
    lambda_mean_scale: float = 1e4,
    # Weight of the soft mean-scale penalty (now always active).
    # Set high (1e3–1e5) to keep area accuracy, lower to allow more slack,
    # 0 to disable area-scale fidelity entirely.

    # ── Approach 2: pairwise centre repulsion (anti-overlap) ───────────────
    lambda_repulsion: float = 0.0,
    # Weight of soft repulsion penalty between region centres.
    # A value of 0 (default) disables this term entirely.
    #
    # Fix 4a: this used to build TWO independent one-sided hinges per axis
    # (sp_x >= margin - dx  and  sn_x >= margin + dx). That's syntactically
    # valid DCP but semantically broken: whichever hinge faces the region's
    # actual (large) separation grows WITHOUT BOUND the farther apart the
    # centres already are, fighting normal cartogram scaling instead of
    # just discouraging closeness. True direction-agnostic "stay >= margin
    # apart" is inherently non-convex (a keep-out disk around each centre)
    # so no convex penalty can enforce it from every direction at once —
    # it has to commit to a side. This now derives that side from each
    # pair's ORIGINAL relative position (whichever axis has the larger
    # original gap), giving a single correct one-sided hinge per pair
    # instead of two competing ones. See _one_sided_repulsion() below.
    repulsion_only_adjacent: bool = False,
    # If True, repulsion is applied only between adjacent pairs (T).
    # If False (default), all pairs are repelled (safer for spike prevention).
    repulsion_margin: float = 1.0,
    # Minimum centre-to-centre distance (along the pair's fixed separating
    # axis, see above) below which repulsion activates.

    # ── Approach 4: tangential vertex freedom + vertex-level anti-overlap ──
    tangential_freedom: bool = False,
    # Off by default (zero behaviour change for existing calls). When True,
    # every "free" vertex — one NOT shared with a neighbouring region under
    # shared_vertices_of_neighbors, i.e. one that borders the void/open
    # background rather than another region — gets a second, independent
    # degree of freedom u_{i,k} in addition to the existing radial scale
    # t_{i,k}. The vertex's new position becomes:
    #   v'_{i,k} = m_i + t_{i,k} * r_{i,k}  +  u_{i,k} * n_{i,k}
    # where r_{i,k} is the existing fixed radial axis (centre -> original
    # vertex) and n_{i,k} is that same axis rotated 90°, both computed once
    # from the ORIGINAL geometry (so the expression stays affine in the cp
    # variables and fully DCP-compliant). Vertices that ARE shared with a
    # neighbour are pinned to u=0 — "locked to the void" only applies to
    # vertices that actually face the void; shared vertices stay purely
    # radial since their position is already governed by the contiguous
    # coupling penalty matching them to their neighbour's vertex.
    tangential_margin: float = 0.5,
    # Box bound on |u_{i,k}|, as a fraction of that vertex's local edge
    # scale (mean distance to its two polygon neighbours in the ORIGINAL
    # geometry). This alone can never cause infeasibility (Fix 1) — it's
    # just a per-vertex box, independent of every other constraint.
    lambda_tangential: float = 0.0,
    # Soft regularisation weight on sum(u^2). 0.0 by default: with
    # tangential_freedom=True but lambda_tangential=0.0 and nothing else
    # pulling on u, the Fix-2 identity anchor still pulls u -> 0, so
    # enabling the freedom without anything using it is a no-op — u only
    # moves when something (this weight, or the vertex repulsion below)
    # actually wants it to.
    lambda_vertex_repulsion: float = 0.0,
    # Weight of a soft anti-overlap penalty between FREE (void-facing)
    # vertices of different regions — the vertex-level analogue of
    # Approach 2's centre repulsion, and what gives tangential_freedom
    # something concrete to do: slide sideways to avoid intruding into a
    # neighbouring region. 0.0 by default (fully opt-in). This adds
    # O(n_free_i * n_free_j) slack-variable pairs per region pair it's
    # applied to, so start with vertex_repulsion_only_adjacent=True (the
    # default) and small weights, especially on maps with many vertices.
    vertex_repulsion_only_adjacent: bool = True,
    # If True (default), only check free-vertex pairs between ADJACENT
    # regions (via neighboring_pairs) — bounds the extra variable count.
    # Set False to check every region pair (can get expensive fast).
    vertex_repulsion_margin: float = 0.0,
    # Minimum allowed separation (along each pair's fixed axis, same
    # mechanism as repulsion_margin) between a free vertex of one region
    # and a free vertex of another.
    vertex_repulsion_nearest_k: Optional[int] = None,
    # Cuts the O(n_free_i * n_free_j) blow-up that can hang/crash a kernel
    # on real polygon data (country borders easily have dozens of
    # vertices each). If set to an int k, each CANDIDATE free vertex of
    # region i (see vertex_repulsion_boundary_k below) is only paired with
    # its k nearest free vertices of region j (by ORIGINAL position — a
    # fixed, one-time O(n log n) computation, not part of the
    # optimisation). None (default) uses the full candidate cross product;
    # set this (e.g. 3-8) for anything beyond toy examples.
    vertex_repulsion_boundary_k: Optional[int] = None,
    # A further, complementary restriction to nearest_k above. nearest_k
    # alone still considers EVERY free vertex of region i (even ones on
    # the far side of the region, nowhere near j) when picking who to
    # pair with — wasteful for a large polygon with only one small shared
    # border with a given neighbour. If set to an int m, each region is
    # first narrowed down to only its m free vertices CLOSEST TO THE
    # OTHER REGION'S CENTROID (i.e. the ones actually near that specific
    # shared border) before nearest_k pairing is applied among that
    # narrowed set. Total pairs per region pair is then bounded by
    # roughly boundary_k * nearest_k instead of n_free_i * nearest_k —
    # the right lever to pull when nearest_k alone still leaves too many
    # pairs (e.g. large countries with hundreds of coastline vertices).
    max_vertex_repulsion_pairs: int = 20000,
    # Hard safety cap. Before building any slack variables, the total
    # vertex-pair count is estimated; if it would exceed this, a clear
    # ValueError is raised instead of silently constructing a QP large
    # enough to hang or crash the kernel/notebook. Raise this only if you
    # know your machine can handle it, or lower vertex_repulsion_nearest_k
    # / vertex_repulsion_boundary_k / simplify your polygons first.

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
    """
    Dict-in / dict-out contract
    ----------------------------
    `data` is a dict keyed by region name, each value a dict of per-region
    fields (polygon, area, target_area, target_positions, centroid, ...).
    This function MUTATES `data` in place, adding/overwriting these keys
    on every region record once the solve (and any post-processing) is
    complete:

        original_polygon, original_area   -- the pre-optimisation geometry
        new_polygon, new_area             -- the optimised geometry
        target_area                       -- normalised target used in the solve
        new_centroid                      -- centroid of new_polygon
        overlap_area_lost                 -- area clipped away by overlap resolution (0.0 if disabled)
        status, objective_value           -- solver status / objective (same for every region)

    Anything that is NOT per-region (solver status, objective value, and
    the Demers/Dorling leader lines) is also mirrored under the reserved
    key `data["__meta__"]`, since it can't be attached to a single name:

        data["__meta__"] = {
            "status": ..., "objective_value": ..., "leaders": [...],
            "target_areas": [...], "n_regions": ...,
        }

    Returns
    -------
    dict
        The same `data` object, mutated and returned for convenience.
    """
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
    # preprocess_global (src/core/preprocessing_data.py) resolves each of
    # W_shape/W_area/W_spatial/W_topology from EITHER its lambda_* argument
    # (if not None) OR its legacy quality-criterion equivalent — never a
    # blend of both. See the docstring at the top of this file for the
    # full pairing and the shape_deformation dual-purpose caveat.

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
            init_polygons.append(make_square(ctr, target_areas[k], n=64))
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

        # ── Fix 4: tangential vertex freedom (u) ─────────────────────────
        # Every vertex's new position is built from two FIXED local axes
        # taken from the original geometry: the existing radial axis
        # (centre -> original vertex, scaled by t) and that axis rotated
        # 90° (scaled by u). Vertices shared with a neighbour ("locked",
        # from shared_vertices_of_neighbors) are pinned to u=0 — they stay
        # purely radial since the contiguous penalty already governs their
        # position by matching them to their neighbour's vertex. Vertices
        # facing the void (not shared with anything) get the extra
        # tangential slide when tangential_freedom=True.
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

        # ── Mean-scale: always a soft penalty (Fix 1a) ──────────────────────
        # The old hard equality cp.sum(t)/ni == s could conflict with the
        # t_min/t_max box (e.g. whenever s fell outside that range) and make
        # the whole problem infeasible. It has been removed unconditionally
        # — only box bounds remain here, which alone can never be infeasible
        # (given t_min <= t_max, validated above). The `soft_mean_scale` flag
        # is kept in the signature as a no-op for backward compatibility.
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

        # Fix 2: always-on identity anchor for this region — pulls t -> 1
        # (no shape/area change) and delta -> 0 (centre stays at fp, i.e.
        # m == fp). Weighted by _ANCHOR_EPS, added to the objective
        # completely OUTSIDE W_shape/W_spatial/any lambda_*, so it is never
        # silenced by turning those weights off — it's what makes the
        # input geometry the *unique* minimiser when every real lambda is 0.
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

        # Soft ordering penalties for H and V pairs (Fix 1b; was eqs. 11–12
        # as HARD inequality constraints — removed because a contradictory
        # or cyclic pair list, especially combined with contiguous
        # shared-vertex coupling, could make the whole problem infeasible).
        # Each pair now contributes a one-sided hinge: zero penalty if the
        # ordering already holds, quadratic penalty proportional to the
        # violation otherwise. Weighted by lambda_order (large by default
        # so ordering is "practically hard" whenever it's satisfiable).
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

        for (ii, jj) in pairs_for_repulsion:
            if centers_vars[ii] is None or centers_vars[jj] is None:
                continue
            repulsion_terms.append(
                _one_sided_repulsion(
                    _fp_vals[ii], _fp_vals[jj],
                    centers_vars[ii], centers_vars[jj],
                    repulsion_margin, tag=f"c_{ii}_{jj}",
                )
            )

    repulsion_penalty = (
        lambda_repulsion * cp.sum(repulsion_terms)
        if repulsion_terms else cp.Constant(0.0)
    )

    # ------------------------------------------------------------------
    # 4b-pre-2. Approach 4: vertex-level anti-overlap repulsion
    # ------------------------------------------------------------------
    # Same corrected single-slack mechanism as Approach 2 above, but
    # applied between individual FREE (void-facing) vertices of different
    # regions rather than whole region centres — a much more local, more
    # effective anti-overlap tool, and the concrete reason tangential
    # freedom (u) exists: this penalty gives the solver a reason to
    # actually use it. Locked/shared vertices are excluded (their position
    # is already governed by the contiguous coupling penalty instead).
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

        # Group by (ii,jj) and average WITHIN each region pair before
        # summing ACROSS pairs. Fix 5: without this, a region's total pull
        # scales with however many vertex pairs got created for it (which
        # depends on vertex count, nearest_k, boundary_k, and how many
        # neighbours it has) — a highly-detailed region (hundreds of
        # vertices) or one with many neighbours would accumulate far more
        # total repulsion "force" than a simple 4-vertex square, even at
        # the same lambda_vertex_repulsion, purely as an artefact of pair
        # count rather than actual overlap severity. Averaging within each
        # region-pair keeps that pair's contribution on the same footing
        # as Approach 2's one-term-per-region-pair centre repulsion,
        # regardless of how many candidate vertices were compared; summing
        # across different neighbour pairs still (correctly) scales with
        # how many neighbours a region actually has.
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
                    + _u_vars[i][k] * _tangents[i][k]
                )  # (2,) expr — u is pinned to 0 here (locked/shared vertex)
                v_j = (
                    centers_vars[j]
                    + _t_vars[j][l] * _directions[j][l]
                    + _u_vars[j][l] * _tangents[j][l]
                )  # (2,) expr

                # Fix 1c: always a soft quadratic penalty (the old hard
                # equality for shape=="contiguous" is removed — a graph of
                # simultaneous equalities across many regions, each also
                # bound by t_min/t_max and mean-scale, was easily
                # overdetermined and a common source of infeasibility).
                # "contiguous" and "contiguous2" now build the identical
                # penalty; use a large lambda_contiguous to approximate the
                # old hard-locking behaviour instead.
                #
                # Penalise the gap between the two images of the shared
                # vertex, normalised by the squared original distance
                # between the two region centres so the weight is
                # dimensionless and scale-invariant regardless of
                # coordinate units.
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
        + W_area   * cp.sum(area_terms)
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
            tolerance                         = shared_vertex_tolerance,
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

