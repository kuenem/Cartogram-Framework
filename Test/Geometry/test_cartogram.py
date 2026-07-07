"""
test_cartogram.py
=================
Isolated and combined tests for CartogramFramework_global_CLAUDE.

Each test builds its own minimal polygon set so the suite is
self-contained — no external files required.

Run:
    python test_cartogram.py
or with pytest:
    pytest test_cartogram.py -v
"""

import numpy as np
import pytest

# ------------------------------------------------------------------ #
#  Real implementations (from util.py)                               #
# ------------------------------------------------------------------ #

from util import CartogramFramework_global_CLAUDE


def make_rect(cx, cy, w, h):
    """Axis-aligned rectangle as a numpy array of shape (4, 2)."""
    hw, hh = w / 2, h / 2
    return np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=float)


def polygon_areanp(poly):
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    y_next = np.roll(y, -1)
    x_next = np.roll(x, -1)
    return 0.5 * abs(np.dot(x, y_next) - np.dot(y, x_next))

# keep a short alias used across the tests
polygon_area = polygon_areanp


def polygon_bounds(poly):
    return {
        'south': min(x[1] for x in poly),
        'east':  max(x[0] for x in poly),
        'north': max(x[1] for x in poly),
        'west':  min(x[0] for x in poly),
    }


def disjoint_pairs_horizontal_and_vertical(polygons):
    bounds = [polygon_bounds(p) for p in polygons]
    horizontal_pairs = []
    vertical_pairs   = []
    for i in range(len(bounds)):
        for j in range(len(bounds)):
            if i == j:
                continue
            if (bounds[i]['east']  < bounds[j]['west'] or
                    bounds[j]['east']  < bounds[i]['west']):
                horizontal_pairs.append((i, j))
            if (bounds[i]['north'] < bounds[j]['south'] or
                    bounds[j]['north'] < bounds[i]['south']):
                vertical_pairs.append((i, j))
    return horizontal_pairs, vertical_pairs


def neighbouring_pairs(polygons, tolerance=1e-8):
    n = len(polygons)
    neighbors = []
    for i in range(n):
        set_i = set()
        for pt in polygons[i]:
            set_i.add((
                round(pt[0] / tolerance) * tolerance,
                round(pt[1] / tolerance) * tolerance,
            ))
        for j in range(i + 1, n):
            for pt in polygons[j]:
                key = (
                    round(pt[0] / tolerance) * tolerance,
                    round(pt[1] / tolerance) * tolerance,
                )
                if key in set_i:
                    neighbors.append((i, j))
                    break
    return neighbors


def common_vertices(poly1, poly2, tolerance=1e-8):
    set1 = {}
    for pt in poly1:
        key = (
            round(pt[0] / tolerance) * tolerance,
            round(pt[1] / tolerance) * tolerance,
        )
        set1.setdefault(key, pt)
    common = []
    seen = set()
    for pt in poly2:
        key = (
            round(pt[0] / tolerance) * tolerance,
            round(pt[1] / tolerance) * tolerance,
        )
        if key in set1 and key not in seen:
            common.append(list(set1[key]))
            seen.add(key)
    return common


def shared_vertices_of_neighbors(polygons, tolerance=1e-8):
    neighbors = neighbouring_pairs(polygons, tolerance)
    result = []
    for i, j in neighbors:
        shared = common_vertices(polygons[i], polygons[j], tolerance)
        result.append([i, j, shared])
    return result


def plot_polys(polygons, title="Polygons", plot_points=False,
               centroids=False, label_vertices=False,
               legend=True, target_centers=None):
    import matplotlib.pyplot as plt
    plt.figure()
    for idx, points in enumerate(polygons):
        points = np.array(points)
        poly = np.vstack([points, points[0]])
        plt.fill(poly[:, 0], poly[:, 1], alpha=0.3, label=f"Poly {idx}")
        if plot_points:
            plt.plot(poly[:, 0], poly[:, 1], marker='o')
        if centroids:
            centroid = np.mean(points, axis=0)
            plt.plot(centroid[0], centroid[1], marker='*', color='red', markersize=5)
        if target_centers is not None:
            for tc in target_centers:
                plt.plot(tc[0], tc[1], marker='X', color='green', markersize=5)
        if label_vertices:
            for i, (x, y) in enumerate(points):
                plt.text(x, y, f"P{idx}_{i}")
    plt.axis('scaled')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    if legend:
        plt.legend()
    plt.show()


# ------------------------------------------------------------------ #
#  Shared fixtures                                                    #
# ------------------------------------------------------------------ #

def make_grid_polygons(rows=2, cols=2, cell=10.0):
    """
    Create a rows×cols grid of non-overlapping rectangles that share edges
    (touching vertices → real neighbours detected by neighbouring_pairs).
    Returns (polygons, target_positions).
    """
    polygons = []
    centers  = []
    for r in range(rows):
        for c in range(cols):
            cx = c * cell + cell / 2
            cy = r * cell + cell / 2
            # No gap between cells so they share edges / vertices
            polygons.append(make_rect(cx, cy, cell, cell))
            centers.append([cx, cy])
    return polygons, centers


def run_cartogram(
    polygons,
    target_positions,
    target_areas=None,
    shape="original",
    cartographic_error=1.0,
    shape_deformation=0.01,
    topological_accuracy=0.0,
    spatial_deformation=1.0,
    local_shape=0.0,
    alpha=1.0,
    epsilon=1e-2,
    b=1e-2,
    neighboring_pairs_arg=None,
):
    """Thin wrapper that mirrors the baseline call in the specification."""
    if target_areas is None:
        target_areas = [polygon_area(p) for p in polygons]

    if neighboring_pairs_arg is None:
        neighboring_pairs_arg = neighbouring_pairs(polygons)

    h_pairs, v_pairs = disjoint_pairs_horizontal_and_vertical(polygons)

    new_polys, areas, status, value = CartogramFramework_global_CLAUDE(
        polygons=polygons,
        target_areas=target_areas,
        fixed_points=None,
        shape=shape,
        target_centers=target_positions,
        horizontal_pairs=h_pairs,
        vertical_pairs=v_pairs,
        neighboring_pairs=neighboring_pairs_arg,
        shared_vertices_of_neighbors=shared_vertices_of_neighbors(polygons),
        cartographic_error=cartographic_error,
        shape_deformation=shape_deformation,
        topological_accuracy=topological_accuracy,
        spatial_deformation=spatial_deformation,
        local_shape=local_shape,
        alpha=alpha,
        epsilon=epsilon,
        b=b,
    )
    return new_polys, areas, status, value


# ================================================================== #
#  TEST GROUP 1 – Solver basics                                       #
# ================================================================== #

class TestSolverBasics:

    def test_solver_returns_optimal(self):
        """Solver should converge on a trivial 2-polygon case."""
        polygons, centers = make_grid_polygons(1, 2)
        _, _, status, _ = run_cartogram(polygons, centers)
        assert status in ("optimal", "optimal_inaccurate"), \
            f"Unexpected status: {status}"

    def test_output_polygon_count(self):
        """Number of output polygons equals number of input polygons."""
        polygons, centers = make_grid_polygons(2, 2)
        new_polys, _, _, _ = run_cartogram(polygons, centers)
        assert len(new_polys) == len(polygons)

    def test_output_vertex_count_preserved(self):
        """Each output polygon keeps the same vertex count as its input."""
        polygons, centers = make_grid_polygons(2, 2)
        new_polys, _, _, _ = run_cartogram(polygons, centers)
        for orig, new in zip(polygons, new_polys):
            assert len(orig) == len(new), "Vertex count changed."

    def test_scalar_target_area_broadcast(self):
        """Passing a scalar target_area (e.g. 10) should work without error."""
        polygons, centers = make_grid_polygons(1, 2)
        _, areas, status, _ = run_cartogram(
            polygons, centers, target_areas=10
        )
        assert status in ("optimal", "optimal_inaccurate")
        assert len(areas) == len(polygons)

    def test_output_coordinates_are_finite(self):
        """All output vertex coordinates must be finite (no NaN / Inf)."""
        polygons, centers = make_grid_polygons(2, 2)
        new_polys, _, _, _ = run_cartogram(polygons, centers)
        for poly in new_polys:
            assert np.all(np.isfinite(poly)), "Non-finite coordinate in output."

    def test_objective_value_is_finite(self):
        """Solver objective value must be a finite number."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, _, value = run_cartogram(polygons, centers)
        assert np.isfinite(value), f"Objective value not finite: {value}"


# ================================================================== #
#  TEST GROUP 2 – Cartographic error (area accuracy)                 #
# ================================================================== #

class TestCartographicError:

    def test_area_preserved_when_target_equals_original(self):
        """
        When target area == original area, output areas should be
        close to original (relative error < 15 %).
        """
        polygons, centers = make_grid_polygons(2, 2)
        orig_areas = [polygon_area(p) for p in polygons]

        _, out_areas, _, _ = run_cartogram(
            polygons, centers,
            target_areas=orig_areas,
            cartographic_error=1.0,
        )
        for orig, out in zip(orig_areas, out_areas):
            assert abs(out - orig) / orig < 0.15, \
                f"Area error too large: orig={orig:.2f}, out={out:.2f}"

    def test_area_grows_with_larger_target(self):
        """Output areas should grow when target is larger than original."""
        polygons, centers = make_grid_polygons(1, 2)
        orig_areas = [polygon_area(p) for p in polygons]
        big_target = [a * 4 for a in orig_areas]

        _, out_areas, _, _ = run_cartogram(
            polygons, centers,
            target_areas=big_target,
            cartographic_error=1.0,
            shape_deformation=0.0,
        )
        for orig, out in zip(orig_areas, out_areas):
            assert out > orig, "Output area should exceed original."

    def test_area_shrinks_with_smaller_target(self):
        """Output areas should shrink when target is smaller than original."""
        polygons, centers = make_grid_polygons(1, 2)
        orig_areas = [polygon_area(p) for p in polygons]
        small_target = [a * 0.25 for a in orig_areas]

        _, out_areas, _, _ = run_cartogram(
            polygons, centers,
            target_areas=small_target,
            cartographic_error=1.0,
            shape_deformation=0.0,
        )
        for orig, out in zip(orig_areas, out_areas):
            assert out < orig, "Output area should be smaller than original."

    def test_high_vs_low_cartographic_error_weight(self):
        """
        High cartographic_error weight → tighter area match than
        low cartographic_error weight.
        """
        polygons, centers = make_grid_polygons(2, 2)
        orig_areas   = [polygon_area(p) for p in polygons]
        target_areas = [a * 2 for a in orig_areas]

        _, areas_high, _, _ = run_cartogram(
            polygons, centers,
            target_areas=target_areas, cartographic_error=1.0,
        )
        _, areas_low, _, _ = run_cartogram(
            polygons, centers,
            target_areas=target_areas, cartographic_error=0.01,
        )

        err_high = sum(abs(o - t) for o, t in zip(areas_high, target_areas))
        err_low  = sum(abs(o - t) for o, t in zip(areas_low,  target_areas))
        assert err_high <= err_low, \
            "Higher cartographic_error weight should yield smaller area error."

    def test_all_areas_positive(self):
        """No output polygon should have zero or negative area."""
        polygons, centers = make_grid_polygons(2, 3)
        _, areas, _, _ = run_cartogram(
            polygons, centers,
            target_areas=[polygon_area(p) * 2 for p in polygons],
        )
        for a in areas:
            assert a > 0, f"Non-positive area encountered: {a}"


# ================================================================== #
#  TEST GROUP 3 – Shape deformation                                   #
# ================================================================== #

class TestShapeDeformation:

    def test_shape_original_no_crash(self):
        """shape='original' should run without error."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(polygons, centers, shape="original")
        assert status in ("optimal", "optimal_inaccurate")

    def test_shape_circle_no_crash(self):
        """shape='circle' should run without error."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers,
            shape="circle", shape_deformation=0.0,
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_shape_square_no_crash(self):
        """shape='square' should run without error."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers,
            shape="square", shape_deformation=0.0,
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_shape_deformation_zero_vs_one_both_finite(self):
        """
        Both extremes of shape_deformation (0 and 1) should yield
        finite objective values.
        """
        polygons, centers = make_grid_polygons(2, 2)

        _, _, _, val_0 = run_cartogram(polygons, centers, shape_deformation=0.0)
        _, _, _, val_1 = run_cartogram(polygons, centers, shape_deformation=1.0)

        assert np.isfinite(val_0), f"shape_deformation=0 gave non-finite obj: {val_0}"
        assert np.isfinite(val_1), f"shape_deformation=1 gave non-finite obj: {val_1}"

    def test_circle_output_roughly_equidistant_vertices(self):
        """
        With shape='circle' and shape_deformation=0, the radial distances
        from the centroid should be nearly equal (coefficient of variation < 20%).
        """
        polygons, centers = make_grid_polygons(1, 2, cell=10.0)
        new_polys, _, status, _ = run_cartogram(
            polygons, centers,
            shape="circle", shape_deformation=0.0,
        )
        assert status in ("optimal", "optimal_inaccurate")
        for poly in new_polys:
            pts = np.array(poly)
            centroid = np.mean(pts, axis=0)
            radii = np.linalg.norm(pts - centroid, axis=1)
            cv = radii.std() / (radii.mean() + 1e-12)
            assert cv < 0.20, f"Circle radii not uniform enough (CV={cv:.3f})."


# ================================================================== #
#  TEST GROUP 4 – Spatial deformation (center attraction)            #
# ================================================================== #

class TestSpatialDeformation:

    def test_centers_close_to_target_when_weight_high(self):
        """
        With high spatial_deformation weight, output centroids should be
        within a reasonable distance of target_centers.
        """
        polygons, centers = make_grid_polygons(2, 2)
        shifted = [[c[0] + 1, c[1] + 1] for c in centers]

        new_polys, _, _, _ = run_cartogram(
            polygons, shifted,
            spatial_deformation=1.0,
            cartographic_error=0.01,
        )
        for poly, target in zip(new_polys, shifted):
            centroid = np.mean(poly, axis=0)
            dist = np.linalg.norm(centroid - np.array(target))
            assert dist < 5.0, \
                f"Centroid {centroid} too far from target {target} (dist={dist:.2f})."

    def test_centers_less_attracted_when_weight_low(self):
        """
        Low spatial_deformation weight → centroids less constrained to targets.
        Total centroid error with weight=0 should be >= error with weight=1.
        """
        polygons, centers = make_grid_polygons(2, 2)
        far_targets = [[c[0] + 20, c[1] + 20] for c in centers]

        new_hi, _, _, _ = run_cartogram(
            polygons, far_targets, spatial_deformation=1.0
        )
        new_lo, _, _, _ = run_cartogram(
            polygons, far_targets, spatial_deformation=0.0
        )

        def total_center_error(polys, targets):
            return sum(
                np.linalg.norm(np.mean(p, axis=0) - np.array(t))
                for p, t in zip(polys, targets)
            )

        err_hi = total_center_error(new_hi, far_targets)
        err_lo = total_center_error(new_lo, far_targets)
        assert err_hi <= err_lo + 1e-3, \
            "High spatial weight should attract centers more than low weight."

    def test_spatial_zero_still_solves(self):
        """spatial_deformation=0 disables center attraction; solver must not crash."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers, spatial_deformation=0.0
        )
        assert status in ("optimal", "optimal_inaccurate")


# ================================================================== #
#  TEST GROUP 5 – Topological accuracy                                #
# ================================================================== #

class TestTopologicalAccuracy:

    def test_topology_off_no_crash(self):
        """topological_accuracy=0.0 disables pairwise terms; should not crash."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers, topological_accuracy=0.0
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_topology_on_no_crash(self):
        """topological_accuracy=1.0 enables pairwise terms; should not crash."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers, topological_accuracy=1.0
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_topology_reduces_objective_when_regions_overlap(self):
        """
        With high topology weight on closely packed regions,
        the objective should still be finite (solver handles separation).
        """
        polygons, centers = make_grid_polygons(3, 3)
        _, _, _, val = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0,
            target_areas=[polygon_area(p) * 2 for p in polygons],
        )
        assert np.isfinite(val)

    def test_horizontal_pairs_detected_correctly(self):
        """
        For a 1×3 row of non-overlapping rectangles, h_pairs should be
        non-empty (leftmost and rightmost are disjoint horizontally).
        """
        polygons, _ = make_grid_polygons(1, 3)
        h_pairs, _ = disjoint_pairs_horizontal_and_vertical(polygons)
        assert len(h_pairs) > 0, \
            "Expected at least one horizontal disjoint pair in a 1×3 layout."

    def test_vertical_pairs_detected_correctly(self):
        """
        For a 3×1 column of non-overlapping rectangles, v_pairs should be
        non-empty (top and bottom are disjoint vertically).
        """
        polygons, _ = make_grid_polygons(3, 1)
        _, v_pairs = disjoint_pairs_horizontal_and_vertical(polygons)
        assert len(v_pairs) > 0, \
            "Expected at least one vertical disjoint pair in a 3×1 layout."


# ================================================================== #
#  TEST GROUP 6 – gap_ij / adjacency separation (NEW FEATURE)        #
# ================================================================== #

class TestGapAdjacency:

    def test_gap_zero_for_all_adjacent_pairs(self):
        """
        With all pairs declared adjacent, gap=0 everywhere.
        Changing epsilon should not affect the objective at all.
        """
        polygons, centers = make_grid_polygons(1, 3)
        adj = neighbouring_pairs(polygons)   # real shared-vertex adjacency

        _, _, _, val_eps0 = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, epsilon=0.0, neighboring_pairs_arg=adj,
        )
        _, _, _, val_eps5 = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, epsilon=5.0, neighboring_pairs_arg=adj,
        )
        assert abs(val_eps0 - val_eps5) < 1.0, \
            "Adjacent pairs have gap=0; epsilon value should not affect objective."

    def test_gap_nonzero_pushes_non_adjacent_apart(self):
        """
        Non-adjacent pairs with large epsilon should have greater
        mean pairwise centroid distance than with epsilon=0.
        """
        polygons, centers = make_grid_polygons(1, 3)
        adj = []   # explicitly declare no adjacency

        new_lo, _, _, _ = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, epsilon=0.0, neighboring_pairs_arg=adj,
        )
        new_hi, _, _, _ = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, epsilon=10.0, neighboring_pairs_arg=adj,
        )

        def mean_pairwise_dist(polys):
            cents = [np.mean(p, axis=0) for p in polys]
            dists = [
                np.linalg.norm(cents[i] - cents[j])
                for i in range(len(cents))
                for j in range(i + 1, len(cents))
            ]
            return np.mean(dists)

        assert mean_pairwise_dist(new_hi) >= mean_pairwise_dist(new_lo) - 1e-3, \
            "Larger epsilon should push non-adjacent regions further apart."

    def test_b_ij_scaling_solves_without_error(self):
        """
        b_ij = b * a_ij with mixed adjacency; solver must still converge.
        """
        polygons, centers = make_grid_polygons(2, 2)
        adj = [(0, 1), (2, 3)]   # partial adjacency

        _, _, status, val = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, b=0.5, neighboring_pairs_arg=adj,
        )
        assert status in ("optimal", "optimal_inaccurate")
        assert np.isfinite(val)

    def test_epsilon_zero_same_as_no_gap(self):
        """epsilon=0 should reproduce the same solution as no gap at all."""
        polygons, centers = make_grid_polygons(2, 2)

        _, _, _, val_zero  = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, epsilon=0.0,
        )
        _, _, _, val_small = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, epsilon=1e-9,
        )
        assert abs(val_zero - val_small) < 1e-2, \
            "epsilon≈0 and epsilon=0 should give nearly identical objectives."


# ================================================================== #
#  TEST GROUP 7 – Diagonal distance d_ij / alpha (NEW FEATURE)       #
# ================================================================== #

class TestDiagonalDistance:

    def test_alpha_zero_no_crash(self):
        """alpha=0 degrades d_ij to |y_i - y_j|; should still solve."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers, topological_accuracy=1.0, alpha=0.0,
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_alpha_one_no_crash(self):
        """Standard alpha=1 case; should solve without error."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, _ = run_cartogram(
            polygons, centers, topological_accuracy=1.0, alpha=1.0,
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_alpha_large_still_solves(self):
        """Very large alpha; solver should still find a feasible solution."""
        polygons, centers = make_grid_polygons(2, 2)
        _, _, status, val = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, alpha=100.0, b=1.0,
        )
        assert status in ("optimal", "optimal_inaccurate")
        assert np.isfinite(val)

    def test_alpha_increases_objective_monotonically(self):
        """
        Objective with alpha=10 should be >= objective with alpha=0
        because larger alpha adds more penalty to the problem.
        """
        polygons, centers = make_grid_polygons(2, 2)

        _, _, _, val_a0  = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, alpha=0.0,  b=1.0,
        )
        _, _, _, val_a10 = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, alpha=10.0, b=1.0,
        )
        assert np.isfinite(val_a0) and np.isfinite(val_a10)
        assert val_a10 >= val_a0 - 1e-3, \
            "Larger alpha should increase (or maintain) the objective value."

    def test_b_zero_makes_d_inactive(self):
        """
        With b=0 the diagonal term vanishes; alpha should have no effect.
        """
        polygons, centers = make_grid_polygons(2, 2)

        _, _, _, val_a0  = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, alpha=0.0,   b=0.0,
        )
        _, _, _, val_a10 = run_cartogram(
            polygons, centers,
            topological_accuracy=1.0, alpha=10.0,  b=0.0,
        )
        assert abs(val_a0 - val_a10) < 1e-3, \
            "With b=0, alpha should have no effect on the objective."


# ================================================================== #
#  TEST GROUP 8 – Combined parameter interactions                     #
# ================================================================== #

class TestCombined:

    def test_baseline_call(self):
        """Exact replica of the baseline call from the specification."""
        polygons, target_positions = make_grid_polygons(2, 2)
        h_pairs, v_pairs = disjoint_pairs_horizontal_and_vertical(polygons)

        new_polys, areas, status, _ = CartogramFramework_global_CLAUDE(
            polygons=polygons,
            target_areas=10,
            fixed_points=None,
            shape="original",
            target_centers=target_positions,
            horizontal_pairs=h_pairs,
            vertical_pairs=v_pairs,
            neighboring_pairs=neighbouring_pairs(polygons),
            shared_vertices_of_neighbors=shared_vertices_of_neighbors(polygons),
            cartographic_error=1.0,
            shape_deformation=0.01,
            topological_accuracy=0.0,
            spatial_deformation=1.0,
            local_shape=0.0,
        )
        plot_polys(
            new_polys,
            title="Baseline – Optimized Polygons",
            centroids=True,
            legend=False,
            target_centers=target_positions,
        )
        print("Areas:", areas)
        assert status in ("optimal", "optimal_inaccurate")

    def test_all_features_enabled(self):
        """All non-trivial weights > 0 and all new features active; must converge."""
        polygons, centers = make_grid_polygons(2, 3)

        new_polys, areas, status, val = run_cartogram(
            polygons, centers,
            target_areas=[polygon_area(p) * 1.5 for p in polygons],
            shape="original",
            cartographic_error=1.0,
            shape_deformation=0.3,
            topological_accuracy=1.0,
            spatial_deformation=0.5,
            alpha=1.0,
            epsilon=0.5,
            b=0.1,
        )
        assert status in ("optimal", "optimal_inaccurate"), \
            f"Combined test failed with status: {status}"
        assert len(new_polys) == len(polygons)
        assert all(np.isfinite(a) for a in areas)

    def test_shape_circle_with_topology(self):
        """Circle shapes + topology penalty should solve together."""
        polygons, centers = make_grid_polygons(1, 3)
        _, _, status, _ = run_cartogram(
            polygons, centers,
            shape="circle",
            shape_deformation=0.0,
            topological_accuracy=1.0,
            epsilon=0.1,
            alpha=0.5,
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_shape_square_with_gap_and_alpha(self):
        """Square shapes + gap + diagonal penalty should solve together."""
        polygons, centers = make_grid_polygons(1, 3)
        _, _, status, _ = run_cartogram(
            polygons, centers,
            shape="square",
            shape_deformation=0.0,
            topological_accuracy=1.0,
            epsilon=0.5,
            alpha=2.0,
            b=0.2,
        )
        assert status in ("optimal", "optimal_inaccurate")

    def test_high_area_and_high_spatial(self):
        """Competing high weights for area and spatial; should still solve."""
        polygons, centers = make_grid_polygons(2, 2)
        orig_areas = [polygon_area(p) * 3 for p in polygons]

        _, areas, status, _ = run_cartogram(
            polygons, centers,
            target_areas=orig_areas,
            cartographic_error=1.0,
            spatial_deformation=1.0,
            shape_deformation=0.0,
        )
        assert status in ("optimal", "optimal_inaccurate")
        assert all(np.isfinite(a) for a in areas)

    def test_circle_topology_gap_alpha_combined(self):
        """All new features active together with circle shape."""
        polygons, centers = make_grid_polygons(2, 2)

        _, _, status, val = run_cartogram(
            polygons, centers,
            shape="circle",
            shape_deformation=0.0,
            cartographic_error=1.0,
            topological_accuracy=1.0,
            spatial_deformation=0.5,
            epsilon=0.3,
            alpha=1.5,
            b=0.05,
        )
        assert status in ("optimal", "optimal_inaccurate")
        assert np.isfinite(val)

    def test_square_topology_gap_alpha_combined(self):
        """All new features active together with square shape."""
        polygons, centers = make_grid_polygons(2, 2)

        _, _, status, val = run_cartogram(
            polygons, centers,
            shape="square",
            shape_deformation=0.0,
            cartographic_error=1.0,
            topological_accuracy=1.0,
            spatial_deformation=0.5,
            epsilon=0.3,
            alpha=1.5,
            b=0.05,
        )
        assert status in ("optimal", "optimal_inaccurate")
        assert np.isfinite(val)


# ================================================================== #
#  Quick standalone runner                                            #
# ================================================================== #

if __name__ == "__main__":
    groups = [
        TestSolverBasics,
        TestCartographicError,
        TestShapeDeformation,
        TestSpatialDeformation,
        TestTopologicalAccuracy,
        TestGapAdjacency,
        TestDiagonalDistance,
        TestCombined,
    ]

    passed = failed = 0
    for group in groups:
        inst = group()
        methods = sorted(m for m in dir(inst) if m.startswith("test_"))
        print(f"\n{'='*62}")
        print(f"  {group.__name__}")
        print(f"{'='*62}")
        for method in methods:
            try:
                getattr(inst, method)()
                print(f"  ✓  {method}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {method}")
                print(f"       {type(exc).__name__}: {exc}")
                failed += 1

    total = passed + failed
    print(f"\n{'='*62}")
    print(f"  Results: {passed}/{total} passed  |  {failed} failed")
    print(f"{'='*62}\n")
