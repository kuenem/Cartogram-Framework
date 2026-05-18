from tabnanny import verbose

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geojson
import cvxpy as cp
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import euclidean
import pulp
from utils import *
from preprocess import *
# from utils.area import *
# from utils.reducingPolygon import *


def plot_poly(points, title="Polygon", label_vertices=False):
    # close polygon
    poly = np.vstack([points, points[0]])

    plt.figure()

    # filled polygon
    plt.fill(poly[:,0], poly[:,1], alpha=0.4)

    # edges
    plt.plot(poly[:,0], poly[:,1], marker='o')

    # label vertices
    if label_vertices:
        for i,(x,y) in enumerate(points):
            plt.text(x, y, f"P{i}")

    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.show()


def plot_polys(polygons, title="Polygons", plot_points=False, centroids=False, label_vertices=False, legend=True, target_centers=None):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure()
    
    for idx, points in enumerate(polygons):
        points = np.array(points)
        
        # close polygon
        poly = np.vstack([points, points[0]])
        
        # filled polygon (slightly transparent so overlaps are visible)
        plt.fill(poly[:,0], poly[:,1], alpha=0.3, label=f"Poly {idx}")
        
        if plot_points:
            # edges
            plt.plot(poly[:,0], poly[:,1], marker='o')

        if centroids:
            centroid = np.mean(points, axis=0)
            plt.plot(centroid[0], centroid[1], marker='*', color='red', markersize=5)

        if target_centers is not None:
            for tc in target_centers:
                plt.plot(tc[0], tc[1], marker='X', color='green', markersize=5)
        
        # optional vertex labels
        if label_vertices:
            for i, (x, y) in enumerate(points):
                plt.text(x, y, f"P{idx}_{i}")
    
    plt.axis('scaled')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title(title)
    if legend:
        plt.legend()
    plt.show()


def compare_polys(poly1, poly2, title="Polygon Comparison", centroid=None, support_lines=False, label_vertices=False):
    # close polygons
    poly1 = np.vstack([poly1, poly1[0]])
    poly2 = np.vstack([poly2, poly2[0]])

    plt.figure()

    # filled polygons
    plt.fill(poly1[:,0], poly1[:,1], alpha=0.4, label='Polygon 1')
    plt.fill(poly2[:,0], poly2[:,1], alpha=0.4, label='Polygon 2')

    # edges
    plt.plot(poly1[:,0], poly1[:,1], marker='o', label='Polygon 1 Edges', alpha=0.7, color='blue')
    plt.plot(poly2[:,0], poly2[:,1], marker='o', label='Polygon 2 Edges', alpha=0.7, color='red')

    # label vertices
    if label_vertices:
        for i,(x,y) in enumerate(poly1):
            plt.text(x, y, f"P{i}")
        for i,(x,y) in enumerate(poly2):
            plt.text(x, y, f"P{i}")

    # plot centroid
    if centroid is not None:
        plt.plot(centroid[0], centroid[1], marker='*', color='black', markersize=10)
        if support_lines:
            for p in poly2:
                plt.plot([centroid[0], p[0]], [centroid[1], p[1]], color='gray', linestyle='--', alpha=0.5)

    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.legend()
    plt.show()


def parse_geojson(path):
    with open(path) as f:
        gj = geojson.load(f)
    return gj


def rdp(points, epsilon):
    points = np.array(points)

    def perpendicular_dist(pt, line_start, line_end):
        if np.all(line_start == line_end):
            return np.linalg.norm(pt - line_start)
        return np.abs(np.cross(line_end - line_start, line_start - pt)) / np.linalg.norm(line_end - line_start)

    def recurse(pts):
        if len(pts) < 3:
            return pts
        dmax, idx = 0, 0
        for i in range(1, len(pts)-1):
            d = perpendicular_dist(pts[i], pts[0], pts[-1])
            if d > dmax:
                idx, dmax = i, d

        if dmax > epsilon:
            left = recurse(pts[:idx+1])
            right = recurse(pts[idx:])
            return np.vstack((left[:-1], right))
        else:
            return np.array([pts[0], pts[-1]])

    return recurse(points).tolist()


def rdp_target(points, m):
    # binary search epsilon to get ~m points
    lo, hi = 0, np.ptp(points)
    for _ in range(20):
        mid = (lo + hi) / 2
        simplified = rdp(points, mid)
        if len(simplified) > m:
            lo = mid
        else:
            hi = mid
    return simplified


def polygon_areanp(points):
    
    # ----------------------------------------
    # STEP 1: Extract x and y coordinates
    # ----------------------------------------
    # points = [[x1, y1], [x2, y2], ..., [xn, yn]]
    #
    # We split them into two separate arrays:
    # x = [x1, x2, ..., xn]
    # y = [y1, y2, ..., yn]
    #
    # This makes vectorized operations (dot products) possible
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # ----------------------------------------
    # STEP 2: Create "shifted" versions
    # ----------------------------------------
    # np.roll(y, -1) shifts all elements left:
    #
    # y =        [y1, y2, y3, ..., yn]
    # roll(y,-1)=[y2, y3, ..., yn, y1]
    #
    # This automatically pairs each point i with the "next" point i+1,
    # and wraps around so the last point connects back to the first.
    #
    # This replaces the need for manual indexing like (i+1) % n
    y_next = np.roll(y, -1)
    x_next = np.roll(x, -1)
    
    # ----------------------------------------
    # STEP 3: Compute the two sums (vectorized)
    # ----------------------------------------
    # Shoelace formula (classical form):
    #
    # A = 1/2 * | sum(x_i * y_{i+1}) - sum(y_i * x_{i+1}) |
    #
    # Here:
    # np.dot(x, y_next) = sum(x_i * y_{i+1})
    # np.dot(y, x_next) = sum(y_i * x_{i+1})
    #
    # So we compute both sums using dot products
    sum1 = np.dot(x, y_next)
    sum2 = np.dot(y, x_next)
    
    # ----------------------------------------
    # STEP 4: Final area
    # ----------------------------------------
    # Take the difference and multiply by 1/2
    # abs(...) ensures positive area regardless of orientation
    area = 0.5 * abs(sum1 - sum2)
    
    return area


def PolyAreaRadialQP(points, target_area):
    
    # Convert input list of points into a NumPy array of shape (n, 2)
    # Each row is a vertex [x_i, y_i]
    X = np.array(points)
    
    # -------------------------------
    # STEP 1: Compute centroid
    # -------------------------------
    # The centroid acts as a fixed reference point.
    # All vertices will move ONLY along rays starting from this point.
    #
    # c = (1/n) * sum_i x_i
    #
    # This ensures:
    # - no rotation
    # - no translation
    # - only radial deformation
    centroid = np.mean(X, axis=0)
    
    # -------------------------------
    # STEP 2: Define optimization variables
    # -------------------------------
    # t_i is a scalar per vertex controlling how far it moves
    #
    # Interpretation:
    # t_i = 1   → point stays unchanged
    # t_i > 1   → point moves outward
    # t_i < 1   → point moves inward
    #
    # These are the ONLY decision variables in the problem
    t = cp.Variable(len(points))
    
    # -------------------------------
    # STEP 3: Define new vertex positions (affine mapping)
    # -------------------------------
    # Each vertex is updated via:
    #
    # x_i' = c + t_i * (x_i - c)
    #
    # where:
    # - (x_i - c) is a fixed direction vector
    # - t_i scales that direction
    #
    # IMPORTANT:
    # This expression is AFFINE in t → keeps the problem convex
    #
    # cp.multiply performs elementwise multiplication:
    # each row (x_i - c) is multiplied by t_i
    new_pts = centroid + cp.multiply(t[:, None], (X - centroid))
    
    # -------------------------------
    # STEP 4: Compute original polygon area
    # -------------------------------
    # This is the TRUE polygon area using the shoelace formula.
    # It is constant (not part of optimization).
    #
    # A0 = original area
    A0 = polygon_areanp(points)
    
    # -------------------------------
    # STEP 5: Area control via convex surrogate
    # -------------------------------
    # TRUE area is:
    # A(t) = sum of terms involving t_i * t_{i+1} → NONCONVEX
    #
    # Instead, we use this key approximation:
    #
    # If all t_i = s, then:
    #     area scales as s^2 * A0
    #
    # So to achieve target_area:
    #     s = sqrt(target_area / A0)
    #
    # We enforce:
    #     mean(t_i) = s
    #
    # This distributes scaling across all vertices
    # and approximates the correct area in a convex way.
    s = np.sqrt(target_area / A0)
    
    # -------------------------------
    # STEP 6: Constraints
    # -------------------------------
    constraints = [
        # Enforce global scaling behavior:
        # average radial expansion must equal s
        cp.mean(t) == s,
        
        # Lower bound to prevent collapse or inversion
        # (points collapsing into centroid or flipping)
        t >= 0.1
    ]
    
    # -------------------------------
    # STEP 7: Objective (shape preservation)
    # -------------------------------
    # We minimize deviation from original shape:
    #
    # minimize sum (t_i - 1)^2
    #
    # This keeps all scaling factors close to 1,
    # meaning:
    # - minimal distortion
    # - uniform deformation preferred
    #
    # This is a convex quadratic objective → QP
    obj = cp.Minimize(cp.sum_squares(t - 1))
    
    # -------------------------------
    # STEP 8: Solve convex optimization problem
    # -------------------------------
    # This is a Quadratic Program (QP):
    # - quadratic objective
    # - linear constraints
    #
    # Guaranteed:
    # - global optimum
    # - efficient solve
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    # -------------------------------
    # STEP 9: Extract solution
    # -------------------------------
    # Evaluate the affine expression with optimal t
    pts = new_pts.value
    
    # -------------------------------
    # STEP 10: Compute resulting TRUE area
    # -------------------------------
    # Even though we optimized a surrogate,
    # we evaluate the real polygon area here
    actual_area = polygon_areanp(pts)
    
    return pts, actual_area


def PolyAreaRadialLP(points, target_area, alpha=1):
    print(points)
    
    X = np.array(points)
    centroid = np.mean(X, axis=0)
    n = len(points)
    
    t = cp.Variable(n)
    z = cp.Variable(n)  # abs values
    
    new_pts = centroid + cp.multiply(t[:, None], (X - centroid))
    
    A0 = polygon_areanp(points)
    s = np.sqrt(target_area / A0)
    
    constraints = [
        cp.mean(t) == s,
        t >= 0.1,
        
        # |t - 1| <= z
        t - 1 <= z,
        -(t - 1) <= z
    ]
    
    obj = cp.Minimize(cp.sum(z))
    
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    pts = new_pts.value
    return pts, polygon_areanp(pts)


def nesting_depth(obj):
    depth = 0

    while isinstance(obj, list):
        if len(obj) == 0:
            break
        obj = obj[0]
        depth += 1

    return depth

def preprocess(polygons, target_areas, fixed_points, cartographic_error, target_centers):
    """
    Preprocess input to ensure consistent format for optimization.
    Handles both single polygon and list of polygons, as well as scalar or list target areas.
    """

    # print(fixed_points)

    #cprint(f"Polygons after conversion to numpy arrays: {polygons}")

    depth = nesting_depth(polygons)

    if depth == 2:
        if fixed_points is None:
            fixed_points = [np.mean(np.array(polygons), axis=0)]  # use centroid as fixed point
        polygons = [polygons]  # wrap single polygon in a list

    elif depth == 3:
        # print(f"length of fixed_points: {len(fixed_points)}")
        if fixed_points is None:
            fixed_points = []
            for polygon in polygons:
                fixed_points.append(np.mean(polygon, axis=0))  # use centroid of each polygon
        elif len(fixed_points) == 2:
            fixed_points_temporal = []
            for polygon in polygons:
                fixed_points_temporal.append(fixed_points)
                # print(f"fixed_points is a list of length two: {fixed_points_temporal}")
            fixed_points = fixed_points_temporal
        pass  # already in correct format
    else:
        print("More nested structure")

    # print(type(fixed_points))
    # print(f"fixed_points allalaalalalalal: {fixed_points}")

    if isinstance(fixed_points, type(None)):
        for polygon in polygons:
            fixed_points.append(np.mean(np.array(polygon), axis=0))  # use centroid as fixed point
    elif isinstance(fixed_points, (list)) and len(fixed_points) == 2:
        for polygon in polygons:
            fixed_points = np.array(fixed_points)  # repeat for each polygon
        # print(f"fixed_points is a list of length one: {fixed_points}")
    elif isinstance(fixed_points[0], (list)):
        fixed_points = [np.array(fp) for fp in fixed_points]

    if isinstance(target_areas, type(None)):
        target_areas = []
        for polygon in polygons:
            target_areas.append(polygon_areanp(polygon))  # use original area as target
    elif isinstance(target_areas, (int, float)):
        target_areas = [target_areas] * len(polygons)  # same area for all
    
    polys = []
    for polygon in polygons:
        polygon = np.array(polygon)
        # print(f"Polygon with {len(polygon)} vertices: {polygon}")
        polys.append(polygon)
    polygons = polys

    if target_centers is None:
        target_centers = fixed_points

    return polygons, target_areas, fixed_points, cartographic_error, target_centers


import numpy as np


def make_circle(center, area, n=100):
    """
    Generate an n-vertex polygon approximating a circle
    with a given area around a center point.

    Parameters
    ----------
    center : Circle center.
    area : Desired area.
    n : Number of vertices.
    Returns
    -------
    np.ndarray of shape (n, 2)
    """

    center = np.asarray(center)

    # Circle area = pi r^2
    r = np.sqrt(area / np.pi)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = center[0] + r * np.cos(angles)
    y = center[1] + r * np.sin(angles)

    return np.column_stack((x, y))


import numpy as np


def make_square(center, area, n=100):
    """
    Generate an n-vertex polygon approximating a square
    with a given area around a center point.

    Parameters
    ----------
    center : Square center.
    area : Desired area.
    n : Number of vertices.
    Returns
    -------
    np.ndarray of shape (n, 2)
    """
    
    center = np.asarray(center)

    # Square side length
    side = np.sqrt(area)

    half = side / 2

    # distribute points approximately equally on edges
    n_side = max(1, n // 4)

    pts = []

    # bottom edge
    xs = np.linspace(-half, half, n_side, endpoint=False)
    for x in xs:
        pts.append([x, -half])

    # right edge
    ys = np.linspace(-half, half, n_side, endpoint=False)
    for y in ys:
        pts.append([half, y])

    # top edge
    xs = np.linspace(half, -half, n_side, endpoint=False)
    for x in xs:
        pts.append([x, half])

    # left edge
    ys = np.linspace(half, -half, n_side, endpoint=False)
    for y in ys:
        pts.append([-half, y])

    pts = np.array(pts)

    # if too few points due to rounding
    while len(pts) < n:
        pts = np.vstack([pts, pts[-1]])

    pts = pts[:n]

    # translate to center
    pts += center

    return pts


def polygon_bounds(poly):
    """Return (min_lon, max_lon, min_lat, max_lat) of a single polygon."""
    bounds = {
    'south': min(x[1] for x in poly),
    'east': max(x[0] for x in poly),
    'north': max(x[1] for x in poly),
    'west': min(x[0] for x in poly)
}

    return bounds


def disjoint_pairs_horizontal_and_vertical(polygons):
    """
    Return two lists of index pairs (i, j) with i < j:
      - horizontal_pairs : polygons whose longitude intervals do NOT overlap
      - vertical_pairs   : polygons whose latitude intervals do NOT overlap
    """
    bounds = [polygon_bounds(p) for p in polygons]

    horizontal_pairs = []
    vertical_pairs = []

    for i in range(len(bounds)):
        for j in range(len(bounds)):
            if i == j:
                continue
            if bounds[i]['east'] < bounds[j]['west'] or bounds[j]['east'] < bounds[i]['west']:
                horizontal_pairs.append((i, j))
            if bounds[i]['north'] < bounds[j]['south'] or bounds[j]['north'] < bounds[i]['south']:
                vertical_pairs.append((i, j))

    return horizontal_pairs, vertical_pairs


def neighbouring_pairs(polygons, tolerance=1e-8):
    """
    Return list of (i, j) pairs such that polygon i and polygon j share
    at least one vertex (within tolerance).
    """
    n = len(polygons)
    neighbors = []
    for i in range(n):
        # Convert polygon i to a set of tuples (rounded coordinates) for fast lookup
        set_i = set()
        for pt in polygons[i]:
            set_i.add( (round(pt[0] / tolerance) * tolerance,
                        round(pt[1] / tolerance) * tolerance) )
        for j in range(i + 1, n):
            for pt in polygons[j]:
                key = (round(pt[0] / tolerance) * tolerance,
                       round(pt[1] / tolerance) * tolerance)
                if key in set_i:
                    neighbors.append((i, j))
                    break   # only need one match to mark them as neighbours
    return neighbors


def common_vertices(poly1, poly2, tolerance=1e-8):
    """
    Return a list of points (each as [lon, lat]) that appear in both polygons.
    Points are considered equal if their distance is < tolerance.
    """
    # Build set of rounded tuples for poly1
    set1 = {}
    for pt in poly1:
        key = (round(pt[0] / tolerance) * tolerance,
               round(pt[1] / tolerance) * tolerance)
        # Keep one representative (the exact coordinates from poly1)
        set1.setdefault(key, pt)   # Python 3.9+ has setdefault, else use:
        # if key not in set1: set1[key] = pt
    common = []
    seen = set()
    for pt in poly2:
        key = (round(pt[0] / tolerance) * tolerance,
               round(pt[1] / tolerance) * tolerance)
        if key in set1 and key not in seen:
            common.append(list(set1[key]))  # store the original point from poly1
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


def PolyAreaRadialLP_all(polygons, target_areas, fixed_points = None, shape = "original", cartographic_error=1, shape_preservation=0):
    """
    polygons: either a single polygon P (list of vertices {v1, v2, . . . , vn}) or a list of polygons {P1, P2, . . . , Pk} (list of list of vertices)
    target_areas: desired area for each polygon (can be a single value or a list of values)
    cartographic_error: weight for shape preservation 
    """

    polygons, target_areas, fixed_points, cartographic_error = preprocess(polygons, target_areas, fixed_points, cartographic_error)

    new_polygons = []
    actual_areas = []

    for polygon, area, fixed_point in zip(polygons, target_areas, fixed_points):
        # print(f"Type of fixed_point: {type(fixed_point)}")
        # print(f"fixed_point: {fixed_point}")
        # print(f"Type of polygon: {type(polygon)}")
        # X = np.array(polygon)
        # centroid = np.mean(X, axis=0)
        n = len(polygon)

        constraints = []
        
        t = cp.Variable(n)
        z = cp.Variable(n)  # abs values
        
        new_pts = fixed_point + cp.multiply(t[:, None], (polygon - fixed_point))

        if shape == "circle":
            # Enabling circle-like shapes by adding a term that encourages uniform scaling of the vertices
            r = np.linalg.norm(polygon - fixed_point, axis=1)
            new_radii = cp.multiply(t, r)
            circle_term = cp.sum_squares(new_radii - cp.mean(new_radii))

        elif shape == "square":
            # Enabling square-like shapes by adding a term that encourages vertices to align with the axes
            centered = new_pts - fixed_point
            u = cp.Variable(n)
            v = cp.Variable(n)
            d = cp.Variable(n)

            constraints += [
                u >= centered[:, 0],
                u >= -centered[:, 0],

                v >= centered[:, 1],
                v >= -centered[:, 1],

                d >= u - v,
                d >= v - u
            ]

            square_term = cp.sum_squares(d)
            # abs_x = cp.abs(centered[:, 0])
            # abs_y = cp.abs(centered[:, 1])
            # square_term = cp.sum(cp.abs(abs_x - abs_y))
            # square_term = cp.sum_squares(cp.minimum(abs_x, abs_y))
        
        A0 = polygon_areanp(polygon)
        s = np.sqrt(area / A0)
        
        constraints += [
            cp.mean(t) == s,
            t >= 0.1,
            
            # |t - 1| <= z
            t - 1 <= z,
            -(t - 1) <= z
        ]

        l1_term = cp.sum(z)
        l2_term = cp.sum_squares(t - 1)

        preservation_term = (1 - shape_preservation) * l1_term + shape_preservation * l2_term
        shape_term = 0
        if shape == "circle":
            shape_term = (1 - shape_preservation) * circle_term

        elif shape == "square":
            shape_term = (1 - shape_preservation) * square_term
        
        obj = cp.Minimize(shape_preservation * preservation_term + (1 - shape_preservation) * shape_term)
        
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        pts = new_pts.value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))

    return new_polygons, actual_areas







def PolyAreaRadialLP_all_demers(
    polygons, 
    target_areas, 
    fixed_points = None, 
    shape = "original", 

    target_centers=None,
    horizontal_pairs=None,
    vertical_pairs=None,
    neighboring_pairs=None,
    shared_vertices_of_neighbors=None,
    
    # quality criteria (1 is good, 0 is bad (error is present))
    cartographic_error=1.0, # Region size like statistical value
    shape_deformation=0.0, # Region shape preserved
    relative_direction=1.0, # west is west etc.
    topological_accuracy=1.0, # neigboorhood preservation
    spatial_deformation=1.0, # center over center
    global_shape=1.0, # Only evaluation
    local_shape=1.0, # Only evaluation
    complexity=1.0, # Only evaluation
    data_ink_ratio=1.0, # Only evaluation
    ):
    """
    Parameters
    ----------
    polygons : list
        Single polygon or list of polygons.

    target_areas : float or list
        Desired target area(s).

    fixed_points : list or None
        Radial centers.

    shape : str
        "original", "circle", or "square"

    All quality criteria weights are in [0,1].

    Larger weight => criterion is more important.
    """

    polygons, target_areas, fixed_points, cartographic_error, target_centers = preprocess(
        polygons, 
        target_areas, 
        fixed_points, 
        cartographic_error,
        target_centers
    )

    # ------------------------------------------------------------------
    # normalize weights
    # ------------------------------------------------------------------

    weights = np.array([
        shape_deformation,
        relative_direction,
        topological_accuracy,
        spatial_deformation,
        global_shape,
        local_shape,
        complexity,
        data_ink_ratio,
        cartographic_error
    ], dtype=float)

    weights = weights / (np.sum(weights) + 1e-12)

    (
        W_shape_def,
        W_rel_dir,
        W_topology,
        W_spatial,
        W_global,
        W_local,
        W_complexity,
        W_dataink,
        W_area
    ) = weights

    new_polygons = []
    actual_areas = []

    for polygon, target_area, fixed_point, target_center in zip(
        polygons, 
        target_areas, 
        fixed_points,
        target_centers
    ):
        # print(f"Polygon: {polygon}")
        if shape == "circle" and shape_deformation == 0.0:
            polygon = make_circle(np.mean(np.array(polygon), axis=0), target_area, n=100)
        elif shape == "square" and shape_deformation == 0.0:
            polygon = make_square(np.mean(np.array(polygon), axis=0), target_area, n=100)
        else:
            polygon = np.array(polygon)
        fixed_point = np.array(fixed_point)
        target_center = np.array(target_center)

        center_shift = cp.Variable(2)
        moving_center = fixed_point + center_shift

        n = len(polygon)

        # --------------------------------------------------------------
        # variables
        # --------------------------------------------------------------

        t = cp.Variable(n)
        # L1 auxiliary variables
        z = cp.Variable(n, nonneg=True)  # abs values / Non-negative
        # radial deformation
        directions = polygon - fixed_point
        new_pts = moving_center + cp.multiply(
            t[:, None], 
            directions
        )
        constraints = []
        min_scale = (
            0.01
            + 0.24 * shape_deformation
        )
        
        # --------------------------------------------------------------
        # target area scaling
        # --------------------------------------------------------------

        A0 = polygon_areanp(polygon)
        if A0 <= 1e-12:
            continue
        s = np.sqrt(target_area / A0)
        
        # constraints += [t >= min_scale]

        constraints += [
            cp.mean(t) == s,
            t >= min_scale,
            
            # |t - 1| <= z
            t - 1 <= z,
            -(t - 1) <= z
        ]

        # --------------------------------------------------------------
        # 1. SHAPE DEFORMATION
        # --------------------------------------------------------------

        l1_term = cp.sum(z)
        l2_term = cp.sum_squares(t - 1)

        shape_def_term = (
            (1 - shape_deformation) * l1_term + 
            shape_deformation * l2_term
        )

        # --------------------------------------------------------------
# TARGET SHAPE TERM (circle / square)
        # --------------------------------------------------------------

        shape_term = 0

        radial_norms = np.linalg.norm(
            directions,
            axis=1
        )

        if shape == "circle":
            new_radii = cp.multiply(t, radial_norms)
            shape_term += cp.sum_squares(
                new_radii - cp.mean(new_radii)
            )

        elif shape == "square":
            M = np.maximum(np.abs(directions[:, 0]), np.abs(directions[:, 1]))
            r = cp.Variable(nonneg=True)
            shape_term = cp.sum_squares(cp.multiply(t, M) - r)

        # elif shape == "square":
        #     # Enabling square-like shapes by adding a term that encourages vertices to align with the axes
        #     centered = new_pts - fixed_point
        #     u = cp.Variable(n, nonneg=True)
        #     v = cp.Variable(n, nonneg=True)
        #     d = cp.Variable(n, nonneg=True)
        #     r = cp.Variable(nonneg=True)
        #     m = cp.Variable(n, nonneg=True)

        #     constraints += [
        #         u >= centered[:, 0],
        #         u >= -centered[:, 0],

        #         v >= centered[:, 1],
        #         v >= -centered[:, 1],

        #         d >= u - v,
        #         d >= v - u,

        #         m >= u,
        #         m >= v
        #     ]

        #     shape_term += cp.sum_squares(d)
        #     # shape_term += cp.sum_squares(m-r)
        
        area_error = cp.square(cp.mean(t) - s)   # convex quadratic penalty

        center_term = cp.sum_squares(
            moving_center - target_center
        )

        objective = cp.Minimize(
            shape_deformation * shape_def_term +
            (1 - shape_deformation) * shape_term +
            cartographic_error * area_error +
            spatial_deformation * center_term
        )
        

        # shape_term = 0
        # if shape == "circle":
        #     shape_term = (1 - shape_deformation) * circle_term

        # elif shape == "square":
        #     shape_term = (1 - shape_deformation) * square_term
        
        # obj = cp.Minimize(shape_deformation * preservation_term + (1 - shape_deformation) * shape_term)
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        pts = new_pts.value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))

    return new_polygons, actual_areas, 1, 1





def CartogramFramework_global(
    polygons, 
    target_areas, 
    fixed_points = None, 
    shape = "original", 

    target_centers=None,
    horizontal_pairs=None,
    vertical_pairs=None,
    neighboring_pairs=None,
    shared_vertices_of_neighbors=None,
    
    # quality criteria (1 is good, 0 is bad (error is present))
    cartographic_error=1.0, # Region size like statistical value
    shape_deformation=0.0, # Region shape preserved
    relative_direction=1.0, # west is west etc.
    topological_accuracy=1.0, # neigboorhood preservation
    spatial_deformation=1.0, # center over center
    global_shape=1.0, # Only evaluation
    local_shape=1.0, # Only evaluation
    complexity=1.0, # Only evaluation
    data_ink_ratio=1.0, # Only evaluation
    ):

    (
        polygons,
        target_areas,
        fixed_points,
        cartographic_error,
        target_centers,
        W_shape_def,
        W_rel_dir,
        W_topology,
        W_spatial,
        W_global,
        W_local,
        W_complexity,
        W_dataink,
        W_area
    ) = preprocess_global(
        polygons,
        target_areas,
        fixed_points,
        cartographic_error,
        target_centers,
        shape_deformation,
        relative_direction,
        topological_accuracy,
        spatial_deformation,
        global_shape,
        local_shape,
        complexity,
        data_ink_ratio,
    )   

    constraints = []

    centers = []

    new_polygon_exprs = []

    shape_terms = []
    area_terms = []
    center_terms = []

    relative_direction_term = 0
    topology_term = 0

    polygon_sizes = []

    original_centers = [
        np.mean(np.array(poly), axis=0)
        for poly in polygons
    ]

    for polygon, target_area, fixed_point, target_center in zip(
        polygons,
        target_areas,
        fixed_points,
        target_centers
    ):

        polygon = np.array(polygon)

        # --------------------------------------------------------
        # optional target shape
        # --------------------------------------------------------

        if shape == "circle" and shape_deformation == 0.0:
            polygon = make_circle(
                np.mean(polygon, axis=0),
                target_area,
                n=100
            )

        elif shape == "square" and shape_deformation == 0.0:
            polygon = make_square(
                np.mean(polygon, axis=0),
                target_area,
                n=100
            )

        fixed_point = np.array(fixed_point)

        target_center = np.array(target_center)

        # # --------------------------------------------------------
        # # approximate polygon size
        # # --------------------------------------------------------

        # bounds = polygon_bounds(polygon)

        # width = bounds["east"] - bounds["west"]

        # height = bounds["north"] - bounds["south"]

        # polygon_sizes.append(
        #     max(width, height) / 2
        # )

        # --------------------------------------------------------
        # movable center
        # --------------------------------------------------------

        center_shift = cp.Variable(2)

        moving_center = fixed_point + center_shift

        centers.append(moving_center)

        # --------------------------------------------------------
        # radial deformation variables
        # --------------------------------------------------------

        n = len(polygon)

        t = cp.Variable(n)

        z = cp.Variable(n, nonneg=True)

        directions = polygon - fixed_point

        new_pts = moving_center + cp.multiply(
            t[:, None],
            directions
        )

        new_polygon_exprs.append(new_pts)

        # --------------------------------------------------------
        # target scaling
        # --------------------------------------------------------

        # scale_sq = cp.Variable(nonneg=True)

        A0 = polygon_areanp(polygon)

        if A0 <= 1e-12:
            continue

        s = np.sqrt(target_area / A0)

        min_scale = (
            0.01
            + 0.24 * shape_deformation
        )

        constraints += [

            cp.mean(t) == s,
            # scale_sq >= cp.square(mean_scale),

            t >= min_scale,

            # absolute value linearization
            t - 1 <= z,
            -(t - 1) <= z
        ]

        # --------------------------------------------------------
        # shape deformation term
        # --------------------------------------------------------

        l1_term = cp.sum(z)

        l2_term = cp.sum_squares(t - 1)

        shape_def_term = (
            (1 - shape_deformation) * l1_term +
            shape_deformation * l2_term
        )

        # --------------------------------------------------------
        # target shape term
        # --------------------------------------------------------

        shape_term = 0

        radial_norms = np.linalg.norm(
            directions,
            axis=1
        )

        if shape == "circle":

            new_radii = cp.multiply(
                t,
                radial_norms
            )

            shape_term += cp.sum_squares(
                new_radii - cp.mean(new_radii)
            )

        elif shape == "square":

            M = np.maximum(
                np.abs(directions[:, 0]),
                np.abs(directions[:, 1])
            )

            r = cp.Variable(nonneg=True)

            shape_term += cp.sum_squares(
                cp.multiply(t, M) - r
            )

        # --------------------------------------------------------
        # area error
        # --------------------------------------------------------

        area_error = cp.square(
            cp.mean(t) - s
        )

    
        
        # --------------------------------------------------------
        # center attraction
        # --------------------------------------------------------

        center_term = cp.sum_squares(
            moving_center - target_center
        )

        # --------------------------------------------------------
        # collect objective terms
        # --------------------------------------------------------

        shape_terms.append(
            shape_deformation * shape_def_term +
            (1 - shape_deformation) * shape_term
        )

        area_terms.append(
            cartographic_error * area_error
        )

        center_terms.append(
            spatial_deformation * center_term
        )

    
    objective = cp.Minimize(

        W_shape_def * cp.sum(shape_terms)

        + W_area * cp.sum(area_terms)

        + W_spatial * cp.sum(center_terms)

        # + W_rel_dir * relative_direction_term

        # + W_topology * topology_term
    )

    # ============================================================
    # SOLVE
    # ============================================================

    prob = cp.Problem(
        objective,
        constraints
    )

    prob.solve(
        solver=cp.CLARABEL,
        verbose=False
    )

    # ============================================================
    # EXTRACT RESULTS
    # ============================================================

    new_polygons = []

    actual_areas = []
    
    print(prob.status)

    for poly_expr in new_polygon_exprs:

        pts = poly_expr.value

        new_polygons.append(pts)

        actual_areas.append(
            polygon_areanp(pts)
        )

    return (
        new_polygons,
        actual_areas,
        prob.status,
        prob.value
    )



def PolyAreaRadialLP_all_demers_new(
    polygons,
    target_areas,
    fixed_points=None,
    shape="original",

    target_centers=None,
    horizontal_pairs=None,
    vertical_pairs=None,
    neighboring_pairs=None,
    shared_vertices_of_neighbors=None,

    # weights
    cartographic_error=1.0,
    shape_deformation=0.0,
    relative_direction=1.0,
    topological_accuracy=1.0,
    spatial_deformation=1.0,
    global_shape=1.0,
    local_shape=1.0,
    complexity=1.0,
    data_ink_ratio=1.0,
):
    """
    Hybrid Radial + Demers Cartogram Optimization

    Features:
    ----------
    - Radial polygon deformation
    - Area scaling
    - Shape preservation
    - Relative direction preservation
    - Approximate topology preservation
    - Demers-style polygon repulsion
    - Target-center attraction

    IMPORTANT:
    ----------
    This solves ONE GLOBAL optimization problem.
    """

    # ============================================================
    # preprocess
    # ============================================================

    polygons, target_areas, fixed_points, cartographic_error, target_centers = preprocess(
        polygons,
        target_areas,
        fixed_points,
        cartographic_error,
        target_centers
    )

    n_polys = len(polygons)

    # ============================================================
    # pair generation
    # ============================================================

    if horizontal_pairs is None or vertical_pairs is None:
        horizontal_pairs, vertical_pairs = \
            disjoint_pairs_horizontal_and_vertical(polygons)

    if neighboring_pairs is None:
        neighboring_pairs = neighbouring_pairs(polygons)

    # ============================================================
    # normalize weights
    # ============================================================

    weights = np.array([
        shape_deformation,
        relative_direction,
        topological_accuracy,
        spatial_deformation,
        global_shape,
        local_shape,
        complexity,
        data_ink_ratio,
        cartographic_error
    ], dtype=float)

    weights = weights / (np.sum(weights) + 1e-12)

    (
        W_shape_def,
        W_rel_dir,
        W_topology,
        W_spatial,
        W_global,
        W_local,
        W_complexity,
        W_dataink,
        W_area
    ) = weights

    # ============================================================
    # global containers
    # ============================================================

    constraints = []

    centers = []

    new_polygon_exprs = []

    shape_terms = []
    area_terms = []
    center_terms = []

    relative_direction_term = 0
    topology_term = 0

    polygon_sizes = []

    # ============================================================
    # original centers
    # ============================================================

    original_centers = [
        np.mean(np.array(poly), axis=0)
        for poly in polygons
    ]

    # ============================================================
    # build polygon variables
    # ============================================================

    for polygon, target_area, fixed_point, target_center in zip(
        polygons,
        target_areas,
        fixed_points,
        target_centers
    ):

        polygon = np.array(polygon)

        # --------------------------------------------------------
        # optional target shape
        # --------------------------------------------------------

        if shape == "circle":
            polygon = make_circle(
                np.mean(polygon, axis=0),
                target_area,
                n=100
            )

        elif shape == "square":
            polygon = make_square(
                np.mean(polygon, axis=0),
                target_area,
                n=100
            )

        fixed_point = np.array(fixed_point)

        target_center = np.array(target_center)

        # --------------------------------------------------------
        # approximate polygon size
        # --------------------------------------------------------

        bounds = polygon_bounds(polygon)

        width = bounds["east"] - bounds["west"]

        height = bounds["north"] - bounds["south"]

        polygon_sizes.append(
            max(width, height) / 2
        )

        # --------------------------------------------------------
        # movable center
        # --------------------------------------------------------

        center_shift = cp.Variable(2)

        moving_center = fixed_point + center_shift

        centers.append(moving_center)

        # --------------------------------------------------------
        # radial deformation variables
        # --------------------------------------------------------

        n = len(polygon)

        t = cp.Variable(n)

        z = cp.Variable(n, nonneg=True)

        directions = polygon - fixed_point

        new_pts = moving_center + cp.multiply(
            t[:, None],
            directions
        )

        new_polygon_exprs.append(new_pts)

        # --------------------------------------------------------
        # target scaling
        # --------------------------------------------------------

        A0 = polygon_areanp(polygon)

        if A0 <= 1e-12:
            continue

        s = np.sqrt(target_area / A0)

        min_scale = (
            0.01
            + 0.24 * shape_deformation
        )

        constraints += [

            cp.mean(t) == s,

            t >= min_scale,

            # absolute value linearization
            t - 1 <= z,
            -(t - 1) <= z
        ]

        # --------------------------------------------------------
        # shape deformation term
        # --------------------------------------------------------

        l1_term = cp.sum(z)

        l2_term = cp.sum_squares(t - 1)

        shape_def_term = (
            (1 - shape_deformation) * l1_term +
            shape_deformation * l2_term
        )

        # --------------------------------------------------------
        # target shape term
        # --------------------------------------------------------

        shape_term = 0

        radial_norms = np.linalg.norm(
            directions,
            axis=1
        )

        if shape == "circle":

            new_radii = cp.multiply(
                t,
                radial_norms
            )

            shape_term += cp.sum_squares(
                new_radii - cp.mean(new_radii)
            )

        elif shape == "square":

            M = np.maximum(
                np.abs(directions[:, 0]),
                np.abs(directions[:, 1])
            )

            r = cp.Variable(nonneg=True)

            shape_term += cp.sum_squares(
                cp.multiply(t, M) - r
            )

        # --------------------------------------------------------
        # area error
        # --------------------------------------------------------

        area_error = cp.square(
            cp.mean(t) - s
        )

        # --------------------------------------------------------
        # center attraction
        # --------------------------------------------------------

        center_term = cp.sum_squares(
            moving_center - target_center
        )

        # --------------------------------------------------------
        # collect objective terms
        # --------------------------------------------------------

        shape_terms.append(
            shape_deformation * shape_def_term +
            (1 - shape_deformation) * shape_term
        )

        area_terms.append(
            cartographic_error * area_error
        )

        center_terms.append(
            spatial_deformation * center_term
        )

    # ============================================================
    # DEMERS-STYLE REPULSION
    # ============================================================

    # horizontal separation

    for i, j in horizontal_pairs:

        min_dist = (
            polygon_sizes[i]
            + polygon_sizes[j]
        )

        constraints += [

            centers[j][0] - centers[i][0]
            >= min_dist
        ]

    # vertical separation

    for i, j in vertical_pairs:

        min_dist = (
            polygon_sizes[i]
            + polygon_sizes[j]
        )

        constraints += [

            centers[j][1] - centers[i][1]
            >= min_dist
        ]

    # ============================================================
    # RELATIVE DIRECTION PRESERVATION
    # ============================================================

    for i in range(n_polys):

        for j in range(i + 1, n_polys):

            dx_aux = cp.Variable(nonneg=True)

            dy_aux = cp.Variable(nonneg=True)

            orig_dx = (
                original_centers[i][0]
                - original_centers[j][0]
            )

            orig_dy = (
                original_centers[i][1]
                - original_centers[j][1]
            )

            new_dx = (
                centers[i][0]
                - centers[j][0]
            )

            new_dy = (
                centers[i][1]
                - centers[j][1]
            )

            constraints += [

                dx_aux >= new_dx - orig_dx,
                dx_aux >= -(new_dx - orig_dx),

                dy_aux >= new_dy - orig_dy,
                dy_aux >= -(new_dy - orig_dy),
            ]

            relative_direction_term += (
                dx_aux + dy_aux
            )

    # ============================================================
    # TOPOLOGY PRESERVATION
    # ============================================================

    for i, j in neighboring_pairs:

        topology_term += cp.sum_squares(
            centers[i] - centers[j]
        )

    # ============================================================
    # GLOBAL OBJECTIVE
    # ============================================================

    objective = cp.Minimize(

        W_shape_def * cp.sum(shape_terms)

        + W_area * cp.sum(area_terms)

        + W_spatial * cp.sum(center_terms)

        + W_rel_dir * relative_direction_term

        + W_topology * topology_term
    )

    # ============================================================
    # SOLVE
    # ============================================================

    prob = cp.Problem(
        objective,
        constraints
    )

    prob.solve(
        solver=cp.CLARABEL,
        verbose=False
    )

    # ============================================================
    # EXTRACT RESULTS
    # ============================================================

    new_polygons = []

    actual_areas = []
    
    print(prob.status)

    for poly_expr in new_polygon_exprs:

        pts = poly_expr.value

        new_polygons.append(pts)

        actual_areas.append(
            polygon_areanp(pts)
        )

    return (
        new_polygons,
        actual_areas,
        prob.status,
        prob.value
    )









# def PolyAreaRadialLP_all_demers_new(
#     polygons,
#     target_areas,
#     fixed_points=None,
#     shape="original",

#     target_centers=None,
#     horizontal_pairs=None,
#     vertical_pairs=None,
#     neighboring_pairs=None,
#     shared_vertices_of_neighbors=None,

#     # weights
#     cartographic_error=1.0,
#     shape_deformation=0.0,
#     relative_direction=1.0,
#     topological_accuracy=1.0,
#     spatial_deformation=1.0,

#     global_shape=1.0, # Only evaluation
#     local_shape=1.0, # Only evaluation
#     complexity=1.0, # Only evaluation
#     data_ink_ratio=1.0, # Only evaluation
# ):
#     """
#     Global LP/QP implementation inspired by:

#     Multicriteria Optimization for Dynamic Demers Cartograms

#     Implements:
#     - Eq 2-4   : separation constraints
#     - Eq 7     : relative direction preservation
#     - Eq 8-9   : weak separation constraints
#     - Eq 10-13: absolute value linearization

#     IMPORTANT:
#     This version solves ONE GLOBAL optimization problem.
#     """

#     # ------------------------------------------------------------
#     # preprocess
#     # ------------------------------------------------------------

#     polygons, target_areas, fixed_points, _, target_centers = preprocess(
#         polygons,
#         target_areas,
#         fixed_points,
#         cartographic_error,
#         target_centers
#     )

#     n_polys = len(polygons)

#     if horizontal_pairs is None:
#         horizontal_pairs, vertical_pairs = \
#             disjoint_pairs_horizontal_and_vertical(polygons)

#     if neighboring_pairs is None:
#         neighboring_pairs = neighbouring_pairs(polygons)

#     # ------------------------------------------------------------
#     # normalize weights
#     # ------------------------------------------------------------

#     weights = np.array([
#         shape_deformation,
#         relative_direction,
#         topological_accuracy,
#         spatial_deformation,
#         cartographic_error
#     ], dtype=float)

#     weights = weights / (weights.sum() + 1e-12)

#     (
#         W_shape,
#         W_rel_dir,
#         W_topology,
#         W_spatial,
#         W_area
#     ) = weights

#     # ------------------------------------------------------------
#     # global containers
#     # ------------------------------------------------------------

#     constraints = []

#     centers = []
#     half_sizes = []

#     all_new_pts = []

#     shape_terms = []
#     area_terms = []
#     spatial_terms = []

#     relative_direction_term = 0
#     topology_term = 0

#     # ------------------------------------------------------------
#     # original centers
#     # ------------------------------------------------------------

#     original_centers = [
#         np.mean(np.array(poly), axis=0)
#         for poly in polygons
#     ]

#     # ------------------------------------------------------------
#     # build variables for every polygon
#     # ------------------------------------------------------------

#     for idx in range(n_polys):

#         polygon = np.array(polygons[idx])

#         target_area = target_areas[idx]

#         fixed_point = np.array(fixed_points[idx])

#         target_center = np.array(target_centers[idx])

#         # --------------------------------------------------------
#         # optional shape conversion
#         # --------------------------------------------------------

#         if shape == "circle":
#             polygon = make_circle(
#                 np.mean(polygon, axis=0),
#                 target_area,
#                 n=100
#             )

#         elif shape == "square":
#             polygon = make_square(
#                 np.mean(polygon, axis=0),
#                 target_area,
#                 n=100
#             )

#         # --------------------------------------------------------
#         # center variables
#         # --------------------------------------------------------

#         center_shift = cp.Variable(2)

#         moving_center = fixed_point + center_shift

#         centers.append(moving_center)

#         # --------------------------------------------------------
#         # square size
#         # --------------------------------------------------------

#         side_length = np.sqrt(target_area)

#         half_size = side_length / 2.0

#         half_sizes.append(half_size)

#         # --------------------------------------------------------
#         # radial deformation variables
#         # --------------------------------------------------------

#         n = len(polygon)

#         t = cp.Variable(n)

#         z = cp.Variable(n, nonneg=True)

#         directions = polygon - fixed_point

#         new_pts = moving_center + cp.multiply(
#             t[:, None],
#             directions
#         )

#         all_new_pts.append(new_pts)

#         # --------------------------------------------------------
#         # target scaling
#         # --------------------------------------------------------

#         A0 = polygon_areanp(polygon)

#         if A0 <= 1e-12:
#             continue

#         s = np.sqrt(target_area / A0)

#         min_scale = (
#             0.01 +
#             0.24 * shape_deformation
#         )

#         constraints += [
#             cp.mean(t) == s,
#             t >= min_scale,

#             t - 1 <= z,
#             -(t - 1) <= z
#         ]

#         # --------------------------------------------------------
#         # shape deformation
#         # --------------------------------------------------------

#         l1_term = cp.sum(z)

#         l2_term = cp.sum_squares(t - 1)

#         shape_term = (
#             (1 - shape_deformation) * l1_term +
#             shape_deformation * l2_term
#         )

#         shape_terms.append(shape_term)

#         # --------------------------------------------------------
#         # area error
#         # --------------------------------------------------------

#         area_error = cp.square(cp.mean(t) - s)

#         area_terms.append(area_error)

#         # --------------------------------------------------------
#         # spatial deformation
#         # --------------------------------------------------------

#         center_term = cp.sum_squares(
#             moving_center - target_center
#         )

#         spatial_terms.append(center_term)

#     # ============================================================
#     # Eq. 2-4 / Eq. 8-9
#     # separation constraints
#     # ============================================================

#     # weak horizontal ordering
#     for (i, j) in horizontal_pairs:

#         constraints += [
#             centers[i][0] <= centers[j][0]
#         ]

#     # weak vertical ordering
#     for (i, j) in vertical_pairs:

#         constraints += [
#             centers[i][1] <= centers[j][1]
#         ]

#     # ============================================================
#     # Eq. 7 + Eq. 10-13
#     # relative direction preservation
#     # ============================================================

#     for i in range(n_polys):

#         for j in range(i + 1, n_polys):

#             dx_aux = cp.Variable(nonneg=True)

#             dy_aux = cp.Variable(nonneg=True)

#             original_dx = (
#                 original_centers[i][0]
#                 - original_centers[j][0]
#             )

#             original_dy = (
#                 original_centers[i][1]
#                 - original_centers[j][1]
#             )

#             current_dx = (
#                 centers[i][0]
#                 - centers[j][0]
#             )

#             current_dy = (
#                 centers[i][1]
#                 - centers[j][1]
#             )

#             # absolute value linearization

#             constraints += [

#                 dx_aux >= current_dx - original_dx,
#                 dx_aux >= -(current_dx - original_dx),

#                 dy_aux >= current_dy - original_dy,
#                 dy_aux >= -(current_dy - original_dy),
#             ]

#             relative_direction_term += (
#                 dx_aux + dy_aux
#             )

#     # ============================================================
#     # topology preservation
#     # ============================================================

#     for (i, j) in neighboring_pairs:

#         topology_term += cp.norm1(
#             centers[i] - centers[j]
#         )

#     # ============================================================
#     # total objective
#     # ============================================================

#     objective = cp.Minimize(

#         W_shape * cp.sum(shape_terms)

#         + W_area * cp.sum(area_terms)

#         + W_spatial * cp.sum(spatial_terms)

#         + W_rel_dir * relative_direction_term

#         + W_topology * topology_term
#     )

#     # ============================================================
#     # solve
#     # ============================================================

#     problem = cp.Problem(
#         objective,
#         constraints
#     )

#     problem.solve(
#         solver=cp.CLARABEL
#     )

#     # ============================================================
#     # extract results
#     # ============================================================

#     new_polygons = []

#     actual_areas = []

#     for pts_expr in all_new_pts:

#         pts = pts_expr.value

#         new_polygons.append(pts)

#         actual_areas.append(
#             polygon_areanp(pts)
#         )

#     return (
#         new_polygons,
#         actual_areas,
#         problem.status,
#         problem.value
#     )













# def PolyAreaRadialLP_all_demers_non_overlap(
#     polygons,
#     target_areas,
#     fixed_points=None,
#     shape="original",
#     target_centers=None,
#     # quality criteria weights (0 = ignore, 1 = full importance)
#     cartographic_error=1.0,
#     shape_deformation=0.0,
#     relative_direction=1.0,
#     topological_accuracy=1.0,
#     spatial_deformation=1.0,
#     global_shape=1.0,
#     local_shape=1.0,
#     complexity=1.0,
#     data_ink_ratio=1.0,
# ):
#     """
#     Radial scaling & translation of polygons with non‑overlap enforcement.

#     For each polygon we:
#       - keep its star‑shape from a fixed radial center,
#       - allow scaling per radial direction (t_i) and a global center shift,
#       - match target area and target center,
#       - preserve shape (deformation vs. circle/square),
#       - and when topological_accuracy > 0, add a penalty that pushes
#         polygons apart so that they stop overlapping.

#     Parameters
#     ----------
#     polygons : list of (N,2) arrays
#     target_areas : list of floats
#     fixed_points : list of (2,) points, radial origins
#     shape : 'original', 'circle', 'square'
#     target_centers : list of (2,) desired centers
#     ... : weights for different criteria (used as coefficients)

#     Returns
#     -------
#     new_polygons : list of (N,2) arrays
#     actual_areas : list of floats
#     """
#     # ------------------------------------------------------------------
#     # preprocess inputs (assumed to return lists of equal length)
#     # ------------------------------------------------------------------
#     polygons, target_areas, fixed_points, cartographic_error, target_centers = preprocess(
#         polygons, target_areas, fixed_points, cartographic_error, target_centers
#     )

#     N = len(polygons)
#     pi = np.pi

#     # ------------------------------------------------------------------
#     # CVXPY variables for all polygons
#     # ------------------------------------------------------------------
#     center_shifts = [cp.Variable(2) for _ in range(N)]          # translation
#     t_list = []                                                 # radial scalings
#     z_list = []                                                 # |t-1| auxiliaries
#     r_vars = []                                                 # for square shape

#     for i, poly in enumerate(polygons):
#         n = len(poly)
#         t_list.append(cp.Variable(n))
#         z_list.append(cp.Variable(n, nonneg=True))
#         r_vars.append(cp.Variable(nonneg=True))   # only used when shape=='square'

#     # ------------------------------------------------------------------
#     # constraints (lower bound on scalings)
#     # ------------------------------------------------------------------
#     constraints = []
#     min_scale = 0.01 + 0.24 * shape_deformation
#     for t in t_list:
#         constraints += [t >= min_scale]

#     # absolute value constraints |t-1| <= z
#     for t, z in zip(t_list, z_list):
#         constraints += [t - 1 <= z, -(t - 1) <= z]

#     # ------------------------------------------------------------------
#     # build objective
#     # ------------------------------------------------------------------
#     obj_expr = 0

#     for i, (poly, target_area, fixed_pt, target_center, t, z, r_var) in enumerate(
#         zip(polygons, target_areas, fixed_points, target_centers, t_list, z_list, r_vars)
#     ):
        
#         if shape == "circle" and shape_deformation == 0:
#             poly = make_circle(np.mean(np.array(poly), axis=0), target_area, n=100)
#         elif shape == "square" and shape_deformation == 0:
#             poly = make_square(np.mean(np.array(poly), axis=0), target_area, n=100)
#         else:
#             poly = np.array(poly)

#         fixed_pt = np.array(fixed_pt)
#         target_center = np.array(target_center)
#         directions = poly - fixed_pt          # radial directions from fixed point
#         radial_norms = np.linalg.norm(directions, axis=1)
#         moving_center = fixed_pt + center_shifts[i]

#         # ----- area scaling factor -----
#         A0 = polygon_areanp(poly)            # original area
#         if A0 < 1e-12:
#             continue
#         s = np.sqrt(target_area / A0)

#         # ----- shape deformation term (blend L1 / L2 of t-1) -----
#         l1_term = cp.sum(z)
#         l2_term = cp.sum_squares(t - 1)
#         shape_def_term = (1 - shape_deformation) * l1_term + shape_deformation * l2_term

#         # ----- target shape term (circle / square / original) -----
#         shape_term = 0
#         if shape == "circle":
#             new_radii = cp.multiply(t, radial_norms)
#             shape_term = cp.sum_squares(new_radii - cp.mean(new_radii))
#         elif shape == "square":
#             M = np.maximum(np.abs(directions[:, 0]), np.abs(directions[:, 1]))
#             shape_term = cp.sum_squares(cp.multiply(t, M) - r_var)

#         # ----- area error (soft penalty on mean(t) == s) -----
#         area_error = cp.square(cp.mean(t) - s)

#         # ----- center term -----
#         center_term = cp.sum_squares(moving_center - target_center)

#         # ----- accumulate objective for this polygon -----
#         obj_expr += (
#             shape_deformation * shape_def_term
#             + (1 - shape_deformation) * shape_term
#             + cartographic_error * area_error
#             + spatial_deformation * center_term
#         )

#     # ------------------------------------------------------------------
#     # non‑overlap penalty (topological accuracy)
#     # ------------------------------------------------------------------
#     if topological_accuracy > 0 and N > 1:
#         for i in range(N):
#             for j in range(i + 1, N):
#                 ci = fixed_points[i] + center_shifts[i]
#                 cj = fixed_points[j] + center_shifts[j]
#                 dist = cp.norm(ci - cj, 2)

#                 # minimal distance desired: treat polygons as circles with
#                 # area = target area
#                 r_i = np.sqrt(target_areas[i] / pi)
#                 r_j = np.sqrt(target_areas[j] / pi)
#                 min_dist = r_i + r_j

#                 # squared hinge penalty: max(0, min_dist - dist)^2
#                 overlap_penalty = cp.square(cp.pos(min_dist - dist))
#                 obj_expr += topological_accuracy * overlap_penalty

#     # ------------------------------------------------------------------
#     # solve
#     # ------------------------------------------------------------------
#     objective = cp.Minimize(obj_expr)
#     prob = cp.Problem(objective, constraints)
#     prob.solve()

#     # ------------------------------------------------------------------
#     # extract results
#     # ------------------------------------------------------------------
#     new_polygons = []
#     actual_areas = []
#     for i, (poly, fixed_pt, t) in enumerate(zip(polygons, fixed_points, t_list)):
#         fixed_pt = np.array(fixed_pt)
#         directions = poly - fixed_pt
#         moving_center = fixed_pt + center_shifts[i].value
#         new_pts = moving_center + np.multiply(t.value[:, None], directions)
#         new_polygons.append(new_pts)
#         actual_areas.append(polygon_areanp(new_pts))

#     return new_polygons, actual_areas











# # def PolyAreaRadialLP_all_demers_non_overlap(
# #     polygons,
# #     target_areas,
# #     fixed_points=None,
# #     shape="original",
# #     target_centers=None,

# #     # quality criteria
# #     cartographic_error=1.0,
# #     shape_deformation=0.0,
# #     relative_direction=1.0,
# #     topological_accuracy=1.0,
# #     spatial_deformation=1.0,
# #     global_shape=1.0,
# #     local_shape=1.0,
# #     complexity=1.0,
# #     data_ink_ratio=1.0,
# # ):
# #     """
# #     Global convex optimization version with:
# #     - radial deformation
# #     - movable centers
# #     - circle/square regularization
# #     - soft area preservation
# #     - soft non-overlap constraints

# #     All criteria are in [0,1].
# #     """

# #     polygons, target_areas, fixed_points, cartographic_error, target_centers = preprocess(
# #         polygons,
# #         target_areas,
# #         fixed_points,
# #         cartographic_error,
# #         target_centers
# #     )

# #     # ------------------------------------------------------------
# #     # normalize weights
# #     # ------------------------------------------------------------

# #     weights = np.array([
# #         shape_deformation,
# #         relative_direction,
# #         topological_accuracy,
# #         spatial_deformation,
# #         global_shape,
# #         local_shape,
# #         complexity,
# #         data_ink_ratio,
# #         cartographic_error
# #     ], dtype=float)

# #     weights = weights / (np.sum(weights) + 1e-12)

# #     (
# #         W_shape_def,
# #         W_rel_dir,
# #         W_topology,
# #         W_spatial,
# #         W_global,
# #         W_local,
# #         W_complexity,
# #         W_dataink,
# #         W_area
# #     ) = weights

# #     # ------------------------------------------------------------
# #     # storage
# #     # ------------------------------------------------------------

# #     all_new_pts = []
# #     all_centers = []
# #     all_original_centers = []
# #     all_radii = []

# #     constraints = []

# #     total_shape_def = 0
# #     total_shape_term = 0
# #     total_area_error = 0
# #     total_center_error = 0
# #     total_overlap_error = 0
# #     total_direction_error = 0

# #     # ============================================================
# #     # BUILD VARIABLES FOR ALL POLYGONS
# #     # ============================================================

# #     for polygon, target_area, fixed_point, target_center in zip(
# #         polygons,
# #         target_areas,
# #         fixed_points,
# #         target_centers
# #     ):

# #         # --------------------------------------------------------
# #         # optional shape replacement
# #         # --------------------------------------------------------

# #         if shape == "circle":
# #             polygon = make_circle(
# #                 np.mean(np.array(polygon), axis=0),
# #                 target_area,
# #                 n=100
# #             )

# #         elif shape == "square":
# #             polygon = make_square(
# #                 np.mean(np.array(polygon), axis=0),
# #                 target_area,
# #                 n=100
# #             )

# #         else:
# #             polygon = np.array(polygon)

# #         fixed_point = np.array(fixed_point)
# #         target_center = np.array(target_center)

# #         n = len(polygon)

# #         # --------------------------------------------------------
# #         # variables
# #         # --------------------------------------------------------

# #         t = cp.Variable(n)

# #         z = cp.Variable(n, nonneg=True)

# #         center_shift = cp.Variable(2)

# #         moving_center = fixed_point + center_shift

# #         directions = polygon - fixed_point

# #         new_pts = moving_center + cp.multiply(
# #             t[:, None],
# #             directions
# #         )

# #         # --------------------------------------------------------
# #         # scaling regularization
# #         # --------------------------------------------------------

# #         deformation_rigidity = (
# #             0.5 * shape_deformation +
# #             0.3 * local_shape +
# #             0.2 * complexity
# #         )

# #         min_scale = 0.01 + 0.24 * deformation_rigidity

# #         constraints += [
# #             t >= min_scale
# #         ]

# #         # --------------------------------------------------------
# #         # area preservation
# #         # --------------------------------------------------------

# #         A0 = polygon_areanp(polygon)

# #         if A0 <= 1e-12:
# #             continue

# #         s = np.sqrt(target_area / A0)

# #         area_error = cp.square(cp.mean(t) - s)

# #         # --------------------------------------------------------
# #         # shape deformation term
# #         # --------------------------------------------------------

# #         constraints += [
# #             t - 1 <= z,
# #             -(t - 1) <= z
# #         ]

# #         l1_term = cp.sum(z)

# #         l2_term = cp.sum_squares(t - 1)

# #         shape_def_term = (
# #             (1 - shape_deformation) * l1_term +
# #             shape_deformation * l2_term
# #         )

# #         # --------------------------------------------------------
# #         # target shape term
# #         # --------------------------------------------------------

# #         shape_term = 0

# #         radial_norms = np.linalg.norm(
# #             directions,
# #             axis=1
# #         )

# #         # ------------------ circle ------------------------------

# #         if shape == "circle":

# #             new_radii = cp.multiply(t, radial_norms)

# #             shape_term += cp.sum_squares(
# #                 new_radii - cp.mean(new_radii)
# #             )

# #             approx_radius = np.sqrt(target_area / np.pi)

# #         # ------------------ square ------------------------------

# #         elif shape == "square":

# #             M = np.maximum(
# #                 np.abs(directions[:, 0]),
# #                 np.abs(directions[:, 1])
# #             )

# #             r = cp.Variable(nonneg=True)

# #             shape_term += cp.sum_squares(
# #                 cp.multiply(t, M) - r
# #             )

# #             approx_radius = np.sqrt(target_area) / 2

# #         else:

# #             approx_radius = np.sqrt(target_area / np.pi)

# #         # --------------------------------------------------------
# #         # center objective
# #         # --------------------------------------------------------

# #         center_term = cp.sum_squares(
# #             moving_center - target_center
# #         )

# #         # --------------------------------------------------------
# #         # store everything
# #         # --------------------------------------------------------

# #         all_new_pts.append(new_pts)

# #         all_centers.append(moving_center)

# #         all_original_centers.append(fixed_point)

# #         all_radii.append(approx_radius)

# #         total_shape_def += shape_def_term

# #         total_shape_term += shape_term

# #         total_area_error += area_error

# #         total_center_error += center_term

# #     # ============================================================
# #     # PAIRWISE TERMS
# #     # ============================================================

# #     k = len(all_centers)

# #     for i in range(k):

# #         for j in range(i + 1, k):

# #             center_i = all_centers[i]
# #             center_j = all_centers[j]

# #             original_i = all_original_centers[i]
# #             original_j = all_original_centers[j]

# #             Ri = all_radii[i]
# #             Rj = all_radii[j]

# #             # ----------------------------------------------------
# #             # ORIGINAL DIRECTION
# #             # ----------------------------------------------------

# #             dir_ij = original_j - original_i

# #             norm_dir = np.linalg.norm(dir_ij)

# #             if norm_dir <= 1e-12:
# #                 continue

# #             dir_ij = dir_ij / norm_dir

# #             # ----------------------------------------------------
# #             # NON-OVERLAP
# #             # ----------------------------------------------------

# #             slack = cp.Variable(nonneg=True)

# #             required_sep = Ri + Rj

# #             constraints += [
# #                 dir_ij @ (center_j - center_i)
# #                 >= required_sep - slack
# #             ]

# #             total_overlap_error += cp.square(slack)

# #             # ----------------------------------------------------
# #             # RELATIVE DIRECTION PRESERVATION
# #             # ----------------------------------------------------

# #             original_vec = original_j - original_i

# #             current_vec = center_j - center_i

# #             total_direction_error += cp.sum_squares(
# #                 current_vec - original_vec
# #             )

# #     # ============================================================
# #     # FINAL OBJECTIVE
# #     # ============================================================

# #     objective = cp.Minimize(

# #         # shape deformation
# #         W_shape_def * total_shape_def +

# #         # target shape
# #         (1 - W_shape_def) * total_shape_term +

# #         # area preservation
# #         W_area * total_area_error +

# #         # center movement
# #         W_spatial * total_center_error +

# #         # overlap prevention
# #         W_topology * total_overlap_error +

# #         # relative direction
# #         W_rel_dir * total_direction_error
# #     )

# #     # ============================================================
# #     # SOLVE
# #     # ============================================================

# #     prob = cp.Problem(objective, constraints)

# #     prob.solve(verbose=False)

# #     # ============================================================
# #     # OUTPUT
# #     # ============================================================

# #     new_polygons = []
# #     actual_areas = []

# #     for pts in all_new_pts:

# #         pts_val = pts.value

# #         new_polygons.append(pts_val)

# #         actual_areas.append(
# #             polygon_areanp(pts_val)
# #         )

# #     return new_polygons, actual_areas











# def PolyAreaRadialLP_all_test(
#     polygons,
#     target_areas,
#     fixed_points=None,
#     shape="original",

#     # legacy
#     cartographic_error=1.0,
#     shape_preservation=0.5,

#     # multicriteria weights
#     shape_deformation=1.0,
#     relative_direction=1.0,
#     topological_accuracy=1.0,
#     spatial_deformation=1.0,
#     global_shape=1.0,
#     local_shape=1.0,
#     complexity=1.0,
#     data_ink_ratio=1.0,

#     solver=None,
#     verbose=False
# ):
#     """
#     Multicriteria convex radial polygon deformation LP.

#     Parameters
#     ----------
#     polygons : list
#         Single polygon or list of polygons.

#     target_areas : float or list
#         Desired target area(s).

#     fixed_points : list or None
#         Radial centers.

#     shape : str
#         "original", "circle", or "square"

#     All multicriteria weights are in [0,1].

#     Larger weight => criterion is more important.
#     """

#     polygons, target_areas, fixed_points, cartographic_error = preprocess(
#         polygons,
#         target_areas,
#         fixed_points,
#         cartographic_error
#     )

#     # ------------------------------------------------------------------
#     # normalize weights
#     # ------------------------------------------------------------------

#     weights = np.array([
#         shape_deformation,
#         relative_direction,
#         topological_accuracy,
#         spatial_deformation,
#         global_shape,
#         local_shape,
#         complexity,
#         data_ink_ratio,
#         cartographic_error
#     ], dtype=float)

#     weights = weights / (np.sum(weights) + 1e-12)

#     (
#         W_shape_def,
#         W_rel_dir,
#         W_topology,
#         W_spatial,
#         W_global,
#         W_local,
#         W_complexity,
#         W_dataink,
#         W_area
#     ) = weights

#     new_polygons = []
#     actual_areas = []

#     # ==================================================================
#     # optimize each polygon
#     # ==================================================================

#     for polygon, target_area, fixed_point in zip(
#         polygons,
#         target_areas,
#         fixed_points
#     ):

#         polygon = np.array(polygon)
#         fixed_point = np.array(fixed_point)

#         n = len(polygon)

#         # --------------------------------------------------------------
#         # variables
#         # --------------------------------------------------------------

#         t = cp.Variable(n)

#         # L1 auxiliary variables
#         z = cp.Variable(n, nonneg=True)

#         # radial deformation
#         directions = polygon - fixed_point

#         new_pts = fixed_point + cp.multiply(
#             t[:, None],
#             directions
#         )

#         constraints = []

#         # --------------------------------------------------------------
#         # target area scaling
#         # --------------------------------------------------------------

#         A0 = polygon_areanp(polygon)

#         if A0 <= 1e-12:
#             continue

#         s = np.sqrt(target_area / A0)

#         constraints += [
#             cp.mean(t) == s,
#             t >= 0.05
#         ]

#         # --------------------------------------------------------------
#         # |t - 1| <= z
#         # --------------------------------------------------------------

#         constraints += [
#             t - 1 <= z,
#             -(t - 1) <= z
#         ]

#         # ==============================================================
#         # 1. SHAPE DEFORMATION
#         # ==============================================================

#         l1_term = cp.sum(z)

#         l2_term = cp.sum_squares(t - 1)

#         shape_def_term = (
#             (1 - shape_preservation) * l1_term +
#             shape_preservation * l2_term
#         )

#         # ==============================================================
#         # 2. RELATIVE DIRECTION PRESERVATION
#         #
#         # Radial deformation inherently preserves direction.
#         # We still softly penalize non-uniform angular scaling.
#         # ==============================================================

#         rel_dir_term = cp.sum_squares(t - cp.mean(t))

#         # ==============================================================
#         # 3. TOPOLOGICAL ACCURACY
#         #
#         # Preserve local edge lengths.
#         # DCP-safe version.
#         # ==============================================================

#         topology_terms = []

#         for i in range(n - 1):

#             orig_len = np.linalg.norm(
#                 polygon[i + 1] - polygon[i]
#             )

#             new_edge = new_pts[i + 1] - new_pts[i]

#             edge_len = cp.Variable(nonneg=True)

#             # epigraph of norm
#             constraints += [
#                 cp.norm(new_edge, 2) <= edge_len
#             ]

#             topology_terms.append(
#                 cp.square(edge_len - orig_len)
#             )

#         topology_term = cp.sum(topology_terms)

#         # ==============================================================
#         # 4. SPATIAL DEFORMATION
#         #
#         # Preserve centroid.
#         # ==============================================================

#         orig_centroid = np.mean(polygon, axis=0)

#         new_centroid = cp.sum(new_pts, axis=0) / n

#         spatial_term = cp.sum_squares(
#             new_centroid - orig_centroid
#         )

#         # ==============================================================
#         # 5. GLOBAL SHAPE
#         #
#         # Encourage coherent scaling.
#         # ==============================================================

#         # global_shape_term = cp.sum_squares(
#         #     t - cp.mean(t)
#         # )

#         mean_t = cp.sum(t) / n

#         global_shape_term = cp.sum_squares(
#             t - mean_t
#         )

#         # ==============================================================
#         # 6. LOCAL SHAPE
#         #
#         # Neighboring vertices should deform similarly.
#         # ==============================================================

#         if n >= 2:
#             local_shape_term = cp.sum_squares(
#                 t[1:] - t[:-1]
#             )
#         else:
#             local_shape_term = 0

#         # ==============================================================
#         # 7. COMPLEXITY
#         #
#         # Smooth second derivative.
#         # ==============================================================

#         if n >= 3:
#             complexity_term = cp.sum_squares(
#                 t[2:] - 2 * t[1:-1] + t[:-2]
#             )
#         else:
#             complexity_term = 0

#         # ==============================================================
#         # 8. DATA INK RATIO
#         #
#         # Encourage compact radial distribution.
#         # ==============================================================

#         radial_norms = np.linalg.norm(
#             directions,
#             axis=1
#         )

#         new_radii = cp.multiply(t, radial_norms)

#         data_ink_term = cp.sum_squares(
#             new_radii - cp.mean(new_radii)
#         )

#         # ==============================================================
#         # 9. CARTOGRAPHIC ERROR
#         #
#         # Soft area consistency.
#         # ==============================================================

#         area_error_term = cp.square(
#             cp.mean(t) - s
#         )

#         # ==============================================================
#         # OPTIONAL SHAPE REGULARIZATION
#         # ==============================================================

#         shape_term = 0

#         # --------------------------------------------------------------
#         # circle
#         # --------------------------------------------------------------

#         if shape == "circle":

#             shape_term += cp.sum_squares(
#                 new_radii - cp.mean(new_radii)
#             )

#         # --------------------------------------------------------------
#         # square
#         # --------------------------------------------------------------

#         elif shape == "square":

#             centered = new_pts - fixed_point

#             u = cp.Variable(n, nonneg=True)
#             v = cp.Variable(n, nonneg=True)
#             d = cp.Variable(n, nonneg=True)

#             constraints += [

#                 u >= centered[:, 0],
#                 u >= -centered[:, 0],

#                 v >= centered[:, 1],
#                 v >= -centered[:, 1],

#                 d >= u - v,
#                 d >= v - u
#             ]

#             shape_term += cp.sum_squares(d)

#         # ==============================================================
#         # FINAL MULTICRITERIA OBJECTIVE
#         # ==============================================================

#         objective = cp.Minimize(

#             W_shape_def * shape_def_term +

#             W_rel_dir * rel_dir_term +

#             W_topology * topology_term +

#             W_spatial * spatial_term +

#             W_global * global_shape_term +

#             W_local * local_shape_term +

#             W_complexity * complexity_term +

#             W_dataink * data_ink_term +

#             W_area * area_error_term +

#             0.25 * shape_term
#         )

#         # ==============================================================
#         # solve
#         # ==============================================================

#         prob = cp.Problem(objective, constraints)

#         try:
#             prob.solve(
#                 solver=solver,
#                 verbose=verbose
#             )

#         except Exception as e:
#             print(f"Solver failed: {e}")
#             continue

#         if new_pts.value is None:
#             print("Optimization failed.")
#             continue

#         pts = np.array(new_pts.value)

#         new_polygons.append(pts)

#         actual_areas.append(
#             polygon_areanp(pts)
#         )

#     return new_polygons, actual_areas