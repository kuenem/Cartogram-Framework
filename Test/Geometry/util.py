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


def plot_polys(polygons, title="Polygons", plot_points=False, centroids=False, label_vertices=False, legend=True, target_centers=None, names=None):
    
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

        if names is not None and idx < len(names):
            centroid = np.mean(points, axis=0)
            area = polygon_areanp(points)
            plt.text(
                centroid[0],
                centroid[1],
                names[idx],
                ha="center",
                va="center",
                fontsize=area
            )

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


def largest_polygon(geojson):
    """
    Given a GeoJSON Feature or geometry dict (Polygon or MultiPolygon),
    returns a Feature with only the largest polygon.
    Passes through plain Polygons unchanged.
    """
    if geojson.get("type") == "Feature":
        geometry = geojson["geometry"]
        properties = geojson.get("properties", {})
    else:
        geometry = geojson
        properties = {}

    if geometry["type"] == "Polygon":
        return geojson  # nothing to do

    if geometry["type"] != "MultiPolygon":
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry['type']}")

    best = max(geometry["coordinates"], key=lambda poly: polygon_areanp(poly[0]))

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": best},
        "properties": properties,
    }

def get_value(df, region, year):
    # col = df.columns[df.iloc[2].eq(year)][0]
    # row = df[df.iloc[:, 0] == region].index[0]
    # return df.loc[row, col]
    col = df.columns[df.iloc[2].astype(str).eq(year)][0]

    mask = (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.lstrip(".")
        .eq(region)
    )

    value = df.loc[mask, col].iloc[0]
    return value




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


# def disjoint_pairs_horizontal_and_vertical(polygons):
#     """
#     Return two lists of index pairs (i, j) with i < j:
#       - horizontal_pairs : polygons whose longitude intervals do NOT overlap
#       - vertical_pairs   : polygons whose latitude intervals do NOT overlap
#       Both directions tho
#     """
#     bounds = [polygon_bounds(p) for p in polygons]

#     horizontal_pairs = []
#     vertical_pairs = []

#     for i in range(len(bounds)):
#         for j in range(len(bounds)):
#             if i == j:
#                 continue
#             if bounds[i]['east'] < bounds[j]['west'] or bounds[j]['east'] < bounds[i]['west']:
#                 horizontal_pairs.append((i, j))
#             if bounds[i]['north'] < bounds[j]['south'] or bounds[j]['north'] < bounds[i]['south']:
#                 vertical_pairs.append((i, j))

#     return horizontal_pairs, vertical_pairs

def disjoint_pairs_horizontal_and_vertical(polygons):
    """Only one direction so either A-B or B-A, not both"""
    bounds = [polygon_bounds(p) for p in polygons]
    horizontal_pairs = []
    vertical_pairs = []
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):  # ← len(bounds), not n
            b_i, b_j = bounds[i], bounds[j]

            if b_i['east'] < b_j['west']:
                horizontal_pairs.append((i, j))
            elif b_j['east'] < b_i['west']:
                horizontal_pairs.append((j, i))

            if b_i['north'] < b_j['south']:
                vertical_pairs.append((i, j))
            elif b_j['north'] < b_i['south']:
                vertical_pairs.append((j, i))

    return horizontal_pairs, vertical_pairs

def disjoint_pairs_horizontal_and_vertical_center(polygons):
    # Use centroids, not bounding boxes
    centroids = [np.mean(np.asarray(p), axis=0) for p in polygons]
    horizontal_pairs = []
    vertical_pairs = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = centroids[j][0] - centroids[i][0]  # positive = j is right of i
            dy = centroids[j][1] - centroids[i][1]  # positive = j is above i

            if abs(dx) >= abs(dy):
                # Primarily horizontal separation
                if dx >= 0:
                    horizontal_pairs.append((i, j))  # i left of j
                else:
                    horizontal_pairs.append((j, i))  # j left of i
            else:
                # Primarily vertical separation
                if dy >= 0:
                    vertical_pairs.append((i, j))    # i below j
                else:
                    vertical_pairs.append((j, i))    # j below i

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

    pairwise_distance_terms = []
    pairwise_h = {}
    pairwise_v = {}

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

    if topological_accuracy == 0.0:
        pass
    else:
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                w_i = np.sqrt(target_areas[i])
                w_j = np.sqrt(target_areas[j])

                w = (w_i + w_j) / 2

                x_i = centers[i][0]
                y_i = centers[i][1]

                x_j = centers[j][0]
                y_j = centers[j][1]

                h = cp.Variable(nonneg=True, name=f"h_{i}_{j}")
                v = cp.Variable(nonneg=True, name=f"v_{i}_{j}")

                dx = x_i - x_j
                dy = y_i - y_j

                constraints += [
                    h >= dx,
                    h >= -dx,

                    v >= dy,
                    v >= -dy,
                ]

                # constraints += [

                #     x_j - x_i >= w,
                #     y_j - y_i >= w,

                # ]

                constraints += [
                    h >= cp.maximum(x_i - x_j, x_j - x_i) - w,
                    v >= cp.maximum(y_i - y_j, y_j - y_i) - w
                ]


                pairwise_h[(i, j)] = h
                pairwise_v[(i, j)] = v
                
                pairwise_distance_terms.append(h + v)
        for i, j in horizontal_pairs:
            constraints += [
                centers[j][0] - centers[i][0] >= w
            ]
        for i, j in vertical_pairs:
            constraints += [
                centers[j][1] - centers[i][1] >= w
            ]

    distance_term = cp.sum(pairwise_distance_terms)
    
    objective = cp.Minimize(

        W_shape_def * cp.sum(shape_terms)

        + W_area * cp.sum(area_terms)

        + W_spatial * cp.sum(center_terms)

        # + W_rel_dir * relative_direction_term

        + W_topology * distance_term
    )

    # ------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------

    prob = cp.Problem(
        objective,
        constraints
    )

    prob.solve(
        solver=cp.CLARABEL,
        verbose=False
    )

    # get RESULTS

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



def CartogramFramework_global_CLAUDE(
    polygons, 
    target_areas, 
    fixed_points=None, 
    shape="original", 

    target_centers=None,
    horizontal_pairs=None,
    vertical_pairs=None,
    neighboring_pairs=None,
    shared_vertices_of_neighbors=None,

    # slope parameter for diagonal distance d_ij
    alpha=1.0,
    # small constant for non-adjacent gap
    epsilon=1e-2,
    # small constant b for b_ij = b * a_ij
    b=1e-2,

    # quality criteria (1 is good, 0 is bad)
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

    # ----------------------------------------------------------
    # Build adjacency lookup from neighboring_pairs
    # ----------------------------------------------------------

    adjacent_set = set()
    if neighboring_pairs is not None:
        for i, j in neighboring_pairs:
            adjacent_set.add((min(i, j), max(i, j)))

    def is_adjacent(i, j):
        return (min(i, j), max(i, j)) in adjacent_set

    # ----------------------------------------------------------
    # gap_ij  (eq. 12):  0 if adjacent, ε if not
    # b_ij    (table):   b * a_ij  where a_ij = 1 if adjacent, 0.1 if not
    # ----------------------------------------------------------

    def gap(i, j):
        return 0.0 if is_adjacent(i, j) else epsilon

    def a_ij(i, j):
        return 1.0 if is_adjacent(i, j) else 0.1

    def b_ij(i, j):
        return b * a_ij(i, j)

    constraints = []
    centers = []
    new_polygon_exprs = []

    shape_terms = []
    area_terms = []
    center_terms = []

    pairwise_distance_terms = []
    pairwise_h = {}
    pairwise_v = {}

    original_centers = [
        np.mean(np.array(poly), axis=0)
        for poly in polygons
    ]

    for polygon, target_area, fixed_point, target_center in zip(
        polygons, target_areas, fixed_points, target_centers
    ):
        polygon = np.array(polygon)

        if shape == "circle" and shape_deformation == 0.0:
            polygon = make_circle(np.mean(polygon, axis=0), target_area, n=100)
        elif shape == "square" and shape_deformation == 0.0:
            polygon = make_square(np.mean(polygon, axis=0), target_area, n=100)

        fixed_point  = np.array(fixed_point)
        target_center = np.array(target_center)

        # --------------------------------------------------------
        # Movable center  (eq. 3):  m_i = c_i + Δ_i
        # --------------------------------------------------------

        center_shift = cp.Variable(2)
        moving_center = fixed_point + center_shift
        centers.append(moving_center)

        # --------------------------------------------------------
        # Radial deformation variables
        # --------------------------------------------------------

        n = len(polygon)
        t = cp.Variable(n)
        z = cp.Variable(n, nonneg=True)

        directions = polygon - fixed_point

        new_pts = moving_center + cp.multiply(t[:, None], directions)
        new_polygon_exprs.append(new_pts)

        # --------------------------------------------------------
        # Target scaling  (eq. 1):  s_i = sqrt(Â_i / A_i)
        # --------------------------------------------------------

        A0 = polygon_areanp(polygon)
        if A0 <= 1e-12:
            continue

        s = np.sqrt(target_area / A0)

        min_scale = 0.01 + 0.24 * shape_deformation

        constraints += [
            cp.mean(t) == s,          # eq. 4
            t >= min_scale,           # eq. 5
            t - 1 <= z,               # eq. 6
            -(t - 1) <= z,            # eq. 7
        ]

        # --------------------------------------------------------
        # Shape deformation term
        # --------------------------------------------------------

        l1_term = cp.sum(z)
        l2_term = cp.sum_squares(t - 1)

        shape_def_term = (
            (1 - shape_deformation) * l1_term +
            shape_deformation * l2_term
        )

        # --------------------------------------------------------
        # Target shape term (circle / square)
        # --------------------------------------------------------

        shape_term = 0
        radial_norms = np.linalg.norm(directions, axis=1)

        if shape == "circle":                              # eq. 20
            new_radii = cp.multiply(t, radial_norms)
            shape_term += cp.sum_squares(new_radii - cp.mean(new_radii))

        elif shape == "square":                            # eq. 18–19
            M = np.maximum(
                np.abs(directions[:, 0]),
                np.abs(directions[:, 1])
            )
            r = cp.Variable(nonneg=True)
            shape_term += cp.sum_squares(cp.multiply(t, M) - r)

        # --------------------------------------------------------
        # Area error
        # --------------------------------------------------------

        area_error = cp.square(cp.mean(t) - s)

        # --------------------------------------------------------
        # Center attraction  (spatial deformation)
        # --------------------------------------------------------

        center_term = cp.sum_squares(moving_center - target_center)

        # --------------------------------------------------------
        # Collect per-region objective terms
        # --------------------------------------------------------

        shape_terms.append(
            shape_deformation * shape_def_term +
            (1 - shape_deformation) * shape_term
        )
        area_terms.append(cartographic_error * area_error)
        center_terms.append(spatial_deformation * center_term)

    # ----------------------------------------------------------
    # Pairwise topology terms  (eqs. 8–16)
    # ----------------------------------------------------------

    if topological_accuracy != 0.0:

        n_regions = len(polygons)

        for i in range(n_regions):
            for j in range(i + 1, n_regions):

                w_i = np.sqrt(target_areas[i])
                w_j = np.sqrt(target_areas[j])
                w   = (w_i + w_j) / 2          # eq. 8  w_ij

                g   = gap(i, j)                 # eq. 12 gap_ij

                x_i = centers[i][0]
                y_i = centers[i][1]
                x_j = centers[j][0]
                y_j = centers[j][1]

                h = cp.Variable(nonneg=True, name=f"h_{i}_{j}")
                v = cp.Variable(nonneg=True, name=f"v_{i}_{j}")

                # ------------------------------------------------
                # Diagonal distance d_ij (table):
                #   d_ij = |(y_i + α(x_i - x_j)) - y_j|
                # Linearised with an auxiliary slack variable.
                # ------------------------------------------------

                d_var  = cp.Variable(nonneg=True, name=f"d_{i}_{j}")
                diag   = y_i + alpha * (x_i - x_j) - y_j   # affine expression

                constraints += [
                    d_var >=  diag,
                    d_var >= -diag,
                ]

                # b_ij scales the diagonal penalty
                bij = b_ij(i, j)

                # ------------------------------------------------
                # H / V ordering constraints with gap  (eqs. 13–16)
                # ------------------------------------------------

                if horizontal_pairs is not None:
                    if (i, j) in horizontal_pairs:          # eq. 13
                        constraints += [x_j - x_i >= w + g]

                if vertical_pairs is not None:
                    if (i, j) in vertical_pairs:            # eq. 14
                        constraints += [y_j - y_i >= w + g]

                # Pairwise separation for ALL pairs  (eqs. 15–16)
                constraints += [
                    h >= cp.maximum(x_i - x_j, x_j - x_i) - w + g,
                    v >= cp.maximum(y_i - y_j, y_j - y_i) - w + g,
                ]

                pairwise_h[(i, j)] = h
                pairwise_v[(i, j)] = v

                # diagonal term weighted by b_ij
                pairwise_distance_terms.append(h + v + bij * d_var)

    distance_term = cp.sum(pairwise_distance_terms) if pairwise_distance_terms else 0

    # ----------------------------------------------------------
    # Objective  (eq. 21)
    # ----------------------------------------------------------

    objective = cp.Minimize(
        W_shape_def * cp.sum(shape_terms)
        + W_area     * cp.sum(area_terms)
        + W_spatial  * cp.sum(center_terms)
        + W_topology * distance_term
    )

    # ----------------------------------------------------------
    # Solve
    # ----------------------------------------------------------

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    new_polygons  = []
    actual_areas  = []

    print(prob.status)

    for poly_expr in new_polygon_exprs:
        pts = poly_expr.value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))

    return (new_polygons, actual_areas, prob.status, prob.value)






def Cartogram_Framework_Template(
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

    pairwise_distance_terms = []
    pairwise_h = {}
    pairwise_v = {}

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

    print(f"polygons type: {type(polygons)}")
    print(f"target_areas type: {type(target_areas)}")
    print(f"fixed_points type: {type(fixed_points)}")
    print(f"shape: {shape}")
    print(f"target_centers type: {type(target_centers)}")
    print(f"horizontal_pairs type: {type(horizontal_pairs)}")
    print(f"vertical_pairs type: {type(vertical_pairs)}")
    print(f"neighboring_pairs type: {type(neighboring_pairs)}")
    print(f"shared_vertices_of_neighbors type: {type(shared_vertices_of_neighbors)}")







def Cartogram_Framework_Template_ALPHA(
    polygons,
    target_areas,
    fixed_points=None,
    shape="original",
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
    ) = preprocess(
        polygons,
        target_areas,
        fixed_points,
        cartographic_error,
        target_centers,
    )

    n_regions = len(polygons)

    if horizontal_pairs is None:
        horizontal_pairs = []

    if vertical_pairs is None:
        vertical_pairs = []

    if neighboring_pairs is None:
        neighboring_pairs = []

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------

    gamma = shape_deformation

    lambda_a = cartographic_error
    lambda_c = spatial_deformation
    lambda_t = topological_accuracy

    eps_gap = 0.1
    t_min = 0.05

    constraints = []
    objective_terms = []

    moving_centers = []
    t_vars = []
    new_vertices_expr = []

    original_areas = []

    # ---------------------------------------------------------
    # Region variables
    # ---------------------------------------------------------

    for i, polygon in enumerate(polygons):

        polygon = np.asarray(polygon)

        n_i = len(polygon)

        c_i = np.mean(polygon, axis=0)

        original_areas.append(
            polygon_areanp(polygon)
        )

        A_i = original_areas[-1]
        Ahat_i = target_areas[i]

        s_i = np.sqrt(Ahat_i / A_i)

        # moving center
        m_i = cp.Variable(2)

        # radial scaling
        t_i = cp.Variable(n_i)

        # abs deformation
        z_i = cp.Variable(n_i, nonneg=True)

        moving_centers.append(m_i)
        t_vars.append(t_i)

        constraints += [
            t_i >= t_min
        ]

        constraints += [
            t_i - 1 <= z_i,
            -(t_i - 1) <= z_i,
        ]

        # -----------------------------------------------------
        # New vertices
        # -----------------------------------------------------

        verts = []

        for k in range(n_i):

            v = polygon[k]

            expr = (
                m_i
                + t_i[k] * (v - c_i)
            )

            verts.append(expr)

        new_vertices_expr.append(verts)

        # -----------------------------------------------------
        # Shape objective
        # -----------------------------------------------------

        shape_term = 0

        if shape == "circle":

            radii = np.linalg.norm(
                polygon - c_i,
                axis=1
            )

            mean_radius = (
                cp.sum(
                    cp.multiply(
                        t_i,
                        radii
                    )
                ) / n_i
            )

            shape_term = cp.sum_squares(
                cp.multiply(t_i, radii)
                - mean_radius
            )

        elif shape == "square":

            M = np.maximum(
                np.abs(
                    polygon[:, 0] - c_i[0]
                ),
                np.abs(
                    polygon[:, 1] - c_i[1]
                ),
            )

            r_i = cp.Variable(nonneg=True)

            shape_term = cp.sum_squares(
                cp.multiply(t_i, M)
                - r_i
            )

        # -----------------------------------------------------
        # Mean-scale area term
        # -----------------------------------------------------

        area_term = cp.square(
            cp.sum(t_i) / n_i
            - s_i
        )

        # -----------------------------------------------------
        # Center term
        # -----------------------------------------------------

        center_term = cp.sum_squares(
            m_i - target_centers[i]
        )

        # -----------------------------------------------------
        # Shape regularizer
        # -----------------------------------------------------

        deform_term = (
            gamma * (
                (1 - gamma) * cp.sum(z_i)
                + gamma * cp.sum_squares(t_i - 1)
            )
            + (1 - gamma) * shape_term
        )

        objective_terms.append(
            deform_term
            + lambda_a * area_term
            + lambda_c * center_term
        )

    # ---------------------------------------------------------
    # Pairwise topology constraints
    # ---------------------------------------------------------

    x = [m[0] for m in moving_centers]
    y = [m[1] for m in moving_centers]

    neighbor_set = {
        tuple(sorted(p))
        for p in neighboring_pairs
    }

    topology_penalty = 0

    # horizontal ordering

    for i, j in horizontal_pairs:

        w_ij = (
            np.sqrt(target_areas[i])
            + np.sqrt(target_areas[j])
        ) / 2

        gap = (
            0
            if tuple(sorted((i, j))) in neighbor_set
            else eps_gap
        )

        constraints += [
            x[j] - x[i]
            >= w_ij + gap
        ]

    # vertical ordering

    for i, j in vertical_pairs:

        w_ij = (
            np.sqrt(target_areas[i])
            + np.sqrt(target_areas[j])
        ) / 2

        gap = (
            0
            if tuple(sorted((i, j))) in neighbor_set
            else eps_gap
        )

        constraints += [
            y[j] - y[i]
            >= w_ij + gap
        ]

    # adjacency penalties

    for i, j in neighboring_pairs:

        hor = cp.Variable(nonneg=True)
        ver = cp.Variable(nonneg=True)

        w_ij = (
            np.sqrt(target_areas[i])
            + np.sqrt(target_areas[j])
        ) / 2

        constraints += [
            hor >= cp.abs(x[i] - x[j]) - w_ij,
            ver >= cp.abs(y[i] - y[j]) - w_ij,
        ]

        topology_penalty += hor + ver

    objective_terms.append(
        lambda_t * topology_penalty
    )

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    objective = cp.Minimize(
        cp.sum(objective_terms)
    )

    prob = cp.Problem(
        objective,
        constraints
    )

    prob.solve(
        solver=cp.OSQP,
        verbose=False
    )

    # ---------------------------------------------------------
    # Construct polygons
    # ---------------------------------------------------------

    new_polygons = []

    for region_expr in new_vertices_expr:

        poly = np.array(
            [v.value for v in region_expr]
        )

        new_polygons.append(poly)

    # ---------------------------------------------------------
    # Actual areas
    # ---------------------------------------------------------

    actual_areas = [
        polygon_area(poly)
        for poly in new_polygons
    ]

    return (
        new_polygons,
        actual_areas,
        prob.status,
        prob.value,
    )

