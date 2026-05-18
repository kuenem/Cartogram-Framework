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


def preprocess_global(
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
):
    """
    Normalize all inputs into consistent list-based numpy formats.
    """

    polygons = normalize_polygons(polygons)

    n = len(polygons)

    fixed_points = normalize_fixed_points(
        polygons,
        fixed_points
    )

    target_areas = normalize_target_areas(
        polygons,
        target_areas
    )

    target_centers = normalize_target_centers(
        polygons,
        fixed_points,
        target_centers
    )

    validate_inputs(
        polygons,
        target_areas,
        fixed_points,
        target_centers
    )

    (W_shape_def,
    W_rel_dir,
    W_topology,
    W_spatial,
    W_global,
    W_local,
    W_complexity,
    W_dataink,
    W_area) = normalize_weights(
    shape_deformation,
    relative_direction,
    topological_accuracy,
    spatial_deformation,
    global_shape,
    local_shape,
    complexity,
    data_ink_ratio,
    cartographic_error
    )

    return (
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
    )


def normalize_polygons(polygons):
    """
    Convert polygons into:
        List[np.ndarray shape=(n_vertices,2)]
    """

    if polygons is None:
        raise ValueError("polygons cannot be None")

    depth = nesting_depth(polygons)

    # ----------------------------------------
    # single polygon
    # ----------------------------------------

    if depth == 2:

        polygons = [polygons]

    # ----------------------------------------
    # already list of polygons
    # ----------------------------------------

    elif depth != 3:

        raise ValueError(
            f"Unsupported polygon nesting depth: {depth}"
        )

    # ----------------------------------------
    # convert all to numpy arrays
    # ----------------------------------------

    normalized = []

    for poly in polygons:

        arr = np.asarray(poly, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 2:

            raise ValueError(
                "Each polygon must have shape (n_vertices, 2)"
            )

        normalized.append(arr)

    return normalized


def nesting_depth(obj):
    depth = 0

    while isinstance(obj, list):
        if len(obj) == 0:
            break
        obj = obj[0]
        depth += 1

    return depth


def normalize_fixed_points(polygons, fixed_points):
    """
    Normalize fixed_points into:
        List[np.ndarray shape=(2,)]
    """

    n = len(polygons)

    # ----------------------------------------
    # default: polygon centroids
    # ----------------------------------------

    if fixed_points is None:

        return [
            np.mean(poly, axis=0)
            for poly in polygons
        ]

    fixed_points = np.asarray(fixed_points, dtype=float)

    # ----------------------------------------
    # single point -> broadcast
    # ----------------------------------------

    if fixed_points.ndim == 1:

        if fixed_points.shape[0] != 2:

            raise ValueError(
                "Single fixed point must have shape (2,)"
            )

        return [
            fixed_points.copy()
            for _ in range(n)
        ]

    # ----------------------------------------
    # list of points
    # ----------------------------------------

    if fixed_points.ndim == 2:

        if fixed_points.shape != (n, 2):

            raise ValueError(
                f"fixed_points must have shape ({n},2)"
            )

        return [
            fp.copy()
            for fp in fixed_points
        ]

    raise ValueError(
        "Unsupported fixed_points format"
    )


def normalize_target_areas(polygons, target_areas):
    """
    Normalize target_areas into:
        List[float]
    """

    n = len(polygons)

    # ----------------------------------------
    # default: original polygon areas
    # ----------------------------------------

    if target_areas is None:

        return [
            polygon_areanp(poly)
            for poly in polygons
        ]

    # ----------------------------------------
    # scalar -> broadcast
    # ----------------------------------------

    if np.isscalar(target_areas):

        return [
            float(target_areas)
            for _ in range(n)
        ]

    # ----------------------------------------
    # list/array
    # ----------------------------------------

    target_areas = list(target_areas)

    if len(target_areas) != n:

        raise ValueError(
            f"target_areas must have length {n}"
        )

    return [
        float(a)
        for a in target_areas
    ]


def normalize_target_centers(
    polygons,
    fixed_points,
    target_centers
):
    """
    Normalize target_centers into:
        List[np.ndarray shape=(2,)]
    """

    n = len(polygons)

    # ----------------------------------------
    # default: fixed points
    # ----------------------------------------

    if target_centers is None:

        return [
            np.array(fp)
            for fp in fixed_points
        ]

    target_centers = np.asarray(
        target_centers,
        dtype=float
    )

    # ----------------------------------------
    # single center -> broadcast
    # ----------------------------------------

    if target_centers.ndim == 1:

        if target_centers.shape[0] != 2:

            raise ValueError(
                "Single target center must have shape (2,)"
            )

        return [
            target_centers.copy()
            for _ in range(n)
        ]

    # ----------------------------------------
    # list of centers
    # ----------------------------------------

    if target_centers.ndim == 2:

        if target_centers.shape != (n, 2):

            raise ValueError(
                f"target_centers must have shape ({n},2)"
            )

        return [
            tc.copy()
            for tc in target_centers
        ]

    raise ValueError(
        "Unsupported target_centers format"
    )


def normalize_weights(
        shape_deformation, 
        relative_direction,
        topological_accuracy,
        spatial_deformation,
        global_shape,
        local_shape,
        complexity,
        data_ink_ratio,
        cartographic_error):
    """
    Normalize weights into a consistent format.
    """

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

    return (W_shape_def,
        W_rel_dir,
        W_topology,
        W_spatial,
        W_global,
        W_local,
        W_complexity,
        W_dataink,
        W_area)



def validate_inputs(
    polygons,
    target_areas,
    fixed_points,
    target_centers
):
    """
    Validate all normalized inputs.
    """

    n = len(polygons)

    if len(target_areas) != n:
        raise ValueError("target_areas length mismatch")

    if len(fixed_points) != n:
        raise ValueError("fixed_points length mismatch")

    if len(target_centers) != n:
        raise ValueError("target_centers length mismatch")

    for i, poly in enumerate(polygons):

        if len(poly) < 3:

            raise ValueError(
                f"Polygon {i} has fewer than 3 vertices"
            )

        area = polygon_areanp(poly)

        if abs(area) < 1e-12:

            raise ValueError(
                f"Polygon {i} has near-zero area"
            )

    for i, area in enumerate(target_areas):

        if area <= 0:

            raise ValueError(
                f"Target area {i} must be positive"
            )