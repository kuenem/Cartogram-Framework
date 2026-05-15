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
# from utils.area import *
# from utils.reducingPolygon import *


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



def PolyAreaRadialLP_Circle(points, target_area, alpha=1.0):
    
    X = np.array(points)
    centroid = np.mean(X, axis=0)
    n = len(points)
    
    # Variables
    t = cp.Variable(n)
    z = cp.Variable(n)   # |t - 1|
    
    # Radial transform
    new_pts = centroid + cp.multiply(t[:, None], (X - centroid))
    
    # Area scaling
    A0 = polygon_areanp(points)
    s = np.sqrt(target_area / A0)
    
    # Distances squared (convex)
    dist2 = cp.sum_squares(new_pts - centroid, axis=1)
    
    # Constraints
    constraints = [
        cp.mean(t) == s,
        t >= 0.1,
        
        # |t - 1| <= z
        t - 1 <= z,
        -(t - 1) <= z
    ]
    
    # Objective
    obj = cp.Minimize(
        cp.sum(z) + alpha * cp.sum(dist2)
    )
    
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    pts = new_pts.value
    return pts, polygon_areanp(pts)


def PolyAreaRadialQP_batch(points_list, target_areas):
    
    new_polygons = []
    actual_areas = []
    
    for points, target_area in zip(points_list, target_areas):
        X = np.array(points)
        centroid = np.mean(X, axis=0)
        
        t = cp.Variable(len(points))
        
        new_pts_expr = centroid + cp.multiply(t[:, None], (X - centroid))
        
        A0 = polygon_areanp(points)
        s = np.sqrt(target_area / A0)
        
        constraints = [
            cp.mean(t) == s,
            t >= 0.1
        ]
        
        obj = cp.Minimize(cp.sum_squares(t - 1))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        pts = new_pts_expr.value
        area = polygon_areanp(pts)
        
        new_polygons.append(pts)
        actual_areas.append(area)
    
    return new_polygons, actual_areas


def PolyAreaRadialQP_batch_nooverlap(points_list, target_areas, target_positions, w_pos=0.1, dmin=0.5, verbose=False):
    
    K = len(points_list)
    
    t_vars = []
    c_vars = []
    new_pts_exprs = []
    constraints = []
    obj_terms = []
    
    original_centroids = []
    
    # -------------------------------
    # Build variables per polygon
    # -------------------------------
    for k in range(K):
        X = np.array(points_list[k])
        c0 = np.mean(X, axis=0)
        original_centroids.append(c0)
        
        t = cp.Variable(len(X))
        c_new = cp.Variable(2)
        
        new_pts = c_new + cp.multiply(t[:, None], (X - c0))
        
        A0 = polygon_areanp(points_list[k])
        s = np.sqrt(target_areas[k] / A0)
        
        constraints += [
            cp.mean(t) == s,
            t >= 0.1
        ]
        
        # shape preservation
        obj_terms.append(cp.sum_squares(t - 1))
        
        # position objective
        obj_terms.append(w_pos * cp.sum_squares(c_new - target_positions[k]))
        
        t_vars.append(t)
        c_vars.append(c_new)
        new_pts_exprs.append(new_pts)
    
    # -------------------------------
    # Non-overlap constraints (approx)
    # -------------------------------
    for i in range(K):
        for j in range(i+1, K):
            if verbose:
                print(f"Adding non-overlap constraint between polygon {i} and {j}") 
            ci0 = original_centroids[i]
            cj0 = original_centroids[j]
            if verbose:
                print(f"Original centroids: {ci0} vs {cj0}")
            # fixed separating direction
            direction = ci0 - cj0
            if verbose:
                print(f"Initial direction: {direction}")
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0])
            direction = direction / np.linalg.norm(direction)
            
            # minimal separation (heuristic)            
            constraints.append(direction @ (c_vars[i] - c_vars[j]) >= dmin)
    
    # -------------------------------
    # Solve
    # -------------------------------
    obj = cp.Minimize(cp.sum(obj_terms))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    # -------------------------------
    # Extract results
    # -------------------------------
    new_polygons = []
    actual_areas = []
    
    for k in range(K):
        pts = new_pts_exprs[k].value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))
    
    return new_polygons, actual_areas


def PolyAreaRadialQP_batch_SAT_MICP(points_list, target_areas, target_positions, w_pos=0.1):
    
    K = len(points_list)
    
    t_vars, c_vars, new_pts_exprs = [], [], []
    constraints, obj_terms = [], []
    original_centroids = []
    axes_per_pair = {}
    
    # -------------------------------
    # helper: separating axes
    # -------------------------------
    def separating_axes(poly):
        
        poly = np.array(poly)  # ✅ FIX: ensure vector operations work
        
        axes = []
        n = len(poly)
        
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            
            edge = p2 - p1  # now valid
            
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            
            if norm > 1e-9:
                axes.append(normal / norm)
        
        return axes 
    
    # -------------------------------
    # build per polygon
    # -------------------------------
    for k in range(K):
        X = np.array(points_list[k])
        c0 = np.mean(X, axis=0)
        original_centroids.append(c0)
        
        t = cp.Variable(len(X))
        c_new = cp.Variable(2)
        
        new_pts = c_new + cp.multiply(t[:, None], (X - c0))
        
        A0 = polygon_areanp(points_list[k])
        s = np.sqrt(target_areas[k] / A0)
        
        constraints += [
            cp.mean(t) == s,
            t >= 0.1
        ]
        
        obj_terms.append(cp.sum_squares(t - 1))
        obj_terms.append(w_pos * cp.sum_squares(c_new - target_positions[k]))
        
        t_vars.append(t)
        c_vars.append(c_new)
        new_pts_exprs.append(new_pts)
    
    # -------------------------------
    # SAT constraints with binaries
    # -------------------------------
    M = 1000
    
    for i in range(K):
        for j in range(i+1, K):
            
            axes = separating_axes(points_list[i]) + separating_axes(points_list[j])
            
            z = cp.Variable(len(axes), boolean=True)
            constraints.append(cp.sum(z) >= 1)
            
            for k, axis in enumerate(axes):
                
                axis = axis.reshape(2,)
                
                # project original polygons
                Xi = np.array(points_list[i])
                Xj = np.array(points_list[j])
                
                min_i = np.min(Xi @ axis)
                max_i = np.max(Xi @ axis)
                min_j = np.min(Xj @ axis)
                max_j = np.max(Xj @ axis)
                
                ci = c_vars[i]
                cj = c_vars[j]
                
                dot_i = axis @ ci
                dot_j = axis @ cj
                
                w = cp.Variable(boolean=True)
                
                # SAT constraints (big-M)
                constraints += [
                    max_i + dot_i <= min_j + dot_j + M*(1 - z[k]) + M*(1 - w),
                    max_j + dot_j <= min_i + dot_i + M*(1 - z[k]) + M*w
                ]
    
    # -------------------------------
    # solve
    # -------------------------------
    obj = cp.Minimize(cp.sum(obj_terms))
    prob = cp.Problem(obj, constraints)
    
    prob.solve(solver=cp.SCIP, verbose=True)  # or GUROBI / CPLEX / SCIP
    
    # -------------------------------
    # extract
    # -------------------------------
    new_polygons = []
    actual_areas = []
    
    for k in range(K):
        pts = new_pts_exprs[k].value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))
    
    return new_polygons, actual_areas


def PolyAreaRadialQP_batch_nooverlap_variante(points_list, target_areas, target_positions, w_pos=0.1, dmin=0.5, verbose=False):
    
    K = len(points_list)

    # compute shape matrices
    shape_mats = []
    for pts in points_list:
        X = np.array(pts)
        c0 = np.mean(X, axis=0)
        centered = X - c0
        
        # covariance-like matrix
        D = centered.T @ centered / len(X)
        
        # use sqrt of matrix (Cholesky for stability)
        try:
            D = np.linalg.cholesky(D)
        except:
            D = np.eye(2)
        
        shape_mats.append(D)
    
    t_vars = []
    c_vars = []
    new_pts_exprs = []
    constraints = []
    obj_terms = []
    
    original_centroids = []
    
    # -------------------------------
    # Build variables per polygon
    # -------------------------------
    for k in range(K):
        X = np.array(points_list[k])
        c0 = np.mean(X, axis=0)
        original_centroids.append(c0)
        
        t = cp.Variable(len(X))
        c_new = cp.Variable(2)
        
        new_pts = c_new + cp.multiply(t[:, None], (X - c0))
        
        A0 = polygon_areanp(points_list[k])
        s = np.sqrt(target_areas[k] / A0)
        
        constraints += [
            cp.mean(t) == s,
            t >= 0.1
        ]
        
        # shape preservation
        obj_terms.append(cp.sum_squares(t - 1))
        
        # position objective
        obj_terms.append(w_pos * cp.sum_squares(c_new - target_positions[k]))
        
        t_vars.append(t)
        c_vars.append(c_new)
        new_pts_exprs.append(new_pts)

    alpha = 1.6

    for i in range(K):
        for j in range(i+1, K):
            ci0 = original_centroids[i]
            cj0 = original_centroids[j]
            
            direction = ci0 - cj0
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0])
            u = direction / np.linalg.norm(direction)
            
            Di = shape_mats[i]
            Dj = shape_mats[j]
            
            # convex RHS (constants!)
            rhs = alpha * (np.linalg.norm(Di @ u) + np.linalg.norm(Dj @ u))
            
            constraints.append(u @ (c_vars[i] - c_vars[j]) >= rhs)                              
    
    # # -------------------------------
    # # Non-overlap constraints (approx)
    # # -------------------------------
    # for i in range(K):
    #     for j in range(i+1, K):
    #         if verbose:
    #             print(f"Adding non-overlap constraint between polygon {i} and {j}") 
    #         ci0 = original_centroids[i]
    #         cj0 = original_centroids[j]
    #         if verbose:
    #             print(f"Original centroids: {ci0} vs {cj0}")
    #         # fixed separating direction
    #         direction = ci0 - cj0
    #         if verbose:
    #             print(f"Initial direction: {direction}")
    #         if np.linalg.norm(direction) < 1e-6:
    #             direction = np.array([1.0, 0.0])
    #         direction = direction / np.linalg.norm(direction)
            
    #         # minimal separation (heuristic)            
    #         constraints.append(direction @ (c_vars[i] - c_vars[j]) >= dmin)
    
    # -------------------------------
    # Solve
    # -------------------------------
    obj = cp.Minimize(cp.sum(obj_terms))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    # -------------------------------
    # Extract results
    # -------------------------------
    new_polygons = []
    actual_areas = []
    
    for k in range(K):
        pts = new_pts_exprs[k].value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))
    
    return new_polygons, actual_areas


def PolyAreaRadialLP_batch_position(
    points_list,
    target_areas,
    target_positions,
    w_pos=1.0,
    alpha=1.2
):
    
    points_list = [np.array(p) for p in points_list]
    target_positions = np.array(target_positions)
    
    K = len(points_list)
    
    t_vars, z_vars = [], []
    c_vars = []
    new_pts_exprs = []
    constraints = []
    obj_terms = []
    
    original_centroids = []
    radii = []
    
    # -------------------------------
    # Precompute geometry
    # -------------------------------
    for k in range(K):
        X = points_list[k]
        c0 = np.mean(X, axis=0)
        original_centroids.append(c0)
        
        r = np.max(np.linalg.norm(X - c0, axis=1))
        radii.append(r)
    
    # -------------------------------
    # Build variables per polygon
    # -------------------------------
    for k in range(K):
        X = points_list[k]
        n = len(X)
        c0 = original_centroids[k]
        
        t = cp.Variable(n)
        z = cp.Variable(n)
        c_new = cp.Variable(2)
        
        new_pts = c_new + cp.multiply(t[:, None], (X - c0))
        
        A0 = polygon_areanp(X)
        s = np.sqrt(target_areas[k] / A0)
        
        constraints += [
            cp.mean(t) == s,
            t >= 0.1,
            
            t - 1 <= z,
            -(t - 1) <= z,
            z >= 0
        ]
        
        # shape objective
        obj_terms.append(cp.sum(z))
        
        # centroid-to-target (L1)
        d = cp.Variable(2)
        constraints += [
            c_new - target_positions[k] <= d,
            -(c_new - target_positions[k]) <= d,
            d >= 0
        ]
        obj_terms.append(w_pos * cp.sum(d))
        
        t_vars.append(t)
        z_vars.append(z)
        c_vars.append(c_new)
        new_pts_exprs.append(new_pts)
    
    # -------------------------------
    # Non-overlap (LP relaxation)
    # -------------------------------
    for i in range(K):
        for j in range(i+1, K):
            
            # separation distance
            dmin = alpha * (radii[i] + radii[j])
            
            # slack variables (soft constraint)
            s_ij = cp.Variable(2)  # slack for x,y
            
            constraints += [
                # enforce separation in both axes (soft)
                c_vars[i] - c_vars[j] >= dmin - s_ij,
                c_vars[j] - c_vars[i] >= dmin - s_ij,
                s_ij >= 0
            ]
            
            # penalize overlap
            obj_terms.append(1000 * cp.sum(s_ij))
    
    # -------------------------------
    # Solve LP
    # -------------------------------
    obj = cp.Minimize(cp.sum(obj_terms))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    # -------------------------------
    # Extract results
    # -------------------------------
    new_polygons = []
    actual_areas = []
    
    for k in range(K):
        pts = new_pts_exprs[k].value
        new_polygons.append(pts)
        actual_areas.append(polygon_areanp(pts))
    
    return new_polygons, actual_areas