import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.geometry import *

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