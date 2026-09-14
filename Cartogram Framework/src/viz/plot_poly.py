import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import Normalize

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.geometry import *

def plot_polys(polygons, title="Polygons", plot_points=False, centroids=False, label_vertices=False, legend=True, target_centers=None, names=None, svg=False):
    
    plt.figure(figsize=[12.8, 9.6])
    
    for idx, points in enumerate(polygons):
        points = np.array(points)

        if points.shape[0] == 0:
            print(f"[plot_polys] skipping Poly {idx}: 0 vertices")
            continue

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
    if svg:
        plt.savefig(f"{title}.svg", format='svg')
    plt.show()


def plot_polys_data(data, 
                    title="Polygons", 
                    plot_original_outlines=False, 
                    plot_points=False, 
                    centroids=False, 
                    label_vertices=False, 
                    legend=False, 
                    target_centers=None, 
                    names=None, 
                    svg=False, 
                    new=True):
    
    plt.figure(figsize=[12.8, 9.6])

    values = []
    values = [data[name]["target_area"] for name in data if name != "__meta__"]

    norm = Normalize(
        vmin=min(values),
        vmax=max(values)
    ) if values else None

    # Choose ONE hue / color map
    cmap = plt.get_cmap("Greens")
    
    for idx, name in enumerate(data):
        if name == '__meta__':
            continue
        
        if new and 'new_polygon' in data[name]:
            points = np.array(data[name]['new_polygon'])
            original = np.array(data[name]['polygon'])
        else:
            points = np.array(data[name]['polygon'])

        if points.shape[0] == 0:
            print(f"[plot_polys_data] skipping '{name}': 0 vertices")
            continue

        # close polygon
        poly = np.vstack([points, points[0]])
        original_poly = np.vstack([original, original[0]]) if plot_original_outlines else None

        plt.plot(original_poly[:,0], original_poly[:,1], color='gray', linestyle='--', linewidth=0.1, alpha=0.2, label=f"Original {name}") if plot_original_outlines else None

        value = data[name]["target_area"]
        polygon_color = cmap(norm(value))
        
        # filled polygon (slightly transparent so overlaps are visible)
        plt.fill(poly[:,0], poly[:,1], alpha=0.3, label=f"Poly {idx}")

        
        
        if plot_points:
            # edges
            plt.plot(poly[:,0], poly[:,1], marker='o')
        
        if centroids:
            centroid = np.mean(points, axis=0)
            plt.plot(centroid[0], centroid[1], marker='*', color='red', markersize=5)
            if plot_original_outlines:
                original_centroid = np.mean(original, axis=0)
                plt.plot(original_centroid[0], original_centroid[1], marker='*', color='blue', markersize=5, label=f"Original Centroid {name}")

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
    
    # plt.axis('scaled')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    # plt.title(title)
    if legend:
        plt.legend()
    if svg:
        plt.savefig(f"{title}.png", format='png')
    plt.show()


def plot_polys_data_coro(data, 
                    title="Polygons", 
                    plot_original_outlines=False, 
                    plot_points=False, 
                    centroids=False, 
                    label_vertices=False, 
                    legend=False, 
                    target_centers=None, 
                    names=None, 
                    svg=False, 
                    new=False):
    
    plt.figure(figsize=[12.8, 9.6])

    values = []
    values = [data[name]["target_area"] for name in data if name != "__meta__"]

    norm = Normalize(
        vmin=min(values),
        vmax=max(values)
    ) if values else None

    # Choose ONE hue / color map
    cmap = plt.get_cmap("Greens")
    
    for idx, name in enumerate(data):
        if name == '__meta__':
            continue
        
        if new and 'new_polygon' in data[name]:
            points = np.array(data[name]['new_polygon'])
            original = np.array(data[name]['polygon'])
        else:
            points = np.array(data[name]['polygon'])

        if points.shape[0] == 0:
            print(f"[plot_polys_data] skipping '{name}': 0 vertices")
            continue

        # close polygon
        poly = np.vstack([points, points[0]])
        original_poly = np.vstack([original, original[0]]) if plot_original_outlines else None

        plt.plot(original_poly[:,0], original_poly[:,1], color='gray', linestyle='--', linewidth=0.1, alpha=0.2, label=f"Original {name}") if plot_original_outlines else None

        value = data[name]["target_area"]
        polygon_color = cmap(norm(value))
        
        # filled polygon (slightly transparent so overlaps are visible)
        plt.fill(poly[:,0], poly[:,1], alpha=0.3, label=f"Poly {idx}", color=polygon_color)

        
        
        if plot_points:
            # edges
            plt.plot(poly[:,0], poly[:,1], marker='o')
        
        if centroids:
            centroid = np.mean(points, axis=0)
            plt.plot(centroid[0], centroid[1], marker='*', color='red', markersize=5)
            if plot_original_outlines:
                original_centroid = np.mean(original, axis=0)
                plt.plot(original_centroid[0], original_centroid[1], marker='*', color='blue', markersize=5, label=f"Original Centroid {name}")

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
    
    # plt.axis('scaled')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    # plt.title(title)
    if legend:
        plt.legend()
    if svg:
        plt.savefig(f"{title}.png", format='png')
    plt.show()


def plot_poly(points, title="Polygon", label_vertices=False):
    # close polygon
    poly = np.vstack([points, points[0]])

    plt.figure()

    # filled polygon
    plt.fill(poly[:,0], poly[:,1], alpha=0.4)

    # edges
    plt.plot(poly[:,0], poly[:,1])

    # label vertices
    if label_vertices:
        for i,(x,y) in enumerate(points):
            plt.text(x, y, f"P{i}")
    ax = plt.gca()
    ax.set_axis_off()

    plt.gca().set_aspect('equal')
    plt.savefig("Limburg.png", format='png')
    # plt.title(title)
    plt.show()