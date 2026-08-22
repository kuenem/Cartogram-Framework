import os
import sys
import numpy as np

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.cartograms import *
from src.utils.common import *
from src.tests import *

def topological_accuracy(cartogram_polygons, map_polygons):
    Ec = set(neighbouring_pairs(cartogram_polygons))
    Em = set(neighbouring_pairs(map_polygons))

    return 1 - len(Ec & Em) / len(Ec | Em)


def cartographic_errors(data, new_area_key="new_area", target_area_key="target_area"):
    errors = []
    for state in data.keys():
        errors.append(abs(data[state][new_area_key] - data[state][target_area_key])/max(data[state][new_area_key], data[state][target_area_key]))
    return errors


def _normalize_unit_area(poly_coords):
    """Center on centroid and rescale to unit area, per Nusrat et al. Sec 4."""
    poly = ShapelyPolygon(poly_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)  
    area = poly.area
    if area <= 0:
        raise ValueError("Broken Polygon: negative or zero area")
    cx, cy = poly.centroid.x, poly.centroid.y
    scale = 1.0 / np.sqrt(area)
    coords = np.asarray(poly.exterior.coords)
    coords = (coords - [cx, cy]) * scale
    return ShapelyPolygon(coords)


def hamming_distance(data, org_data, map_key = "polygon", cartogram_key = "new_polygon"):
    errors = []
    for state in data.keys():
        original_poly = org_data[state][map_key]
        new_poly = data[state][cartogram_key]
        p_orig = _normalize_unit_area(original_poly)
        p_new = _normalize_unit_area(new_poly)
        sym_diff_area = p_orig.symmetric_difference(p_new).area
        errors.append(sym_diff_area / 2.0)
    return errors


def quality_criteria(data, org_data = None, stat_data = None, new_area_key="new_area", map_key = "polygon", target_area_key="target_area", cartogram_key = "new_polygon"):
    errors = {}
    if not org_data:
        org_data = data
    if not stat_data:
        stat_data = data
    carto_errors = cartographic_errors(data, new_area_key=new_area_key, target_area_key=target_area_key)
    errors["Mean cartographic error"] = np.mean(carto_errors)
    errors["Max cartographic error"] = np.max(carto_errors)

    cartogram_polygons = [data[stat][cartogram_key] for stat in data.keys()]
    map_polygons = [org_data[stat][map_key] for stat in org_data.keys()]

    errors["Topological accuracy"] = topological_accuracy(cartogram_polygons, map_polygons)
    errors["Mean Hamming distance"] = np.mean(hamming_distance(data, org_data, map_key=map_key, cartogram_key=cartogram_key))
    errors["Max Hamming distance"] = np.max(hamming_distance(data, org_data, map_key=map_key, cartogram_key=cartogram_key))