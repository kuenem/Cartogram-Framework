import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.io import *
from src.geometry import *
from src.utils import *
from src.viz import *
from src.core import *
from src.tests import *

def dorling_cartogram(region, level, year="2016"):
    data, polygons, target_areas, target_positions, names, centroid = loader(region, level, year=year)
    region_names = [k for k in data if k != "__meta__"]
    polygons = []
    polygons = [np.asarray(data[name]["polygon"]) for name in region_names]

    horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)

    new_data = CartogramFramework_global(
        data        = data,
        fixed_points    = None,
        shape           = "circle",
        target_centers  = target_positions,
        horizontal_pairs = horizontal_pairs,
        vertical_pairs   = vertical_pairs,
        neighboring_pairs = neighbouring_pairs(polygons),
        shared_vertices_of_neighbors=shared_vertices_of_neighbors(polygons),

        lambda_shape    = 0.1,   # small — circles form without needing shape pressure
        lambda_area     = 1.0,
        lambda_center   = 0.5,   # moderate centre fidelity keeps circles near geography
        lambda_topology = 1.0,

        gamma = 0.0,   # REQUIRED — pure E_circle
        beta  = 1.0,
        alpha = 1.0,
        epsilon = 1e-2,
        b       = 1e-2,
        t_min   = 0.01,
        t_max   = 5.0,
        shape_deformation = 0.0,   # triggers make_circle()
    )

    return new_data