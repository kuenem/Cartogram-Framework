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

def contiguous_cartogram(region, level, year="2016"):
    data, polygons, target_areas, target_positions, names, centroid = loader(region, level, year=year)
    region_names = [k for k in data if k != "__meta__"]
    polygons = []
    polygons = [np.asarray(data[name]["polygon"]) for name in region_names]

    horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)

    new_data = CartogramFramework_global(
        data        = data,
        fixed_points    = None,
        shape           = "contiguous2",
        target_centers  = target_positions,
        horizontal_pairs = horizontal_pairs,
        vertical_pairs   = vertical_pairs,
        neighboring_pairs = neighbouring_pairs(polygons),
        shared_vertices_of_neighbors = shared_vertices_of_neighbors(polygons),

        lambda_shape      = 0.5,
        lambda_area       = 1.0,
        lambda_center     = 0.0,
        lambda_topology   = 0.0,
        lambda_contiguous = 80.0,
        soft_mean_scale   = True,
        lambda_mean_scale = 1e4,
        t_min = 0.01,
        t_max = 1.5,
        lambda_vertex_distance = 10.0,
        vertex_distance_margin = 1.5,
        lambda_repulsion  = 0.0,
        contiguous_area_ratio_cap = np.inf,
        contiguous_use_topology   = False,
        gamma = 1.0,
        beta  = 1.0,
        alpha = 1.0,
        epsilon = 1e-2,
        b       = 1e-2,

        # ── Post-processing ───────────────────────────────────────────
        postprocess_contiguous  = True,     # flip to False to skip entirely
        postprocess_snap_method = "mean",   # midpoint snap on shared vertices
    )

    return new_data