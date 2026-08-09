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

def noncontiguous_cartogram(region, level, year="2016"):
    data, polygons, target_areas, target_positions, names, centroid = loader(region, level, year=year)
    region_names = [k for k in data if k != "__meta__"]
    polygons = []
    polygons = [np.asarray(data[name]["polygon"]) for name in region_names]

    horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)

    new_data = CartogramFramework_global(
        data=data,
        shape="original",

        target_centers=None,          # don't move the centroids yet

        neighboring_pairs=None,
        horizontal_pairs=None,
        vertical_pairs=None,
        shared_vertices_of_neighbors=None,
        soft_mean_scale=True,

        lambda_shape=0.0,             # let area scaling dominate for shrinkage
        lambda_area=1.0,              # keep the soft mean-scale / area target active
        lambda_center=0.0,
        lambda_topology=0.0,

        area_scale=0.1,

        gamma=1.0,
        beta=1.0,

        t_min=1e-8,
        t_max=10000.0,
    )

    return new_data