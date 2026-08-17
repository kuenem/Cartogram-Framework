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
from src.core.preprocessing_data import preprocess, auto_area_scale

def noncontiguous_cartogram(region, level, year="2016", area_scale="auto", area_scale_cap=1.0):
    """
    Parameters
    ----------
    area_scale : float or "auto" (default)
        "auto" derives the largest-safe uniform area_scale so that no
        region's target area exceeds `area_scale_cap` * its own original
        area (see `auto_area_scale`). This is what prevents the
        relatively-biggest-growing region from ballooning past its own
        original footprint and overlapping neighbours that never moved.
        Pass a float to opt back into the old fixed-multiplier behaviour.
    area_scale_cap : float
        Growth-ratio ceiling used when area_scale="auto". 1.0 (default)
        means "no region may exceed its own original area". Values < 1.0
        add extra safety margin; values > 1.0 permit some controlled
        overgrowth if you've verified your specific layout tolerates it.
    """
    data, polygons, target_areas, target_positions, names, centroid = loader(region, level, year=year)
    region_names = [k for k in data if k != "__meta__"]
    polygons = []
    polygons = [np.asarray(data[name]["polygon"]) for name in region_names]

    horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)

    if area_scale == "auto":
        # Run the same sum-normalisation preprocess() would do internally,
        # purely to get the normalised target areas to compute the safe
        # scale from. CartogramFramework_global() re-runs preprocess()
        # itself on the untouched `data`, so this doesn't consume or
        # mutate anything the actual solve needs.
        _, _, normalised_target_areas, _, _, _ = preprocess(data, None, 1.0, None)
        resolved_area_scale = auto_area_scale(data, normalised_target_areas, cap=area_scale_cap)
        print(
            f"  noncontiguous_cartogram: auto area_scale = {resolved_area_scale:.4f} "
            f"(worst-case region growth capped at {area_scale_cap}x its original area)"
        )
    else:
        resolved_area_scale = float(area_scale)

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

        area_scale=resolved_area_scale,

        gamma=1.0,
        beta=1.0,

        t_min=1e-8,
        t_max=10000.0,
    )

    return new_data