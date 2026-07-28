import geojson
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.geometry import *
from src.utils import *
from src.viz import *

def parse_geojson(path):
    with open(path) as f:
        gj = geojson.load(f)
    return gj


def get_all_points(path_to_file, verbose=False):
    gj = parse_geojson(path_to_file)

    points = {}

    for feature in gj["features"]:

        if "name" in feature["properties"]:
            name = feature["properties"]["name"]
        elif "NAME" in feature["properties"]:
            name = feature["properties"]["NAME"]
        elif "coty_name" in feature["properties"]:
            name = feature["properties"]["coty_name"]
        elif "GEN" in feature["properties"]:
            name = feature["properties"]["GEN"]
        elif "krs_name_short" in feature["properties"]:
            name = feature["properties"]["krs_name_short"][0]
        elif "statnaam" in feature["properties"]:
            name = feature["properties"]["statnaam"]
        else:
            name = feature["properties"]["ISO_CODE"]

        points[name] = {
            "type": feature["geometry"]["type"],
            "coords": feature["geometry"]["coordinates"],
        }
        

    return points



def outer_ring(poly):
    return poly[0]   # exterior ring


def get_polygon_data(path_to_file, data, exclude={'Alaska', 'Hawaii', 'Puerto Rico'}):

    polygons = []
    target_areas = []
    target_positions = []
    names = []
    centroid = []

    points_dict = get_all_points(path_to_file)

    for name, geom in points_dict.items():
        data[name] = {}

        geom_type = geom["type"]
        coords = geom["coords"]
        
        if geom_type == "Polygon":

            points = outer_ring(coords)

        elif geom_type == "MultiPolygon":

            candidates = []

            for poly in coords:

                ring = outer_ring(poly)
                area = polygon_areanp(ring)

                candidates.append((area, ring))

            points = max(candidates, key=lambda x: x[0])[1]

        else:
            print(f"Skipping unsupported geometry {geom_type} for {name}")
            continue

        points = np.asarray(points, dtype=float)

        names.append(name)

        area = float(polygon_areanp(points))
        target_areas.append(area)
        data[name]["area"] = area

        polygons.append(points)
        data[name]["polygon"] = points

        center = np.mean(points, axis=0)
        target_positions.append(center)
        data[name]["target_positions"] = center
        centroid.append(center)
        data[name]["centroid"] = center

    if exclude:
        filtered = [(name, pts, trgt_rs) for name, pts, trgt_rs in zip(names, polygons, target_areas) if name not in exclude]
        names = [n for n, _, _ in filtered]
        polygons = [p for _, p, _ in filtered]
        target_areas = [trgt_rs for _, _, trgt_rs in filtered]

    return data, polygons, target_areas, target_positions, names, centroid