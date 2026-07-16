import pandas as pd
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.io import *

def load_data(file_path):
    """
    Load data from a CSV file and return a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    

def loader(region, level, year=2020):

    geojson_path = os.path.join(project_dir, f'data/geojson/{region}/{level}.geojson')
    data_path = os.path.join(project_dir, f'data/statistics/{region}/{level}.csv')
    
    df = load_data(data_path)
    polygons, target_areas, target_positions, names, centroid = get_polygon_data(geojson_path)

    counter = 0

    for i, (target_area, name) in enumerate(zip(target_areas, names)):
        match = df[(df["Entity"] == name) & (df["Year"] == year)]

        if not match.empty:
            target_areas[i] = float(match["all years"].iloc[0])
        else:
            counter += 1
            continue  # Skip to the next iteration if no match is found
            # print(f"No match found for {name} in year {year}. Keeping original area: {target_areas[i]}")
    print(f"Number of unmatched regions for year {year}: {counter}")

    return polygons, target_areas, target_positions, names, centroid

