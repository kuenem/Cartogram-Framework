import os
import sys
import unicodedata
import warnings
from typing import List, Optional

import pandas as pd

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.io import *

# Column name candidates, checked in order. Add to these lists as new
# datasets/naming conventions show up -- this is the only place that needs
# to change.
NAME_COLUMN_CANDIDATES: List[str] = [
    "Entity", "ISO_CODE", "NUTS_NAME", "shapeName", "GEN",
    "coty_name", "krs_name_short", "statnaam", "name", "NAME", "NAME_1",
]
VALUE_COLUMN_CANDIDATES: List[str] = [
    "Population (people)", "all years", "Population", "value", "Value", "gdp", "electoral_college", "Data",
]


def load_data(file_path):

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def _detect_column(df: pd.DataFrame, candidates: List[str], kind: str,
                    override: Optional[str] = None) -> str:

    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"Requested {kind} column '{override}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return override

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not auto-detect a {kind} column. Tried {candidates}, "
        f"available columns: {list(df.columns)}. "
        f"Pass `{kind}_column=...` explicitly to `loader()`."
    )


def _normalize_name(name: str) -> str:

    normalized = str(name).casefold()
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(
        c for c in decomposed
        if not unicodedata.combining(c) and c.isalnum()
    )


def _build_lookup(df: pd.DataFrame, name_col: str, value_col: str) -> dict:

    lookup = {}
    dupes = set()
    for _, row in df.iterrows():
        name = row[name_col]
        if pd.isna(name):
            continue
        value = row[value_col]
        if pd.isna(value):
            continue
        if name in lookup:
            dupes.add(name)
            continue  # keep first occurrence
        lookup[name] = float(value)

    if dupes:
        warnings.warn(
            f"Duplicate region name(s) in '{name_col}' column, kept first "
            f"occurrence only: {sorted(dupes)}"
        )
    return lookup


def loader(
    region: str,
    level: str,
    year: Optional[int] = None,
    name_column: Optional[str] = None,
    value_column: Optional[str] = None,
    year_column: str = "Year",
    geojson_path: Optional[str] = None,
    data_path: Optional[str] = None,
    fuzzy_fallback: bool = True,
):

    geojson_path = geojson_path or os.path.join(
        project_dir, f'data/geojson/{region}/{level}.geojson'
    )
    data_path = data_path or os.path.join(
        project_dir, f'data/statistics/{region}/{level}.csv'
    )

    df = load_data(data_path)
    if df is None:
        raise FileNotFoundError(f"Could not load statistics file: {data_path}")

    data = {}

    data, polygons, target_areas, target_positions, names, centroid = get_polygon_data(geojson_path, data)

    name_col = _detect_column(df, NAME_COLUMN_CANDIDATES, "name", override=name_column)
    value_col = _detect_column(df, VALUE_COLUMN_CANDIDATES, "value", override=value_column)

    if year is not None and year_column in df.columns:
        df = df[df[year_column] == year]

    lookup = _build_lookup(df, name_col, value_col)
    fuzzy_lookup = {_normalize_name(k): v for k, v in lookup.items()} if fuzzy_fallback else {}

    counter = 0
    unmatched = []
    for i, name in enumerate(names):
        if name in lookup:
            target_areas[i] = lookup[name]
            data[name]["target_area"] = lookup[name]
            continue
        if fuzzy_fallback and _normalize_name(name) in fuzzy_lookup:
            target_areas[i] = fuzzy_lookup[_normalize_name(name)]
            data[name]["target_area"] = fuzzy_lookup[_normalize_name(name)]
            continue
        counter += 1
        unmatched.append(name)
        # target_areas[i] left as original polygon area (shoelace fallback)

    print(f"Matched name column: '{name_col}', value column: '{value_col}'")
    print(f"Number of unmatched regions for year {year}: {counter}")
    if unmatched:
        print(f"Unmatched region names: {unmatched}")

    return data, polygons, target_areas, target_positions, names, centroid
