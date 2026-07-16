"""
load_data.py

Generalized statistics loader for the CartogramFramework.

Handles CSV schemas that vary across your datasets:
  - Entity, Year, all years                      (old world/time-series format)
  - ISO_CODE, Population (people), ...            (world, by ISO code)
  - NUTS_NAME, Population (people), ...            (EU NUTS regions)
  - shapeName, Population (people), ...            (Germany/US states, geoBoundaries style)

The loader auto-detects which column holds the region identifier and which
holds the statistic to use as target area, so `loader()` keeps its old
call signature/behavior for existing data but also works with the new
schemas without you having to touch calling code.
"""

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
    "coty_name", "krs_name_short", "statnaam", "name", "NAME",
]
VALUE_COLUMN_CANDIDATES: List[str] = [
    "Population (people)", "all years", "Population", "value", "Value",
]


def load_data(file_path):
    """
    Load data from a CSV file and return a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the loaded data, or None on error.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def _detect_column(df: pd.DataFrame, candidates: List[str], kind: str,
                    override: Optional[str] = None) -> str:
    """Pick the first matching column from `candidates`, or use `override`
    if given. Raises with a helpful message if nothing matches, since a
    silent wrong guess here would corrupt every target area downstream.
    """
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
    """Whitespace/case/accent-insensitive key for fallback matching, e.g.
    ' Abruzzo ' vs 'Abruzzo', or 'Åland' vs 'Aland'/'ALAND'. Diacritics are
    stripped via NFKD decomposition, since umlauts/accents differ across
    your NL/DE/FR datasets between geojson and CSV encodings."""
    normalized = " ".join(str(name).split()).casefold()
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _build_lookup(df: pd.DataFrame, name_col: str, value_col: str) -> dict:
    """Build {name: value} lookup. Warns (once) on duplicate region names
    instead of silently keeping the first or last row -- duplicates
    usually mean the name column doesn't uniquely key the dataset."""
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
    """
    Load geometry + statistics for a region/level and merge them into
    target areas, keeping the original polygon area as fallback for any
    region without a statistics match.

    Parameters:
    - region, level: used to build default geojson/csv paths, as before:
        data/geojson/{region}/{level}.geojson
        data/statistics/{region}/{level}.csv
    - year: if given AND the CSV has a `year_column` (default "Year"),
        filters rows to that year first (old Entity/Year/"all years"
        format). If the CSV has no such column (e.g. the ISO_CODE /
        NUTS_NAME / shapeName population files), this is a no-op.
    - name_column: force which CSV column identifies the region (e.g.
        "shapeName"). If None, auto-detected from NAME_COLUMN_CANDIDATES.
        Must match whatever property get_all_points() picked as `name`
        for the corresponding geojson (name/NAME/coty_name/GEN/
        krs_name_short/statnaam/ISO_CODE).
    - value_column: force which CSV column holds the statistic to use as
        target area. If None, auto-detected from VALUE_COLUMN_CANDIDATES.
    - geojson_path / data_path: override the default path construction
        entirely (useful for one-off files that don't follow the
        data/{geojson,statistics}/{region}/{level} convention).
    - fuzzy_fallback: if a region isn't found under an exact name match,
        retry with whitespace/case-insensitive comparison before giving
        up (handles things like accented names or stray whitespace).

    Returns: (polygons, target_areas, target_positions, names, centroid)
        -- same as before. target_areas are floats; any unmatched region
        keeps its original polygon area (shoelace) as target, so it's
        visually flagged (Cartographic error -> 0 for that region) but
        does not raise or need to be dropped from the map.
    """
    geojson_path = geojson_path or os.path.join(
        project_dir, f'data/geojson/{region}/{level}.geojson'
    )
    data_path = data_path or os.path.join(
        project_dir, f'data/statistics/{region}/{level}.csv'
    )

    df = load_data(data_path)
    if df is None:
        raise FileNotFoundError(f"Could not load statistics file: {data_path}")

    polygons, target_areas, target_positions, names, centroid = get_polygon_data(geojson_path)

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
            continue
        if fuzzy_fallback and _normalize_name(name) in fuzzy_lookup:
            target_areas[i] = fuzzy_lookup[_normalize_name(name)]
            continue
        counter += 1
        unmatched.append(name)
        # target_areas[i] left as original polygon area (shoelace fallback)

    print(f"Matched name column: '{name_col}', value column: '{value_col}'")
    print(f"Number of unmatched regions for year {year}: {counter}")
    if unmatched:
        print(f"Unmatched region names: {unmatched}")

    return polygons, target_areas, target_positions, names, centroid
