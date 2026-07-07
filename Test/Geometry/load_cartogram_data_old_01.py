"""
load_cartogram_data.py
======================
Unified data loader for cartogram experiments.

Usage
-----
    from load_cartogram_data import load_cartogram_data

    polygon, target_areas, target_positions, names, centroid = load_cartogram_data(
        country="USA",
        detail="states",
        data_type="population",
        year=2016,
        data_dir="data/statistics",   # root folder where CSVs live
        geojson_dir="data/geojson",   # root folder where GeoJSONs live
    )

Supported combinations
-----------------------
country  | detail   | data_type        | years available
---------|----------|------------------|---------------------------------
USA      | states   | population       | 2010–2016 (Census NST-EST)
USA      | states   | election_turnout | 2016 (MIT Election Lab)
USA      | states   | drug_mortality   | 2016 (CDC WONDER)
USA      | counties | population       | 2020 (Census decennial)
NL       | states   | population       | 2017 (CBS Statline)
NL       | states   | house_density    | 2017 (CBS Statline)
NL       | states   | water_surface    | 2017 (CBS Statline)
NL       | counties | population       | 2017 (CBS Statline)
Germany  | states   | population       | 2016 (Destatis)
Germany  | counties | population       | 2016 (Destatis)
France   | states   | population       | 2016 (INSEE)
France   | counties | population       | 2016 (INSEE)
World    | —        | population       | 2016 (World Bank)
World    | —        | gdp              | 2016 (World Bank)
World    | —        | forest_area      | 2016 (World Bank)

GeoJSON sources (download and place at geojson_dir/<country>/<detail>.geojson)
-------------------------------------------------------------------------------
USA states:   https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json
USA counties: https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json
NL states:    https://cartomap.github.io/nl/wgs84/provincie_2017.geojson
NL counties:  https://cartomap.github.io/nl/wgs84/gemeente_2017.geojson
Germany states:   https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json
Germany counties: https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/4_kreise/4_niedrig.geo.json
France states:    https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson
France counties:  https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson
World:            https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson

Statistical data sources (CSV — see DATA_SOURCES dict below for direct URLs)
------------------------------------------------------------------------------
USA population (states, 2010–2016):
  https://www2.census.gov/programs-surveys/popest/tables/2010-2016/state/totals/nst-est2016-01.csv  HAVE IT
  → data/statistics/USA/nst-est2016-01.csv

USA election turnout 2016:
  https://dataverse.harvard.edu/api/access/datafile/3641280   (MIT Election Lab) 	status	"ERROR"  message	"You may not download this file without the required Guestbook response for guestbookID 458."
  → data/statistics/USA/1976-2020-president.csv
  Filter: year==2016, keep state + totalvotes column per state

USA drug-poisoning mortality 2016:
  https://data.cdc.gov/api/views/b3z3-ydkn/rows.csv?accessType=DOWNLOAD  (CDC WONDER)   code	"not_found" error	true    message	"Cannot find view with id b3z3-ydkn"
  → data/statistics/USA/drug_poisoning_mortality.csv

USA counties population 2020:
  https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv HAVE IT
  → data/statistics/USA/co-est2021-alldata.csv

NL provinces (states) — population, house_density, water_surface — 2017:
  https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned JUST LEADS TO A GENERIC WEBSITE HAVE TO FIND DATA MYSELF
  Export as CSV → data/statistics/NL/cbs_provinces_2017.csv
  (Columns needed: RegioNaam, Bevolking_1, Woningdichtheid_7, OppervlakteWater_20)

NL municipalities (counties) population 2017:
  https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned JUST LEADS TO A GENERIC WEBSITE HAVE TO FIND DATA MYSELF
  Same export, filter on gemeente level → data/statistics/NL/cbs_municipalities_2017.csv

Germany states population 2016:
  https://www-genesis.destatis.de/datenbank/online/statistic/12411/table/12411-9010 DOES NOT WORK
  → data/statistics/Germany/destatis_bundeslaender_2016.csv

Germany counties population 2016:
  https://www-genesis.destatis.de/datenbank/online/statistic/12411/table/12411-9020 DOES NOT WORK
  → data/statistics/Germany/destatis_kreise_2016.csv

France regions population 2016:
  https://www.insee.fr/fr/statistiques/fichier/1893198/estim-pop-reg-sexe-gca_1975-2023.xls
  (convert to CSV) → data/statistics/France/insee_regions_2016.csv

France departments population 2016:
  https://www.insee.fr/fr/statistiques/fichier/1893198/estim-pop-dep-sexe-gca_1975-2023.xls
  (convert to CSV) → data/statistics/France/insee_departments_2016.csv

World population + GDP + forest area 2016:
  World Bank bulk download: https://data.worldbank.org/indicator/SP.POP.TOTL  (population)
                            https://data.worldbank.org/indicator/NY.GDP.MKTP.CD (GDP)
                            https://data.worldbank.org/indicator/AG.LND.FRST.ZS (forest %)
  → data/statistics/World/worldbank_population.csv
  → data/statistics/World/worldbank_gdp.csv
  → data/statistics/World/worldbank_forest.csv
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from shapely.geometry import shape, MultiPolygon, Polygon


# ---------------------------------------------------------------------------
# Configuration: map country/detail/data_type → file paths and parse logic
# ---------------------------------------------------------------------------

# Regions to exclude by default per geography
DEFAULT_EXCLUDES: dict[str, set[str]] = {
    "USA_states":    {"Alaska", "Hawaii", "Puerto Rico"},
    "USA_counties":  set(),          # keep all; caller can pass extras
    "NL_states":     set(),
    "NL_counties":   set(),
    "Germany_states":  set(),
    "Germany_counties": set(),
    "France_states":  {"Corse"},     # optional – remove if you want Corsica
    "France_counties": set(),
    "World":         set(),
}

# Name property inside each GeoJSON's feature.properties
GEOJSON_NAME_FIELD: dict[str, str] = {
    "USA_states":     "name",
    "USA_counties":   "NAME",
    "NL_states":      "statnaam",    # cartomap NL uses statnaam for provinces
    "NL_counties":    "statnaam",
    "Germany_states": "NAME_1",
    "Germany_counties": "NAME_2",
    "France_states":  "nom",
    "France_counties": "nom",
    "World":          "ADMIN",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cartogram_data(
    country: Literal["USA", "NL", "GER", "FR", "World"],
    detail: Literal["states", "counties"] | None,
    data_type: str,
    year: int,
    geojson_dir: str = "data/geojson",
    data_dir: str = "data/statistics",
    exclude: Optional[set[str]] = None,
    verbose: bool = False,
) -> tuple[list, list[float], list, list[str], list]:
    """
    Load and preprocess data for cartogram experiments.

    Parameters
    ----------
    country      : One of "USA", "NL", "GER", "FR", "World"
    detail       : "states" or "counties" (None for "World")
    data_type    : e.g. "population", "gdp", "election_turnout", etc.
    year         : The statistical year to use
    geojson_dir  : Root directory containing <country>/<detail>.geojson files
    data_dir     : Root directory containing <Country>/<file>.csv files
    exclude      : Additional region names to exclude on top of defaults
    verbose      : Print debug info

    Returns
    -------
    polygon          : list of coordinate arrays (one per region, outer ring)
    target_areas     : list of float data values (proportional to square areas)
    target_positions : list of [x, y] centroid positions
    names            : list of region name strings
    centroid         : list of (lon, lat) tuples (geographic centroids)
    """
    key = _make_key(country, detail)
    exclude_set = DEFAULT_EXCLUDES.get(key, set()) | (exclude or set())

    # ------------------------------------------------------------------
    # 1. Load GeoJSON
    # ------------------------------------------------------------------
    geojson_path = _resolve_geojson_path(geojson_dir, country, detail)
    features = _load_geojson_features(geojson_path, verbose=verbose)

    print(f"Loading GeoJSON from: {geojson_path}")
    print(f"With the following features: {features}")
    print(f"With the following features: {features[0].keys()}")
    print(type(features[0]))
    print(features[0]["properties"])

    name_field = GEOJSON_NAME_FIELD.get(key, "properties.name")

    print(f"Using name field: {name_field}")

    # ------------------------------------------------------------------
    # 2. Load statistical data
    # ------------------------------------------------------------------
    stat_df = _load_stat_data(country, detail, data_type, year, data_dir, verbose=verbose)

    # ------------------------------------------------------------------
    # 3. Build output lists
    # ------------------------------------------------------------------
    polygon: list = []
    target_areas: list[float] = []
    target_positions: list = []
    names: list[str] = []
    centroid: list = []

    for feat in features:
        name = feat["properties"].get(name_field) or feat["properties"].get("name", "")
        if name in exclude_set:
            if verbose:
                print(f"  Excluding: {name}")
            continue

        # Geometry → largest polygon ring
        coords, geo_centroid = _extract_polygon_and_centroid(feat)
        if coords is None:
            if verbose:
                print(f"  Skipping {name}: empty geometry")
            continue

        # Statistical value
        value = _lookup_value(stat_df, name, year, data_type, country, detail, verbose=verbose)
        if value is None:
            if verbose:
                print(f"  No data for '{name}', skipping")
            continue

        names.append(name)
        polygon.append(coords)
        target_areas.append(float(value))
        target_positions.append(list(np.mean(coords, axis=0)))
        centroid.append(geo_centroid)

    if verbose:
        print(f"Loaded {len(names)} regions.")

    return polygon, target_areas, target_positions, names, centroid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_key(country: str, detail: str | None) -> str:
    if country == "World":
        return "World"
    return f"{country}_{detail}"


def _resolve_geojson_path(root: str, country: str, detail: str | None) -> str:
    root = os.path.join("/home/kuenem/Documents/development/lectures/Master Thesis/", root)
    mapping = {
        "World":          os.path.join(root, "world", "world.geojson"),
        "USA_states":     os.path.join(root, "usa",   "states.geojson"),
        "USA_counties":   os.path.join(root, "usa",   "counties.geojson"),
        "NL_states":      os.path.join(root, "netherlands", "states.geojson"),
        "NL_counties":    os.path.join(root, "netherlands", "counties.geojson"),
        "Germany_states":   os.path.join(root, "germany", "states.geojson"),
        "Germany_counties": os.path.join(root, "germany", "counties.geojson"),
        "France_states":    os.path.join(root, "france",  "states.geojson"),
        "France_counties":  os.path.join(root, "france",  "counties.geojson"),
    }
    key = "World" if country == "World" else f"{country}_{detail}"
    path = mapping.get(key)
    if path is None:
        raise ValueError(f"Unknown country/detail combination: {country}/{detail}")
    return path


def _load_geojson_features(path: str, verbose: bool = False) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GeoJSON not found: {path}\n"
            "See the module docstring for download URLs."
        )
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj.get("features", [])
    if verbose:
        print(f"GeoJSON loaded: {path}  ({len(features)} features)")
    return features


def _extract_polygon_and_centroid(
    feature: dict,
) -> tuple[list | None, tuple[float, float] | None]:
    """Return (outer_ring_coords, (lon_centroid, lat_centroid)) for the
    largest polygon in the feature (handles MultiPolygon automatically)."""
    geom = feature.get("geometry")
    if geom is None:
        return None, None

    shp = shape(geom)
    if shp.is_empty:
        return None, None

    # Pick largest polygon for MultiPolygon
    if isinstance(shp, MultiPolygon):
        shp = max(shp.geoms, key=lambda p: p.area)

    if not isinstance(shp, Polygon):
        return None, None

    coords = list(shp.exterior.coords)
    coords_np = np.array(coords)
    c = shp.centroid
    return coords_np.tolist(), (c.x, c.y)


# ---------------------------------------------------------------------------
# Statistical data loaders
# ---------------------------------------------------------------------------

def _load_stat_data(
    country: str,
    detail: str | None,
    data_type: str,
    year: int,
    data_dir: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Dispatch to the appropriate loader and return a tidy DataFrame with
    columns ['name', 'value']. All name-matching is done case-insensitively
    and with leading-dot stripping (Census format)."""

    loaders = {
        ("USA", "states",   "population"):       _load_usa_state_population,
        ("USA", "states",   "election_turnout"):  _load_usa_state_election,
        ("USA", "states",   "drug_mortality"):    _load_usa_state_drug_mortality,
        ("USA", "counties", "population"):        _load_usa_county_population,
        ("NL",  "states",   "population"):        _load_nl_stat,
        ("NL",  "states",   "house_density"):     _load_nl_stat,
        ("NL",  "states",   "water_surface"):     _load_nl_stat,
        ("NL",  "counties", "population"):        _load_nl_stat,
        ("Germany", "states",   "population"):    _load_germany_stat,
        ("Germany", "counties", "population"):    _load_germany_stat,
        ("France", "states",   "population"):     _load_france_stat,
        ("France", "counties", "population"):     _load_france_stat,
        ("World", None, "population"):            _load_world_stat,
        ("World", None, "gdp"):                   _load_world_stat,
        ("World", None, "forest_area"):           _load_world_stat,
    }

    key = (country, detail, data_type)
    loader = loaders.get(key)
    if loader is None:
        raise ValueError(
            f"Unsupported combination: country={country}, detail={detail}, data_type={data_type}.\n"
            f"Supported keys: {list(loaders.keys())}"
        )
    data_dir = os.path.join("/home/kuenem/Documents/development/lectures/Master Thesis/", data_dir)
    df = loader(data_dir, data_type, year, verbose=verbose)
    if verbose:
        print(f"  Stat data loaded: {len(df)} rows")
    return df


def _lookup_value(
    df: pd.DataFrame,
    name: str,
    year: int,
    data_type: str,
    country: str,
    detail: str | None,
    verbose: bool = False,
) -> float | None:
    """Case-insensitive lookup of `name` in df['name'], return df['value']."""
    mask = df["name"].str.lower() == name.lower()
    rows = df[mask]
    if rows.empty:
        # Try partial match as fallback (e.g., "Île-de-France" vs "ile-de-france")
        import unicodedata
        def norm(s):
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
        mask2 = df["name"].apply(norm) == norm(name)
        rows = df[mask2]
    if rows.empty:
        return None
    return rows.iloc[0]["value"]


# ---- USA ---------------------------------------------------------------

def _load_usa_state_population(data_dir: str, data_type: str, year: int,
                                verbose: bool = False) -> pd.DataFrame:
    """
    Parse nst-est2016-01.csv (Census Bureau NST-EST format).
    The file has messy headers; rows 3+ are data with col-0 = state name
    (leading dot), columns 3–9 = 2010–2016 estimates.
    """
    path = os.path.join(data_dir, "USA", "nst-est2016-01.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\n"
            "Download from: https://www2.census.gov/programs-surveys/popest/"
            "tables/2010-2016/state/totals/nst-est2016-01.csv"
        )
    raw = pd.read_csv(path, header=None, skiprows=3)

    # Year columns: 2010 is col-index 3, each subsequent year +1
    year_col_offset = year - 2010
    if year_col_offset < 0 or year_col_offset > 6:
        raise ValueError(f"Year {year} not in range 2010–2016 for this file.")
    value_col = 3 + year_col_offset

    records = []
    for _, row in raw.iterrows():
        raw_name = str(row.iloc[0]).strip()
        if pd.isna(row.iloc[0]) or not raw_name.startswith("."):
            continue  # skip totals, NaN footers
        state_name = raw_name.lstrip(".")
        raw_val = str(row.iloc[value_col]).replace(",", "").strip()
        try:
            val = float(raw_val)
        except ValueError:
            continue
        records.append({"name": state_name, "value": val})
    return pd.DataFrame(records)


def _load_usa_state_election(data_dir: str, data_type: str, year: int,
                              verbose: bool = False) -> pd.DataFrame:
    """
    MIT Election Lab 1976–2020 presidential results.
    https://dataverse.harvard.edu/api/access/datafile/3641280
    CSV columns include: year, state, totalvotes
    """
    path = os.path.join(data_dir, "USA", "1976-2020-president.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\n"
            "Download from: https://dataverse.harvard.edu/api/access/datafile/3641280\n"
            "and save as data/statistics/USA/1976-2020-president.csv"
        )
    df = pd.read_csv(path)
    df = df[df["year"] == year].copy()
    # totalvotes is the same for all rows of a state in that year → take first
    agg = (
        df.groupby("state", as_index=False)["totalvotes"]
        .first()
        .rename(columns={"state": "name", "totalvotes": "value"})
    )
    # State names come in ALLCAPS in this file → title-case them
    agg["name"] = agg["name"].str.title()
    return agg[["name", "value"]]


def _load_usa_state_drug_mortality(data_dir: str, data_type: str, year: int,
                                    verbose: bool = False) -> pd.DataFrame:
    """
    CDC WONDER drug poisoning mortality.
    https://data.cdc.gov/api/views/b3z3-ydkn/rows.csv?accessType=DOWNLOAD
    Expected columns: State, Year, Age-adjusted Rate
    """
    path = os.path.join(data_dir, "USA", "drug_poisoning_mortality.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\n"
            "Download from: https://data.cdc.gov/api/views/b3z3-ydkn/rows.csv?accessType=DOWNLOAD"
        )
    df = pd.read_csv(path)
    df = df[df["Year"] == year].copy()
    rate_col = "Age-adjusted Rate"
    df = df[["State", rate_col]].rename(columns={"State": "name", rate_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).reset_index(drop=True)


def _load_usa_county_population(data_dir: str, data_type: str, year: int,
                                  verbose: bool = False) -> pd.DataFrame:
    """
    Census POPESTIMATE county file (co-est2021-alldata.csv).
    https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv
    """
    path = os.path.join(data_dir, "USA", "co-est2021-alldata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\n"
            "Download from: https://www2.census.gov/programs-surveys/popest/datasets/"
            "2020-2021/counties/totals/co-est2021-alldata.csv"
        )
    df = pd.read_csv(path, encoding="latin1")
    col = f"POPESTIMATE{year}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not in file. Available years: "
                         f"{[c for c in df.columns if c.startswith('POPESTIMATE')]}")
    df = df[df["SUMLEV"] == 50].copy()  # county rows only
    df["name"] = df["CTYNAME"].str.replace(r"\s+(County|Parish|Borough|Census Area|Municipality)$",
                                             "", regex=True)
    return df[["name", col]].rename(columns={col: "value"})


# ---- Netherlands -------------------------------------------------------

def _load_nl_stat(data_dir: str, data_type: str, year: int,
                   verbose: bool = False) -> pd.DataFrame:
    """
    CBS Statline exports.
    Province file expected columns: RegioNaam, Bevolking_1, Woningdichtheid_7, OppervlakteWater_20
    Municipality file: RegioNaam, Bevolking_1
    See module docstring for download URLs.
    """
    # Try province file first, fall back to municipality
    for fname in (f"states_nl.csv", f"counties_nl.csv"):
        path = os.path.join(data_dir, "NL", fname)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(
            f"No CBS file found under {os.path.join(data_dir, 'NL')}.\n"
            "See module docstring for CBS Statline download instructions."
        )

    col_map = {
        "population":   "Bevolking op 1 januari (aantal)",
        "house_density": "Woningdichtheid_7",
        "water_surface": "OppervlakteWater_20",
    }
    value_col = col_map.get(data_type)
    if value_col is None:
        raise ValueError(f"data_type '{data_type}' not supported for NL. "
                         f"Choose from: {list(col_map)}")

    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.rename(columns={"RegioNaam": "name", value_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["name", "value"]].dropna(subset=["value"])


# ---- Germany -----------------------------------------------------------

def _load_germany_stat(data_dir: str, data_type: str, year: int,
                        verbose: bool = False) -> pd.DataFrame:
    """
    Destatis exports (Bundeslaender or Kreise).
    Expected CSV columns: Name, Bevoelkerung (population), year column if multi-year.
    """
    fname_map = {
        "states":   f"destatis_bundeslaender_{year}.csv",
        "counties": f"destatis_kreise_{year}.csv",
    }
    # Try both since we don't receive detail here; pick whichever exists
    for fname in fname_map.values():
        path = os.path.join(data_dir, "Germany", fname)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(
            f"No Destatis file found under {os.path.join(data_dir, 'Germany')}.\n"
            "See module docstring for download URLs."
        )

    df = pd.read_csv(path, sep=";", decimal=",", encoding="latin1")
    # Normalise column names
    name_col = next((c for c in df.columns if "Name" in c or "name" in c.lower()), df.columns[0])
    val_col  = next((c for c in df.columns if "Bev" in c or "pop" in c.lower() or "Pop" in c), df.columns[1])
    df = df.rename(columns={name_col: "name", val_col: "value"})
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(r"[\s,]", "", regex=True),
                                 errors="coerce")
    return df[["name", "value"]].dropna(subset=["value"])


# ---- France ------------------------------------------------------------

def _load_france_stat(data_dir: str, data_type: str, year: int,
                       verbose: bool = False) -> pd.DataFrame:
    """
    INSEE population estimates (regions or departments).
    Expected columns: Libellé, <year> (population total for that year).
    """
    for fname in (f"insee_regions_{year}.csv", f"insee_departments_{year}.csv"):
        path = os.path.join(data_dir, "France", fname)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(
            f"No INSEE file found under {os.path.join(data_dir, 'France')}.\n"
            "See module docstring for download URLs."
        )

    df = pd.read_csv(path, sep=";", decimal=",", encoding="latin1")
    name_col = next((c for c in df.columns if "Lib" in c or "nom" in c.lower()), df.columns[0])
    year_col = str(year)
    if year_col not in df.columns:
        raise ValueError(f"Column '{year_col}' not in {path}. Columns: {df.columns.tolist()}")
    df = df.rename(columns={name_col: "name", year_col: "value"})
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(r"[\s]", "", regex=True),
                                 errors="coerce")
    return df[["name", "value"]].dropna(subset=["value"])


# ---- World -------------------------------------------------------------

def _load_world_stat(data_dir: str, data_type: str, year: int,
                      verbose: bool = False) -> pd.DataFrame:
    """
    World Bank CSV (wide format): Country Name | Country Code | ... | <year> | ...
    Files:
      worldbank_population.csv
      worldbank_gdp.csv
      worldbank_forest.csv
    """
    fname_map = {
        "population": "worldbank_population.csv",
        "gdp":        "worldbank_gdp.csv",
        "forest_area": "worldbank_forest.csv",
    }
    fname = fname_map.get(data_type)
    if fname is None:
        raise ValueError(f"data_type '{data_type}' not supported for World. "
                         f"Choose from: {list(fname_map)}")

    path = os.path.join(data_dir, "World", fname)
    if not os.path.exists(path):
        urls = {
            "population": "https://data.worldbank.org/indicator/SP.POP.TOTL",
            "gdp":        "https://data.worldbank.org/indicator/NY.GDP.MKTP.CD",
            "forest_area": "https://data.worldbank.org/indicator/AG.LND.FRST.ZS",
        }
        raise FileNotFoundError(
            f"Not found: {path}\nDownload from: {urls[data_type]}"
        )

    # World Bank CSVs have 4 header rows; skip them
    df = pd.read_csv(path, skiprows=4, encoding="latin1")
    year_col = str(year)
    if year_col not in df.columns:
        available = [c for c in df.columns if c.isdigit()]
        raise ValueError(f"Year {year} not available. Found: {available[:5]}…")
    df = df.rename(columns={"Country Name": "name", year_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["name", "value"]].dropna(subset=["value"])


# ---------------------------------------------------------------------------
# Quick smoke-test (only runs when executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Example: USA states population 2016 (requires files to exist)
    try:
        polygon, target_areas, target_positions, names, centroid = load_cartogram_data(
            country="USA",
            detail="states",
            data_type="population",
            year=2016,
            geojson_dir="data/geojson",
            data_dir="data/statistics",
            verbose=True,
        )
        print(f"\nResult: {len(names)} regions")
        print(f"First 5 names:  {names[:5]}")
        print(f"First 5 areas:  {target_areas[:5]}")
        print(f"First position: {target_positions[0]}")
        print(f"First centroid: {centroid[0]}")
    except FileNotFoundError as e:
        print(f"[Expected during test — files not present] {e}", file=sys.stderr)
