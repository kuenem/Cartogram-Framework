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
NL       | states   | population       | 1988, 2000, 2005, 2019, 2020 (CBS Statline)
NL       | counties | population       | 1988, 2000, 2005, 2019, 2020 (CBS Statline)
Germany  | states   | population       | 2024 (Destatis 12411-0042)
Germany  | counties | population       | 2020–2024 (Destatis 12411-0015)
France   | states   | population       | 2016 (INSEE)
France   | counties | population       | 2016 (INSEE)
World    | —        | population       | 2016 (World Bank)
World    | —        | gdp              | 2016 (World Bank)
World    | —        | forest_area      | 2016 (World Bank)

Note on NL/Germany: the original plan was to also pull house_density and
water_surface for NL, and to use a "Bevoelkerung" style simple CSV for
Germany. The CBS Statline portal and the Destatis GENESIS table pages do
not expose stable, scriptable download URLs (both lead to interactive
query builders), so the actual files used here were built by hand through
each site's table/export UI and are population-only, single-indicator
exports. See the loader functions below for the exact (messy)
column layout of each file, since both NL and Germany ship raw
"as-exported" structure rather than tidy long-format CSVs.

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

Statistical data sources (CSV)
------------------------------------------------------------------------------
USA population (states, 2010–2016):
  https://www2.census.gov/programs-surveys/popest/tables/2010-2016/state/totals/nst-est2016-01.csv  [WORKS]
  → data/statistics/USA/nst-est2016-01.csv

USA election turnout 2016:
  MIT Election Lab dataverse page requires a guestbook click-through before
  download, so the bare API URL 403s. Get it manually from:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42MVDX
  → data/statistics/USA/1976-2020-president.csv

USA drug-poisoning mortality 2016:
  The CDC WONDER view id used previously (b3z3-ydkn) no longer resolves.
  Use CDC WONDER's "Underlying Cause of Death" online query tool instead:
  https://wonder.cdc.gov/ucd-icd10.html  (query drug poisoning ICD-10 codes,
  group by state, year 2016) and export the result.
  → data/statistics/USA/drug_poisoning_mortality.csv

USA counties population 2020:
  https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv  [WORKS]
  → data/statistics/USA/co-est2021-alldata.csv

NL provinces (states) population:
  CBS Statline table 70072ned ("Bevolking; geslacht, leeftijd, burgerlijke
  staat en regio") does not expose a stable direct-download URL — its portal
  link only opens the interactive table. Use the StatLine UI
  (https://opendata.cbs.nl/statline/#/CBS/nl/dataset/70072ned/table) to
  select "Regio's" = provinces, export as CSV (";"-separated, semicolon
  decimal comma). Years available in the export used here: 1988, 2000,
  2005, 2019, 2020.
  → data/statistics/NL/states_nl.csv

NL municipalities (counties) population:
  Same StatLine table 70072ned, with "Regio's" = gemeenten instead of
  provincies.
  → data/statistics/NL/counties_nl.csv

Germany states population:
  Destatis GENESIS table 12411-0042 ("Durchschnittliche Bevölkerung:
  Bundesländer, Jahre"). The /datenbank/online/statistic/.../table/...
  URLs used previously no longer resolve — GENESIS tables must be opened
  via the table search (https://www-genesis.destatis.de/genesis/online,
  search "12411-0042") and exported as CSV from there. The export used
  here only has 2024 data (single year, "Insgesamt" total column).
  → data/statistics/Germany/states_de.csv

Germany counties population:
  Destatis GENESIS table 12411-0015 ("Bevölkerung: Kreise, Stichtag").
  Same access path as above (search "12411-0015"). Export used here
  covers 31.12.2020–31.12.2024 (5 reference dates).
  → data/statistics/Germany/counties_de.csv

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
    country: Literal["USA", "NL", "Germany", "France", "World"],
    detail: Literal["states", "counties"] | None,
    data_type: str,
    year: int,
    geojson_dir: str = "data/geojson",
    data_dir: str = "data/statistics",
    exclude: Optional[set[str]] = None,
    fill_with_values_from_other_years: bool = False,
    verbose: bool = False,
) -> tuple[list, list[float], list, list[str], list]:
    """
    Load and preprocess data for cartogram experiments.

    Parameters
    ----------
    country      : One of "USA", "NL", "Germany", "France", "World"
    detail       : "states" or "counties" (None for "World")
    data_type    : e.g. "population", "gdp", "election_turnout", etc.
    year         : The statistical year to use
    geojson_dir  : Root directory containing <country>/<detail>.geojson files
    data_dir     : Root directory containing <Country>/<file>.csv files
    exclude      : Additional region names to exclude on top of defaults
    fill_with_values_from_other_years :
        If a region has no value for `year` in the source data, backfill
        it from the closest year (by absolute distance, ties broken
        towards the earlier year) that does have a value for that region,
        instead of dropping the region. Currently only implemented for
        NL (states/counties); other loaders accept and ignore this flag.
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

    name_field = GEOJSON_NAME_FIELD.get(key, "name")

    # ------------------------------------------------------------------
    # 2. Load statistical data
    # ------------------------------------------------------------------
    data_dir = os.path.join("/home/kuenem/Documents/development/lectures/Master Thesis/", data_dir)
    stat_df = _load_stat_data(
        country, detail, data_type, year, data_dir,
        fill_with_values_from_other_years=fill_with_values_from_other_years,
        verbose=verbose,
    )

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
    fill_with_values_from_other_years: bool = False,
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
    df = loader(
        data_dir, detail, data_type, year,
        fill_with_values_from_other_years=fill_with_values_from_other_years,
        verbose=verbose,
    )
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

def _load_usa_state_population(data_dir: str, detail: str | None, data_type: str, year: int,
                                fill_with_values_from_other_years: bool = False,
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


def _load_usa_state_election(data_dir: str, detail: str | None, data_type: str, year: int,
                              fill_with_values_from_other_years: bool = False,
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


def _load_usa_state_drug_mortality(data_dir: str, detail: str | None, data_type: str, year: int,
                                    fill_with_values_from_other_years: bool = False,
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


def _load_usa_county_population(data_dir: str, detail: str | None, data_type: str, year: int,
                                  fill_with_values_from_other_years: bool = False,
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

def _load_nl_stat(data_dir: str, detail: str | None, data_type: str, year: int,
                   fill_with_values_from_other_years: bool = False,
                   verbose: bool = False) -> pd.DataFrame:
    """
    CBS Statline table 70072ned export, as actually downloaded through the
    StatLine UI. Long format, ';'-separated, UTF-8 with BOM, decimal comma.

    Columns:
        Geslacht;Leeftijd;Burgerlijke staat;Regio's;Perioden;
        Bevolking op 1 januari (aantal);Gemiddelde bevolking  (aantal)

    One row per (region, year); we use the "Bevolking op 1 januari"
    (population on Jan 1st) column as the population figure, since it has
    no missing values (unlike the "Gemiddelde bevolking" / average
    population column, which is NaN for the earliest year in the export).

    Province rows carry a " (PV)" suffix on the region name (e.g.
    "Utrecht (PV)") to disambiguate from same-named municipalities; this
    is stripped here so names line up with the GeoJSON's plain province
    names. The country-total row ("Nederland") is dropped.

    Only data_type == "population" is supported — the export used here
    does not include house_density or water_surface (CBS does not expose
    a stable scriptable download for table 70072ned; see module docstring).

    Missing values. Every region has a row for every year present in the
    file, but the population *value* is frequently NaN — typically because
    a municipality didn't yet exist, or had already merged into another
    one, around that reference date (e.g. "'s-Gravenmoer" only has a value
    for 1988, all later years are NaN since it merged into another
    municipality). By default such regions are simply dropped for years
    where they have no value (same as before). If
    fill_with_values_from_other_years=True, a region missing a value for
    the requested `year` is instead backfilled from the *closest available
    year* for that same region (ties broken by picking the earlier year);
    a region is only dropped if it has no value in any year of the file.
    """
    if data_type != "population":
        raise ValueError(
            f"data_type '{data_type}' not supported for NL. Only 'population' "
            "is available from the CBS export used here. See module docstring."
        )

    fname_map = {"states": "states_nl.csv", "counties": "counties_nl.csv"}
    fname = fname_map.get(detail)
    if fname is None:
        raise ValueError(f"detail '{detail}' not supported for NL. Choose 'states' or 'counties'.")

    path = os.path.join(data_dir, "NL", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\nSee module docstring for CBS Statline export instructions."
        )

    full = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    available_years = sorted(full["Perioden"].unique())
    if year not in available_years:
        raise ValueError(f"Year {year} not in {path}. Available years: {available_years}")

    full["name"] = full["Regio's"].str.replace(r"\s*\(PV\)$", "", regex=True)
    full = full[full["name"] != "Nederland"]  # drop country total

    value_col = "Bevolking op 1 januari (aantal)"
    full["value"] = pd.to_numeric(full[value_col], errors="coerce")

    if not fill_with_values_from_other_years:
        df = full[full["Perioden"] == year]
        return df[["name", "value"]].dropna(subset=["value"])

    # For each region, pick the value from the year closest to the
    # requested one (ties -> earlier year), among years that have a value.
    def pick_closest(group: pd.DataFrame) -> float:
        have_value = group.dropna(subset=["value"])
        if have_value.empty:
            return float("nan")
        diffs = (have_value["Perioden"] - year).abs()
        # stable sort: abs distance asc, then year asc (so ties prefer earlier year)
        order = have_value.assign(_dist=diffs).sort_values(["_dist", "Perioden"])
        return order.iloc[0]["value"]

    filled = (
        full.groupby("name", sort=False)
        .apply(pick_closest, include_groups=False)
        .rename("value")
        .reset_index()
    )

    if verbose:
        requested = full[full["Perioden"] == year].set_index("name")["value"]
        n_filled = int(((requested.isna()) & filled.set_index("name")["value"].notna()).sum())
        if n_filled:
            print(f"  Filled {n_filled} regions from a different year (requested year {year} was NaN)")

    return filled.dropna(subset=["value"])


# ---- Germany -----------------------------------------------------------

def _load_germany_stat(data_dir: str, detail: str | None, data_type: str, year: int,
                        fill_with_values_from_other_years: bool = False,
                        verbose: bool = False) -> pd.DataFrame:
    """
    Destatis GENESIS exports, as actually downloaded ("as-exported" raw
    structure — see module docstring for why no tidy version exists).
    Both files are ';'-separated, UTF-8 with BOM, with several metadata/
    header rows above the data and a footer block (source note, copyright,
    timestamp) below it that must be skipped/filtered.

    states_de.csv (table 12411-0042, Bundesländer):
        7 metadata/header rows, then one row per state:
            Jahr;Bundesland;<18 numeric/flag columns>
        Column 18 (0-indexed) is "Insgesamt / Insgesamt" — total population,
        both sexes, all nationalities combined. Only year 2024 is present.

    counties_de.csv (table 12411-0015, Kreise):
        6 metadata/header rows, then one row per county:
            AGS;Name;val_2020;e;val_2021;e;val_2022;e;val_2023;e;val_2024;e
        i.e. 5 reference dates (31.12.2020 .. 31.12.2024), each as a
        (value, flag) column pair starting at column index 2.
        County names carry a type suffix (", Landkreis" / ", kreisfreie
        Stadt" / ", Kreis") which is stripped here so they line up with
        the plain names in the GeoJSON.

    Footer rows (the "__________" separator and the notes/copyright/
    timestamp lines below it) are dropped via the NaN/non-numeric name
    or value, since they don't parse as state/county names or numbers.
    """
    if data_type != "population":
        raise ValueError(
            f"data_type '{data_type}' not supported for Germany. Only 'population' "
            "is available from the Destatis exports used here."
        )

    if detail == "states":
        path = os.path.join(data_dir, "Germany", "states_de.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Not found: {path}\nSee module docstring for Destatis 12411-0042 export instructions."
            )
        if year != 2024:
            raise ValueError(
                f"Year {year} not available for Germany states — the export used here "
                "only contains 2024. Re-export table 12411-0042 with more years if needed."
            )
        df = pd.read_csv(path, sep=";", skiprows=7, header=None, encoding="utf-8-sig")
        df = df.rename(columns={0: "year", 1: "name", 18: "value"})
        df = df[["name", "value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["name", "value"])

    elif detail == "counties":
        path = os.path.join(data_dir, "Germany", "counties_de.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Not found: {path}\nSee module docstring for Destatis 12411-0015 export instructions."
            )
        if year < 2020 or year > 2024:
            raise ValueError(
                f"Year {year} not available for Germany counties — the export used here "
                "covers 2020-2024 (reference date 31.12 of each year)."
            )
        value_col = 2 + 2 * (year - 2020)
        df = pd.read_csv(path, sep=";", skiprows=6, header=None, encoding="utf-8-sig")
        df = df.rename(columns={0: "ags", 1: "name", value_col: "value"})
        df = df[["name", "value"]].copy()
        df["name"] = df["name"].astype(str).str.replace(
            r",\s*(kreisfreie Stadt|Landkreis|Kreis)$", "", regex=True
        )
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["name", "value"])

    else:
        raise ValueError(f"detail '{detail}' not supported for Germany. Choose 'states' or 'counties'.")



# ---- France ------------------------------------------------------------

def _load_france_stat(data_dir: str, detail: str | None, data_type: str, year: int,
                       fill_with_values_from_other_years: bool = False,
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

def _load_world_stat(data_dir: str, detail: str | None, data_type: str, year: int,
                      fill_with_values_from_other_years: bool = False,
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

    examples = [
        dict(country="USA", detail="states", data_type="population", year=2016),
        dict(country="NL", detail="states", data_type="population", year=2020),
        dict(country="NL", detail="counties", data_type="population", year=2019),
        dict(country="Germany", detail="states", data_type="population", year=2024),
        dict(country="Germany", detail="counties", data_type="population", year=2022),
    ]

    for kwargs in examples:
        print(f"\n--- {kwargs} ---")
        try:
            polygon, target_areas, target_positions, names, centroid = load_cartogram_data(
                geojson_dir="data/geojson",
                data_dir="data/statistics",
                verbose=True,
                **kwargs,
            )
            print(f"Result: {len(names)} regions")
            print(f"First 5 names:  {names[:5]}")
            print(f"First 5 areas:  {target_areas[:5]}")
        except FileNotFoundError as e:
            print(f"[Files not present] {e}", file=sys.stderr)

