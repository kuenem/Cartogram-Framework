# Cartogram Framework

A Python framework for generating and evaluating cartograms. It includes
Demers (squares), Dorling (circles), non-contiguous, and contiguous variants,
as well as plotting and quality-metric helpers.

## Project layout

- `src/cartograms/`: high-level drivers for the four cartogram variants
- `src/core/`: optimisation and preprocessing routines
- `src/io/`: GeoJSON and statistical-data loaders
- `src/geometry/`: polygon and adjacency utilities
- `src/tests/`: quality metrics, example data, and recommender experiments
- `src/viz/`: plotting and quality-criteria visualisations
- `data/`: GeoJSON boundaries and statistical data for supported regions
- `notebooks/`: exploratory examples and thesis experiments
- `R/`: R implementation used for baseline comparisons

## Setup

From the repository root, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project is currently run directly from the repository, so commands that
import `src` should be executed from this directory.

## Quick start

The high-level drivers accept a region configuration, a geographic level, and
an optional year. Supported region configurations are listed in
`notebooks/regions.json`.

```python
from src.cartograms import demers_cartogram
from src.utils.common import get_region

region, level, year = get_region("netherlands")
data = demers_cartogram(region, level, year)
```

Use `dorling_cartogram`, `noncontiguous_cartogram`, or
`contiguous_cartogram` for the other cartogram types. Results can be plotted
with `src.viz.plot_poly.plot_polys_data` and evaluated with
`src.tests.experiments.quality_criteria`.

## Verification

Run this lightweight smoke check from the repository root after setup:

```bash
python -m compileall -q src
python -c "import src.cartograms, src.io, src.viz; print('imports: ok')"
```

The notebooks contain the computational examples and may take substantially
longer because they solve full cartogram instances. Open
`notebooks/cartograms.ipynb` for the main end-to-end examples and
`notebooks/recommender.ipynb` for the parameter recommender experiments.

## Adding data

Each dataset needs matching polygon and statistical files under `data/`.
Region-specific GeoJSON handling lives in `src/io/geojson_parser.py`, while
statistical column mappings are handled in `src/io/load_data.py`. New data
should include a region-name column, a value column, and a year column.

The baseline comparison script is `R/cartogrampackage.R`.