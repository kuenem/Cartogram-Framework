# Cartogram Framework

A Python framework for generating and evaluating cartograms, including Demers (square), Dorling (circle), non-contiguous, and contiguous variants.

## What's included

- `src/core/CartogramFramework.py`: main optimisation entry point via `CartogramFramework_global`
- `src/io/`: including loader for statistical data and geojsons
- `src/viz/`: containing plotting methods
- `src/tests/`: evaluation metrics for cartogram quality and recommender system code
- `notebooks/`: contains notebooks with example code especially interesting: cartograms.ipynb and recommender.ipynb. cartograms.ipynb is wrapped with the correct parameter choices from src/cartograms/XXX.py. A more raw approach should be possible through exploration.ipynb (DISCLAIMER Not used in a while)
- `data/` : Containing data for usa, netherlands, germany, india, taiwan/china and the world. GeoJson identifier has to be in src/io/geojson_parser.py in the if else loop and name column and value column name have to be present in src/io/load_data.py for new data. The data should also contain a year column. 

## Install

Create a Python environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

## Tests
Run `notebooks/cartograms.ipynb` 

## Typical workflow

1. Load a dataset with `loader()`. (Has all standard regions with the right year etc.)
2. Build adjacency, ordering, and shared-vertex relationships from the input polygons.
3. Run `CartogramFramework_global(...)` with the desired shape mode.
4. Visualize the result with `plot_polys_data(...)`.

## Notes

- The framework expects polygon geometry and statistics data in the project's `data/` folders.
- In R the script that was used for creating of the baseline is `lala.R` has to be modified for the according region.
- `shapely` is mainly used for metrics and evaluation helpers, while the optimisation path depends on `cvxpy`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`, and `geojson`.