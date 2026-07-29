# Cartogram Framework

A Python framework for generating and evaluating cartograms, including Demers (square), Dorling (circle), original-shape, and contiguous variants.

## What's included

- `src/core/CartogramFramework.py`: main optimisation entry point via `CartogramFramework_global`
- `src/io/load_data.py`: dataset loader for geojson + statistics files
- `src/viz/plot_poly.py`: plotting helpers for polygons and solved cartograms
- `src/tests/cartogram_metrics.py`: evaluation metrics for cartogram quality
- `notebooks/exploration.ipynb`: interactive experimentation notebook

## Install

Create a Python environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

## Typical workflow

1. Load a dataset with `loader()`.
2. Build adjacency, ordering, and shared-vertex relationships from the input polygons.
3. Run `CartogramFramework_global(...)` with the desired shape mode.
4. Visualize the result with `plot_polys_data(...)` or `plot_polys(...)`.

## Notes

- The framework expects polygon geometry and statistics data in the project's `data/` folders.
- `shapely` is mainly used for metrics and evaluation helpers, while the optimisation path depends on `cvxpy`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`, and `geojson`.