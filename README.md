# Master Thesis: Cartogram Framework

This repository contains the implementations, data, experiments, and evaluation notebooks for a master's thesis on cartogram generation and quality assessment.

The repository is organised around two implementations:

- [`Cartogram Framework`](Cartogram%20Framework/README.md): a Python research framework for generating and evaluating cartograms, including contiguous, non-contiguous, Dorling, and square cartograms.
- [`cartogram-cpp`](cartogram-cpp/README.md): a C++20 implementation of a fast flow-based cartogram generator.

## Repository Layout

```text
Cartogram Framework/
  data/       Input GeoJSON and statistical data
  notebooks/  Experiments, visualisations, metrics, and recommender-system work
  R/          R-based baseline scripts and generated outputs
  src/        Python framework, geometry, I/O, metrics, and visualisation code

cartogram-cpp/
  include/    Public C++ headers
  src/        C++ implementation and command-line entry point
  sample_data/Example datasets
  tests/      Unit, stress, and fuzzing tests
```

## Python Framework

The Python implementation is the main research environment for exploring cartogram variants, comparing output quality, and experimenting with the recommender system.

### Setup

```bash
cd "Cartogram Framework"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebooks require a Jupyter environment. The main starting point is [`notebooks/cartograms.ipynb`](Cartogram%20Framework/notebooks/cartograms.ipynb). The recommender-system experiments are in [`notebooks/recommender.ipynb`](Cartogram%20Framework/notebooks/recommender.ipynb); [`notebooks/exploration.ipynb`](Cartogram%20Framework/notebooks/exploration.ipynb) contains more exploratory work.

### Typical Workflow

1. Select a matching GeoJSON file and statistics file from `Cartogram Framework/data/`.
2. Load and preprocess the data with the modules in `src/io/` and `src/core/`.
3. Generate a cartogram with the framework entry points in `src/cartograms/` and `src/core/`.
4. Visualise and evaluate the result using `src/viz/` and `src/tests/`.

The expected statistical data includes a region identifier, a value column, and a year column. New geographic datasets may also require an identifier mapping in the GeoJSON parser and a matching data-column mapping in the loader.

More Python-specific notes are available in [`Cartogram Framework/README.md`](Cartogram%20Framework/README.md).

## C++ Generator

The C++ implementation builds a `cartogram` command-line program using C++20, CMake, Conan, CGAL, and FFTW. To build and test it, follow the detailed instructions in [`cartogram-cpp/README.md`](cartogram-cpp/README.md).

After installation, a cartogram can be generated from a GeoJSON file and a CSV file containing target values:

```bash
cd cartogram-cpp
cartogram path/to/map.geojson path/to/values.csv
```

Sample input datasets are available in `cartogram-cpp/sample_data/`. Run `cartogram --help` to see all command-line options.

## Data and Reproducibility

Input geographic data is stored as GeoJSON and accompanying statistical data is stored as CSV where possible. Notebooks may write derived data, metrics, plots, or cartogram outputs alongside the experiments that produced them. Check the relevant notebook and project README for the exact dataset and parameters used by an experiment.

## License and Citation

The licensing terms for the C++ generator and its generated data are documented in [`cartogram-cpp/LICENSE`](cartogram-cpp/LICENSE). The C++ implementation is based on the flow-based method described by:

> Gastner, M. T., Seguy, V., and More, P. *Fast flow-based algorithm for creating density-equalizing map projections*. Proceedings of the National Academy of Sciences, 115(10), E2156-E2164, 2018. <https://doi.org/10.1073/pnas.0400280101>

Please also consult the individual project documentation for implementation-specific licensing and attribution requirements.