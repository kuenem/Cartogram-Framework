# Cartogram Framework — Comparative Testing Suite Design

This document specifies a testing suite for evaluating the unified cartogram
framework against reference implementations (R `cartogram` package: Dorling,
non-contiguous, Dougenik; `mgastner/cartogram-cpp`: flow-based contiguous).
It covers three axes: **runtime**, **quality criteria**, and **robustness**
(unintended overlaps / gaps), plus dataset selection and statistical
comparison methodology.

---

## 1. Benchmark Datasets

### 1.1 Primary corpus — Miaji, Singhania, Goh, Le & Gastner (2026)

Miaji et al., "Topology-Preserving Line Densification for Creating Contiguous
Cartograms" (*Cartography and Geographic Information Science*, 2026,
doi:10.1080/15230406.2026.2625329), assembled 32 real-world map/data pairs
specifically to stress-test contiguous cartogram algorithms, deliberately
choosing cases with numerous islands and holes, identical geometries paired
with different datasets, both very large and very small regions, and complex
borders. This is close to an ideal match for what you need, and it was
built and open-sourced for exactly this purpose: comparative algorithm
evaluation, not just illustration.

Two things make it unusually reusable:

- **It's openly available**: figshare doi:10.6084/m9.figshare.28478408
  (geometries + data as used in the paper).
- **It ships a built-in difficulty stratification.** For each region they
  define a target density `ρ_i = A_i^target / A_i^actual`, and a per-map
  **density disparity** `D = max_i(ρ_i) / min_i(ρ_i)`. They split the 32
  maps into four groups: `D<100`, `100≤D<1000`, `1000≤D<10000`, `D≥10000`.
  Reuse this stratification directly — it is exactly the axis your
  contiguous-mode debugging cares about (vertex explosion, spikes, and
  overlaps all get worse as disparity grows).

Cases directly relevant to your archipelago/small-area findings are already
in the corpus: **Bahamas, Malaysia, Indonesia, Singapore** (islands/holes/
enclaves), plus **World by country** and **USA by county** at the extreme
end (D > 10⁵). These let you reproduce and extend your own finding about
flow-based area deficits for archipelago nations on a dataset a third party
already vetted for this exact purpose, rather than defending the choice
yourself.

### 1.2 Secondary corpus — Alam/Kobourov/Schneider/Veeramoni (2013) + Nusrat/Alam/Kobourov (2016)

USA, Germany, and Italy, each at 2–4 schematization levels, with GDP and
population (2010). Explicitly justified by the authors as guarding against
bias from relying on only one or two datasets, and picked so region counts,
dual-graph edge counts, and polygon complexity vary across the three
countries (Alam, Kobourov, Schneider & Veeramoni, *An Experimental Study of
Algorithms for Cartogram Generation*). Nusrat, Alam & Kobourov reused the
identical maps in *Evaluating Cartogram Effectiveness* (TVCG 2018) for the
same reason.

Value for you: published ε (cartographic error), τ (adjacency error), θ
(angular orientation error), and δ (Hamming distance) values exist for
several algorithms (diffusion, circular-arc, rectangular, T-shape) on this
exact data. That gives you an external sanity check — if your R/`cartogram-cpp`
baselines on this data land far from the published numbers, something in
your pipeline (not the algorithm) is likely off.

Italy's dataset also contains genuine island regions (Sicily, Sardinia),
so it doubles as a small non-synthetic contiguity edge case even outside
the Miaji corpus.

### 1.3 Your own loaders (USA, NL, DE, FR, World)

Keep these as a third, "production realism" tier — they're what your
framework will actually be used on, they already have working ingestion,
and they're where your CBS/Destatis-specific parsing quirks live. Use them
for full end-to-end validation and for the archipelago/small-area
follow-up you flagged, but not as the primary comparability benchmark,
since no other paper's numbers are directly comparable against your
specific CBS/GENESIS extracts.

---

## 2. Quality-Criteria Catalog

Notation follows your thesis (§3.2–3.3): region `v`/`i`, obtained area
`o(v)`/`A_i^cart`, target area `w(v)`/`A_i^target`, original area `a(v)`,
adjacency set `T`, width proxy `w_ij = ½(√A*_i + √A*_j)`.

### A. Statistical distortion (area accuracy)

| Metric | Formula | Source |
|---|---|---|
| Average cartographic error, ε | ε = (1/N) Σ_v \|o(v)−w(v)\| / max{o(v),w(v)} | Alam, Kobourov & Veeramoni 2015 (Comput. Graph. Forum 34(3)) — **already in your `cartogram_metrics.py`** |
| Maximum relative area error, E_max | E_max = max_i \|A_i^cart/A_i^target − 1\| | Gastner, Seguy & More 2018 (PNAS); adopted by Miaji et al. 2026 as the stricter, size-agnostic variant of ε — **worth adding alongside ε**, since it penalizes small regions equally to large ones and is what the Miaji corpus reports |
| Success rate | \|o(v)−a(v)\| / \|w(v)−a(v)\| | Kämper, Kobourov & Nöllenburg 2013, via Alam et al.'s *Experimental Study*. Only meaningful for deformation-style algorithms (diffusion, circular-arc, your soft-contiguous mode) — not for combinatorial ones (T-shape, rectangular) |

### B. Topological accuracy (adjacency)

| Metric | Formula | Source |
|---|---|---|
| Adjacency/topology error, τ | τ = 1 − \|E_c∩E_m\| / \|E_c∪E_m\| | Alam et al. 2015; Nusrat, Alam & Kobourov 2016 — **already implemented**. Note: this metric is uninformative for your hard-contiguous mode (τ is always 0 by construction), so report it mainly for Dorling/non-contiguous/soft-contiguous comparisons |

### C. Orientation & shape distortion

| Metric | Formula | Source |
|---|---|---|
| Angular orientation error, θ | avg. change in slope of the line between centroids of adjacent-pair regions | Heilmann, Keim, Panse & Sips 2004 — **already implemented** |
| Orthogonal orientation error, ρ | fraction of region pairs whose N–S/E–W ordering flips | Alam et al., *Experimental Study* — cheaper approximation of θ; optional, use if you want a metric less sensitive to small-angle noise |
| Hamming distance / symmetric difference, δ | normalize both polygons to unit area, superimpose optimally over translation, δ = area not common to both | Alam et al. 2015 — **already implemented**. Same quantity is also used by Sun 2020 and Cano et al. 2015 under the name "symmetric difference" |
| Turning-angle distortion, Ψ / modified Ψ_M | turning-function comparison (perimeter-normalized curvature) | Arkin, Chew, Huttenlocher, Kedem & Mitchell 1990, imported into cartograms by Alam et al.'s *Experimental Study*. Ψ is rotation-invariant, Ψ_M is not. Their own comparison found δ tracks intuitive shape distortion better than Ψ/Ψ_M — treat these as optional cross-checks, not primary metrics |
| Fréchet distance | min "leash length" needed to walk both boundary curves in step | Sun 2020a; adopted by Miaji et al. 2026. Sensitive to point *order*, complements δ (which is order-agnostic) |
| Hausdorff distance | max over one boundary of the min distance to the other boundary | Okabe & Miller 1996; Sun 2020a; Miaji et al. 2026. Captures worst-case local deviation that an area-based metric like δ can hide |

### D. Topological integrity (validity) — your "overlap" metric

| Metric | Formula | Source |
|---|---|---|
| Self-intersections | count of edge pairs *within* one polygon that cross | Sun 2020a; Miaji et al. 2026 |
| Overlapping intersections | count of edge pairs *between different* polygons that cross | Sun 2020a; Miaji et al. 2026 — this is the standard literature name for what you're calling "overlap" |
| Overlap area (severity), Ω_area | Ω_area = Σ_{i≠j} Area(R_i ∩ R_j) / Σ_i A_i^cart | **Proposed extension** (not in the cited papers, which report only counts). A count of 40 hairline-thin overlaps is a very different failure from a count of 4 overlaps covering 20% of the map; report both. |

Nusrat/Alam/Kobourov's adjacency error τ is *not* a substitute for this —
it measures whether the right pairs touch, not whether any pair invalidly
overlaps. Sun (2020a) and Miaji et al. (2026) introduced topological
integrity precisely because τ goes to zero for contiguous cartograms
regardless of how badly they overlap.

### E. Gap / adjacency-fidelity — no ready-made literature metric

Nobody has published a direct polygon-boundary "gap" metric for
contiguous cartograms; the closest published analogue is Nickel, Sondag,
Meulemans et al. (2019, *Computing Stable Demers Cartograms*), who quantify
Demers-cartogram quality by the distance between the squares of regions
that should be adjacent — i.e., an inter-region boundary distance, just
for squares rather than arbitrary polygons. §4 below adapts that idea
to polygons and to your own `gij` notation, since your framework already
encodes "0 gap for adjacent pairs, ε gap for non-adjacent pairs" as a
constraint (Eq. 3.12) — the gap metric is just checking whether the
constraint's intent held after post-processing.

### F. Complexity & runtime

| Metric | Notes |
|---|---|
| Polygonal complexity | max/mean vertices per region after post-processing. Relevant since contiguous locking and gap-closing can both inflate vertex counts (cf. Alam, Biedl, Felsner, Kaufmann, Kobourov & Ueckerdt's 8-corner bound as a reference point for combinatorial methods) |
| Runtime | wall-clock time to convergence — see §5 for a fair comparison protocol across iterative vs. non-iterative algorithms |

---

## 3. Overlap Metric — Implementation

```python
from shapely.geometry import Polygon
from shapely.validation import make_valid
from itertools import combinations

def topological_integrity(regions: dict[str, Polygon]) -> dict:
    """
    regions: {region_id: shapely Polygon/MultiPolygon}, post-processed
             cartogram output (already through your gap-closing /
             overlap-resolution steps if any).

    Returns counts (Sun 2020a / Miaji et al. 2026 style) AND area severity
    (proposed extension).
    """
    fixed = {rid: make_valid(poly) if not poly.is_valid else poly
             for rid, poly in regions.items()}

    self_intersections = sum(1 for poly in regions.values() if not poly.is_valid)

    overlap_count = 0
    overlap_area = 0.0
    total_area = sum(p.area for p in fixed.values())

    for (id_a, poly_a), (id_b, poly_b) in combinations(fixed.items(), 2):
        if not poly_a.intersects(poly_b):
            continue
        inter = poly_a.intersection(poly_b)
        if inter.area > 1e-9:          # ignore boundary-only touching
            overlap_count += 1
            overlap_area += inter.area

    return {
        "self_intersections": self_intersections,
        "overlap_count": overlap_count,
        "overlap_area_fraction": overlap_area / total_area if total_area else 0.0,
    }
```

`make_valid` (Shapely ≥ 2.0 / GEOS ≥ 3.10) is the robust way to handle the
self-intersecting output your homothetic vertex transform can produce
before you try to compute intersections — this is directly relevant to
the `resolve_polygon_overlaps` regression you flagged, since silently
skipping invalid polygons instead of repairing them would undercount
overlaps.

---

## 4. Gap Metric — Implementation

Reframe your own constraint (Eq. 3.12: `gij = 0` for adjacent pairs, `ε`
otherwise) as a post-hoc check: for every pair the framework *intended* to
keep touching, measure how far apart the output polygons actually ended up.

```python
def gap_error(regions: dict[str, Polygon], adjacency_T: set[tuple],
              width_proxy: dict[tuple, float], gap_tol_fraction: float = 0.01):
    """
    adjacency_T: your T set — pairs (i, j) that should be adjacent (gij = 0)
    width_proxy: precomputed w_ij = 0.5*(sqrt(A*_i)+sqrt(A*_j)) per pair,
                 reusing the same normalization your topology constraints
                 already use (Eq. 3.12) instead of an arbitrary length scale
    gap_tol_fraction: flag a pair as a real gap if dist > gap_tol_fraction * w_ij
    """
    violations = []
    for (i, j) in adjacency_T:
        dist = regions[i].distance(regions[j])       # 0 if touching/overlapping
        tol = gap_tol_fraction * width_proxy[(i, j)]
        if dist > tol:
            violations.append((i, j, dist, dist / width_proxy[(i, j)]))

    n_pairs = len(adjacency_T)
    gap_count = len(violations)
    gap_mean_normalized = (sum(v[3] for v in violations) / n_pairs) if n_pairs else 0.0

    return {
        "gap_count": gap_count,
        "gap_rate": gap_count / n_pairs if n_pairs else 0.0,
        "gap_mean_normalized": gap_mean_normalized,
        "violations": violations,   # keep for per-map debugging (e.g. India-style cases)
    }
```

This gives you exactly the "hard spike/validity disqualifier" the Pareto
search was missing: any parameter combination producing `gap_count > 0`
or `overlap_count > 0` above a tolerance can be hard-disqualified before
the search even looks at ε/θ/δ, rather than only being penalized softly.

---

## 5. Stress-Test Parameter Matrix (overlap/gap specific)

| Parameter swept | Range | Hypothesis to test |
|---|---|---|
| Density disparity (data selection) | Miaji groups 1→4 | Overlap count rises with disparity for hard-contiguous mode; use this as your primary difficulty axis rather than inventing a new one |
| `shape`: `"contiguous"` vs `"contiguous2"` | hard vs soft closure | Hard mode trades overlap risk for infeasibility risk; soft mode trades it for gap risk — confirm both failure modes show up and quantify the trade-off with §3/§4 metrics simultaneously |
| `t_max` | tight → loose | Loosening reduces gap_count but increases overlap_count/overlap_area (your India-case pattern) |
| `contiguous_area_ratio_cap` | tight → loose | Tight cap → more unlocked shared vertices → higher gap_count; loose cap → small regions crushed → possibly higher overlap_area on the crushed region's other borders |
| `lambda_vertex_distance`, `lambda_repulsion` | off vs on, low vs high | Should reduce overlap_count/overlap_area with only a small ε/δ cost; test whether they introduce new gaps as a side effect |
| Archipelago subset (Bahamas, Indonesia, Malaysia, Singapore from Miaji corpus + Philippines from your own data) | fixed set | Directly reproduces and extends your flow-based-method area-deficit finding under your own framework's overlap/gap metrics |

---

## 6. Runtime Testing Methodology

Naive wall-clock comparison across iterative vs. non-iterative algorithms
is misleading (an iterative method stopped early looks "fast" but may be
far from converged). Miaji et al. (2026) address this by:

- Running every iterative algorithm to its own convergence criterion
  (e.g., stop when max relative area error < 1%) rather than a fixed
  iteration count, **or**
- Reporting *time-to-reach-a-fixed-error-level*, letting a slower-per-
  iteration but faster-converging method "catch up" to a faster-but-
  weaker one, then comparing times at matched quality.

Adopt the second (iso-quality) approach for comparing your QP-based
solves against R `cartogram`'s Dougenik/Dorling and `cartogram-cpp`'s
flow method: report time-to-ε<0.01 rather than a single wall-clock
number, run on identical hardware, and disable parallelism uniformly
across all methods being compared (Miaji et al. found enabling it
inconsistently distorted their own comparison).

---

## 7. Statistical Comparison Methodology

- **Paired design.** Every algorithm runs on the same map/dataset, so
  comparisons are paired, not independent samples.
- **Two algorithms:** Wilcoxon signed-rank test (non-parametric — cartogram
  error/overlap distributions are rarely normal on small map sets), as
  used by Duncan & Gastner (2022/2024) for tool comparisons; report the
  confidence interval for the pseudomedian difference (Bauer 1972) rather
  than only a p-value.
- **More than two algorithms:** Friedman test (non-parametric analogue of
  the repeated-measures ANOVA that Nusrat et al. 2016 used) followed by
  pairwise Wilcoxon with Holm–Bonferroni correction for post-hoc
  comparisons.
- **Stratify by difficulty group before pooling.** Miaji et al.'s own
  results show algorithm ranking flips between low- and high-disparity
  groups (F4Carto wins on Groups 1–2, 5FCarto/BFB dominate 3–4); a single
  pooled average across all 32 maps would hide exactly the failure modes
  you're trying to characterize.

---

## 8. Suite Architecture

```
testing_suite/
  datasets/
    miaji2026_manifest.yaml       # figshare doi + per-map metadata (density disparity, group)
    legacy_akv_manifest.yaml      # USA/DE/IT GDP+population, schematization levels
    own_loaders/                  # your existing load_data.py entries (USA, NL, DE, FR, World)
  adapters/
    framework_adapter.py          # your CartogramFramework_4 / cartogram_min.py
    r_cartogram_adapter.py        # subprocess/rpy2 wrapper: dorling, noncontig, cartogram_cont
    cartogram_cpp_adapter.py      # subprocess wrapper around mgastner/cartogram-cpp binary
  metrics/
    statistical.py                # ε, E_max, success rate
    geographic.py                 # θ, ρ, δ, Ψ, Ψ_M, Fréchet, Hausdorff (extend cartogram_metrics.py)
    topological.py                # τ, overlap_count/area (§3), gap_count/rate (§4)  <- new module
    complexity.py                 # vertex counts, runtime (iso-quality, §6)
  runners/
    run_suite.py                  # cartesian product of map × algorithm × parameter-set
    stress_matrix.py              # generates the §5 sweep configs
  analysis/
    aggregate.py                  # group by density-disparity tier
    stats_tests.py                # Wilcoxon / Friedman + post-hoc (§7)
    report.py                     # tables/figures for Thesis Ch. 4
  results/                        # raw + aggregated output (json/csv), one row per run
```

`resolve_polygon_overlaps` and `postprocess_gaps` should sit *before* the
metrics layer in the pipeline, and the suite should record metrics both
pre- and post-postprocessing — that split is what will let you show,
concretely, how much of your final overlap/gap numbers are solved by the
optimization itself versus patched afterward, which is exactly the kind
of result Chapter 4 currently lacks.

---

## References

- Alam, M. J., Kobourov, S. G., & Veeramoni, S. (2015). Quantitative measures for cartogram generation techniques. *Computer Graphics Forum*, 34(3), 351–360.
- Alam, M. J., Kobourov, S., Schneider, M., & Veeramoni, S. *An Experimental Study of Algorithms for Cartogram Generation*. University of Arizona technical report.
- Alam, M. J., Biedl, T., Felsner, S., Kaufmann, M., Kobourov, S. G., & Ueckerdt, T. (2013). Computing cartograms with optimal complexity. *Discrete & Computational Geometry*, 50(3), 784–810.
- Arkin, E. M., Chew, L. P., Huttenlocher, D. P., Kedem, K., & Mitchell, J. S. (1990). An efficiently computable metric for comparing polygonal shapes. *SODA '90*.
- Duncan, I. K., & Gastner, M. T. (2022/2024). Comparative evaluation of the web-based contiguous cartogram generation tool go-cart.io. *PLOS ONE*, 19(5), e0298192.
- Gastner, M. T., Seguy, V., & More, P. (2018). Fast flow-based algorithm for creating density-equalizing map projections. *PNAS*, 115(10), E2156–E2164.
- Heilmann, R., Keim, D. A., Panse, C., & Sips, M. (2004). RecMap: Rectangular map approximations. *IEEE InfoVis 2004*.
- Kämper, J.-H., Kobourov, S. G., & Nöllenburg, M. (2013). Circular-arc cartograms. *IEEE PacificVis 2013*.
- Miaji, N. Z., Singhania, A., Goh, M. E., Le, C., & Gastner, M. T. (2026). Topology-preserving line densification for creating contiguous cartograms. *Cartography and Geographic Information Science*. doi:10.1080/15230406.2026.2625329. Data: doi:10.6084/m9.figshare.28478408.
- Nickel, S., Sondag, M., Meulemans, W., Chimani, M., Kobourov, S., Peltonen, J., & Nöllenburg, M. (2019). Computing stable Demers cartograms. *Graph Drawing and Network Visualization (GD 2019)*, 46–60.
- Nusrat, S., Alam, M. J., & Kobourov, S. (2016/2018). Evaluating cartogram effectiveness. *IEEE TVCG*, 24(2), 1077–1090.
- Okabe, A., & Miller, H. J. (1996). Exact computational methods for calculating distances between objects in a cartographic database. *Cartography and GIS*, 23(4), 180–195.
- Sun, S. (2020a). Applying forces to generate cartograms: A fast and flexible transformation framework. *Cartography and Geographic Information Science*, 47(5), 381–399.
