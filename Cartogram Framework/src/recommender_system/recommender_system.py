from __future__ import annotations

import copy
from typing import Optional

import numpy as np


import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_dir)

from src.io import *               
from src.geometry import *                
from src.utils import *                    
from src.core import *                         
from src.core.preprocessing_data import preprocess, auto_area_scale



BASE_PRESETS: dict[str, dict] = {
    "contiguous": dict(
        shape="contiguous2",
        lambda_shape=0.5,
        lambda_area=1.0,
        lambda_center=0.0,
        lambda_topology=0.0,
        lambda_contiguous=30.0,
        soft_mean_scale=True,
        lambda_mean_scale=1e5,
        t_min=0.01,
        t_max=8.0,          
        lambda_vertex_distance=3.0,
        vertex_distance_margin=1.5,
        lambda_repulsion=0.0,
        contiguous_area_ratio_cap=np.inf,
        contiguous_use_topology=False,
        gamma=1.0,
        beta=1.0,
        alpha=1.0,
        epsilon=1e-2,
        b=1e-2,
        postprocess_contiguous=False,
        postprocess_snap_method="mean",
        postprocess_disputed_pixels=True,
        postprocess_gaps=True,
        postprocess_raster_resolution=800,
        postprocess_raster_area_bias=0.05,
        postprocess_equalize_areas=True,
        postprocess_equalize_max_passes=30,
        postprocess_equalize_tolerance=0.02,
    ),
    "contiguous-without-postprocess": dict(
            shape="contiguous2",
            lambda_shape=0.5,
            lambda_area=1.0,
            lambda_center=0.0,
            lambda_topology=0.0,
            lambda_contiguous=30.0,
            soft_mean_scale=True,
            lambda_mean_scale=1e5,
            t_min=0.01,
            t_max=8.0,          
            lambda_vertex_distance=3.0,
            vertex_distance_margin=1.5,
            lambda_repulsion=0.0,
            contiguous_area_ratio_cap=np.inf,
            contiguous_use_topology=False,
            gamma=1.0,
            beta=1.0,
            alpha=1.0,
            epsilon=1e-2,
            b=1e-2,
            postprocess_snap_method="mean",
            postprocess_disputed_pixels=False,
            postprocess_gaps=False,
            postprocess_raster_resolution=800,
            postprocess_raster_area_bias=0.05,
            postprocess_equalize_areas=False,
            postprocess_equalize_max_passes=30,
            postprocess_equalize_tolerance=0.02,
        ),
    "demers": dict(
        shape="square",
        lambda_shape=0.1,
        lambda_area=1.0,
        lambda_center=0.0,
        lambda_topology=1.0,
        gamma=0.0,
        beta=1.0,
        alpha=1.0,
        epsilon=1e-2,
        b=1e-2,
        t_min=0.01,
        shape_deformation=0.0,
    ),
    "dorling": dict(
        shape="circle",
        lambda_shape=0.1,
        lambda_area=1.0,
        lambda_center=0.5,
        lambda_topology=1.0,
        gamma=0.0,
        beta=1.0,
        alpha=1.0,
        epsilon=1e-2,
        b=1e-2,
        t_min=0.01,
        t_max=5.0,
        shape_deformation=0.0,
        lambda_repulsion=1.0,
        repulsion_only_adjacent=False,
        repulsion_margin_mode="circle_radius",
        postprocess_declump_circles=True,
    ),
    "noncontiguous": dict(
        shape="original",
        lambda_shape=0.0,
        lambda_area=1.0,
        lambda_center=0.0,
        lambda_topology=0.0,
        gamma=1.0,
        beta=1.0,
        t_min=1e-8,
        t_max=10000.0,
        area_scale="auto",
        area_scale_cap=1.0,
    ),
}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "lambda_shape": (0.0, 5.0),
    "lambda_area": (0.1, 5.0),
    "lambda_center": (0.0, 3.0),
    "lambda_topology": (0.0, 3.0),
    "lambda_contiguous": (5.0, 150.0),
    "lambda_vertex_distance": (0.0, 20.0),
    "lambda_mean_scale": (1e3, 3e5),
    "t_min": (1e-4, 0.5),
    "t_max": (2.0, 30.0),
    "postprocess_raster_area_bias": (0.0, 0.3),
    "postprocess_equalize_tolerance": (0.005, 0.1),
}


SLIDER_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "size_accuracy": {
        "lambda_area": (0.3, 3.0),
        "lambda_mean_scale": (3e4, 2.5e5),
        "postprocess_equalize_tolerance": (0.05, 0.005),   # 1.0 = tighter
        "postprocess_raster_area_bias": (0.0, 0.15),
    },
    "shape_preservation": {
        "lambda_shape": (0.0, 3.0),
    },
    "position_accuracy": {
        "lambda_center": (0.0, 2.0),
    },
    "neighborhoods_kept": {
        "lambda_topology": (0.0, 3.0),
    },
    "seamlessness": {

        "lambda_contiguous": (10.0, 100.0),
    },
    "robustness": {
        "lambda_vertex_distance": (8.0, 0.0),   
        "t_max": (5.0, 15.0),                   
    },
}

SLIDER_NAMES = list(SLIDER_PARAM_RANGES)


# ---------------------------------------------------------------------------
# 3. Recommendation
# ---------------------------------------------------------------------------

def _interp_delta(knob: float, base: float, low: float, high: float) -> float:
    """Delta-from-base at slider position `knob` (piecewise-linear through
    (0, low), (0.5, base), (1, high)). Zero at knob=0.5 by construction."""
    knob = float(np.clip(knob, 0.0, 1.0))
    if knob <= 0.5:
        value = low + (knob / 0.5) * (base - low)
    else:
        value = base + ((knob - 0.5) / 0.5) * (high - base)
    return value - base


def recommend_params(
    cartogram_type: str = "contiguous",
    size_accuracy: float = 0.5,
    shape_preservation: float = 0.5,
    position_accuracy: float = 0.5,
    neighborhoods_kept: float = 0.5,
    seamlessness: float = 0.5,
    robustness: float = 0.5,
) -> tuple[dict, dict]:
    """
    Map six 0-1 sliders onto a parameter dict for `cartogram_type`.

    All sliders default to 0.5 (= this thesis's tuned base preset,
    unchanged). Values outside [0, 1] are clipped.

    Returns
    -------
    (params, trace)
        `params` is ready to splat into the matching driver call (see
        `build_cartogram`). `trace` records every slider's value and
        exactly which parameters it moved, for `explain()`.
    """
    if cartogram_type not in BASE_PRESETS:
        raise ValueError(
            f"Unknown cartogram_type {cartogram_type!r}; "
            f"expected one of {list(BASE_PRESETS)}"
        )

    sliders = dict(
        size_accuracy=size_accuracy,
        shape_preservation=shape_preservation,
        position_accuracy=position_accuracy,
        neighborhoods_kept=neighborhoods_kept,
        seamlessness=seamlessness,
        robustness=robustness,
    )

    params = copy.deepcopy(BASE_PRESETS[cartogram_type])
    trace = {
        "cartogram_type": cartogram_type,
        "sliders": {k: float(np.clip(v, 0.0, 1.0)) for k, v in sliders.items()},
        "changes": [],   # (slider, param, before, after)
        "skipped": [],   # (slider, param, reason) -- param doesn't exist for this type
    }

    # Accumulate every slider's delta-from-base per parameter, then apply once.
    deltas: dict[str, float] = {}
    for slider_name, knob in sliders.items():
        for param, (low, high) in SLIDER_PARAM_RANGES[slider_name].items():
            if param not in params:
                trace["skipped"].append(
                    (slider_name, param, f"{cartogram_type} has no {param}")
                )
                continue
            base_value = BASE_PRESETS[cartogram_type][param]
            delta = _interp_delta(knob, base_value, low, high)
            if delta == 0.0:
                continue
            deltas[param] = deltas.get(param, 0.0) + delta
            trace["changes"].append((slider_name, param, "delta", delta))

    for param, total_delta in deltas.items():
        before = params[param]
        after = before + total_delta
        if param in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[param]
            clipped = float(np.clip(after, lo, hi))
            if clipped != after:
                trace["changes"].append(("bound_clip", param, after, clipped))
            after = clipped
        params[param] = after
        trace["changes"].append(("__final__", param, before, after))

    trace["final_params"] = copy.deepcopy(params)
    return params, trace


def explain(trace: dict) -> str:
    """Human-readable printout of a recommend_params() trace."""
    lines = [f"Cartogram type: {trace['cartogram_type']}", "Sliders:"]
    for name in SLIDER_NAMES:
        lines.append(f"  {name}: {trace['sliders'][name]:.2f}")

    finals = {p: (b, a) for tag, p, b, a in trace["changes"] if tag == "__final__"}
    if finals:
        lines.append("Resulting parameter changes:")
        for param, (before, after) in finals.items():
            if isinstance(before, float):
                lines.append(f"  {param}: {before:.4g} -> {after:.4g}")
            else:
                lines.append(f"  {param}: {before} -> {after}")
    else:
        lines.append("No parameters moved from the base preset (all sliders at 0.5, or no-ops for this type).")

    if trace["skipped"]:
        lines.append("Ignored (parameter doesn't apply to this cartogram type):")
        for slider_name, param, reason in trace["skipped"]:
            lines.append(f"  {slider_name} -> {param}: {reason}")

    return "\n".join(lines)


def build_cartogram(
    region,
    level,
    year: str = "2016",
    cartogram_type: str = "contiguous",
    size_accuracy: float = 0.5,
    shape_preservation: float = 0.5,
    position_accuracy: float = 0.5,
    neighborhoods_kept: float = 0.5,
    seamlessness: float = 0.5,
    robustness: float = 0.5,
    overrides: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Recommend parameters from the six sliders above and immediately solve.

    Returns (data, params, trace) — `data` is the same dict shape your
    existing contiguous_cartogram() / dorling_cartogram() / etc. driver
    functions return.
    """

    params, trace = recommend_params(
        cartogram_type=cartogram_type,
        size_accuracy=size_accuracy,
        shape_preservation=shape_preservation,
        position_accuracy=position_accuracy,
        neighborhoods_kept=neighborhoods_kept,
        seamlessness=seamlessness,
        robustness=robustness,
    )

    if overrides:
        params.update(overrides)
        trace["manual_overrides"] = overrides

    if verbose:
        print(explain(trace))
        print()

    data, polygons, target_areas, target_positions, names, centroid = loader(
        region, level, year=year
    )
    region_names = [k for k in data if k != "__meta__"]
    polygons = [np.asarray(data[name]["polygon"]) for name in region_names]

    if cartogram_type == "noncontiguous":
        area_scale = params.pop("area_scale", "auto")
        area_scale_cap = params.pop("area_scale_cap", 1.0)
        if area_scale == "auto":
            _, _, normalised_target_areas, _, _, _ = preprocess(data, None, 1.0, None)
            resolved_area_scale = auto_area_scale(data, normalised_target_areas, cap=area_scale_cap)
            if verbose:
                print(
                    f"  build_cartogram: auto area_scale = {resolved_area_scale:.4f} "
                    f"(worst-case region growth capped at {area_scale_cap}x its original area)"
                )
        else:
            resolved_area_scale = float(area_scale)

        new_data = CartogramFramework_global(
            data=data,
            target_centers=None,
            neighboring_pairs=None,
            horizontal_pairs=None,
            vertical_pairs=None,
            shared_vertices_of_neighbors=None,
            area_scale=resolved_area_scale,
            **params,
        )
    else:
        horizontal_pairs, vertical_pairs = disjoint_pairs_horizontal_and_vertical_center(polygons)
        new_data = CartogramFramework_global(
            data=data,
            fixed_points=None,
            target_centers=target_positions,
            horizontal_pairs=horizontal_pairs,
            vertical_pairs=vertical_pairs,
            neighboring_pairs=neighbouring_pairs(polygons),
            shared_vertices_of_neighbors=shared_vertices_of_neighbors(polygons),
            **params,
        )

    return new_data, params, trace
