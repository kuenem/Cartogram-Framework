import os
import sys
import numpy as np
import pytest

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.recommender_system import *
from src.core import *
from src.tests import *

def test_default_sliders_reproduce_base_preset():
    for ctype in BASE_PRESETS:
        params, trace = recommend_params(cartogram_type=ctype)  # all sliders = 0.5
        assert params == BASE_PRESETS[ctype]
        assert trace["changes"] == [] or all(c[0] == "bound_clip" for c in trace["changes"])

def test_knob_extremes_hit_declared_bounds():
    # knob=0 -> low, knob=1 -> high, for every (slider, param) pair, per type
    for slider, param_ranges in SLIDER_PARAM_RANGES.items():
        for param, (low, high) in param_ranges.items():
            for ctype, preset in BASE_PRESETS.items():
                if param not in preset:
                    continue
                params0, _ = recommend_params(cartogram_type=ctype, **{slider: 0.0})
                params1, _ = recommend_params(cartogram_type=ctype, **{slider: 1.0})
                lo, hi = PARAM_BOUNDS.get(param, (-np.inf, np.inf))
                assert params0[param] == pytest.approx(np.clip(low, lo, hi))
                assert params1[param] == pytest.approx(np.clip(high, lo, hi))

def test_monotonicity_per_param():
    # value(knob) should be monotone between the declared low/high (catches sign errors)
    for slider, param_ranges in SLIDER_PARAM_RANGES.items():
        for param in param_ranges:
            for ctype in BASE_PRESETS:
                if param not in BASE_PRESETS[ctype]:
                    continue
                vals = [recommend_params(cartogram_type=ctype, **{slider: k})[0][param]
                        for k in np.linspace(0, 1, 11)]
                assert all(a <= b for a, b in zip(vals, vals[1:])) or all(a >= b for a, b in zip(vals, vals[1:]))

def test_skip_logic_for_inapplicable_params():
    # e.g. lambda_contiguous only exists for "contiguous"
    _, trace = recommend_params(cartogram_type="dorling", seamlessness=1.0)
    assert any(p == "lambda_contiguous" for _, p, _ in trace["skipped"])

def test_no_two_sliders_silently_collide():
    # guards against a future edit where two sliders touch the same param
    # without anyone noticing the deltas now add instead of overriding
    seen = {}
    for slider, param_ranges in SLIDER_PARAM_RANGES.items():
        for param in param_ranges:
            seen.setdefault(param, []).append(slider)
    collisions = {p: s for p, s in seen.items() if len(s) > 1}
    assert collisions == {}, f"unexpected shared params: {collisions}"