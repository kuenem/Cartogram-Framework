import os, sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_dir)

from src.io import *

def _normalize_name(name: str) -> str:
    """Whitespace/case/punctuation/accent-insensitive key for fallback
    matching, e.g. ' Abruzzo ' vs 'Abruzzo', 'New Hampshire' vs
    'NewHampshire', or 'Åland' vs 'Aland'/'ALAND'. Diacritics are stripped
    via NFKD decomposition, and non-alphanumeric separators are removed so
    spacing/underscore/hyphen differences collapse to the same key."""
    normalized = str(name).casefold()
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(
        c for c in decomposed
        if not unicodedata.combining(c) and c.isalnum()
    )


def load_external_cartogram(cartogram_geojson_path, region, level, year=None,
                             name_column=None, value_column=None, fuzzy_fallback=True,
                             cartogram_name_property=None):
    org_data, *_ = loader(region, level, year=year,
                           name_column=name_column, value_column=value_column,
                           fuzzy_fallback=fuzzy_fallback)

    # Rescale raw statistic -> actual target area, conserving total original area.
    # (Same normalization the Framework applies internally before solving —
    # loader() only hands back the raw CSV value.)
    total_orig_area = sum(rec["area"] for rec in org_data.values())
    total_stat = sum(rec["target_area"] for rec in org_data.values())
    scale = total_orig_area / total_stat
    for rec in org_data.values():
        rec["target_area"] *= scale

    cart_data, *_ = get_polygon_data(cartogram_geojson_path, {}, name_property=cartogram_name_property)

    total_cart_area = sum(rec["area"] for rec in cart_data.values())
    linear_scale = (total_orig_area / total_cart_area) ** 0.5
    for rec in cart_data.values():
        rec["polygon"] = rec["polygon"] * linear_scale
        rec["area"] *= linear_scale ** 2
        rec["centroid"] = rec["centroid"] * linear_scale

    fuzzy_cart = {_normalize_name(k): k for k in cart_data} if fuzzy_fallback else {}

    merged, unmatched = {}, []
    for name, rec in org_data.items():
        merged[name] = dict(rec)
        merged[name]["original_area"] = rec["area"]  # success_rate needs this key

        if name in cart_data:
            src = cart_data[name]
        elif fuzzy_fallback and _normalize_name(name) in fuzzy_cart:
            src = cart_data[fuzzy_cart[_normalize_name(name)]]
        else:
            unmatched.append(name)
            continue

        merged[name]["new_polygon"] = src["polygon"]
        merged[name]["new_area"] = src["area"]

    if unmatched:
        print(f"{len(unmatched)} region(s) missing from external cartogram: {unmatched}")
        for name in unmatched:
            del merged[name]

    return merged, org_data