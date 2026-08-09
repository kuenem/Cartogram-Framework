import numpy as np
import json
from pathlib import Path

def nesting_depth(lst) -> int:
    if not isinstance(lst, (list, np.ndarray)):
        return 0
    if len(lst) == 0:
        return 1
    first = lst[0]
    if isinstance(first, (list, np.ndarray)):
        first = np.asarray(first)
        if first.ndim >= 2:
            return 3
        return 2
    return 2


CONFIG_FILE = Path("/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/notebooks/regions.json")


def get_region(name):
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        configs = json.load(f)

    try:
        config = configs[name]
    except KeyError:
        raise ValueError(
            f"Unknown region '{name}'. Available: {', '.join(configs)}"
        )

    return config["region"], config["level"], config["year"]