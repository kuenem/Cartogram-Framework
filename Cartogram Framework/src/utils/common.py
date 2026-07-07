import numpy as np

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