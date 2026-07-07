import numpy as np

def polygon_areanp(pts: np.ndarray) -> float:
    """Signed shoelace area (absolute value = unsigned area)."""
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def make_circle(center: np.ndarray, area: float, n: int = 64) -> np.ndarray:
    r = np.sqrt(area / np.pi)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + r * np.column_stack([np.cos(angles), np.sin(angles)])


def make_square(center: np.ndarray, area: float, n: int = 64) -> np.ndarray:
    half = np.sqrt(area) / 2
    # n points distributed around the perimeter
    corners = np.array([
        [ half,  half],
        [-half,  half],
        [-half, -half],
        [ half, -half],
    ])
    pts_per_side = max(n // 4, 1)
    pts = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        for t in np.linspace(0, 1, pts_per_side, endpoint=False):
            pts.append(a + t * (b - a))
    return center + np.array(pts)