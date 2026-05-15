import heapq
import numpy as np

def triangle_area(a, b, c):
    return abs(np.cross(b - a, c - a)) / 2

def visvalingam(points, target_n):
    pts = [np.array(p) for p in points]
    n = len(pts)

    if target_n >= n:
        return points

    areas = []
    for i in range(1, n-1):
        area = triangle_area(pts[i-1], pts[i], pts[i+1])
        heapq.heappush(areas, (area, i))

    removed = set()
    while n - len(removed) > target_n:
        area, i = heapq.heappop(areas)
        if i in removed:
            continue
        removed.add(i)

    return [p.tolist() for i,p in enumerate(pts) if i not in removed]


def imai_iri(points, epsilon):
    pts = np.array(points)
    n = len(pts)

    def valid(i, j):
        a, b = pts[i], pts[j]
        for k in range(i+1, j):
            d = np.abs(np.cross(b - a, a - pts[k])) / np.linalg.norm(b - a)
            if d > epsilon:
                return False
        return True

    # graph edges
    edges = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if valid(i, j):
                edges[i].append(j)

    # shortest path
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[0] = 0

    for i in range(n):
        for j in edges[i]:
            if dist[i] + 1 < dist[j]:
                dist[j] = dist[i] + 1
                prev[j] = i

    # reconstruct
    path = []
    cur = n - 1
    while cur != -1:
        path.append(cur)
        cur = prev[cur]

    return pts[path[::-1]].tolist()