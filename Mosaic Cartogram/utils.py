from collections import defaultdict, deque
import math
import random
from DualGraph import DualGraph
from Tile import Tile
from Configuration import Configuration
from utils import *


def compute_schnyder_spanning_tree(GM):
    # For toy version: BFS tree
    root = GM.vertices[0]
    visited = {root}
    tree = []

    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in GM.neighbors(u):
            if v not in visited:
                visited.add(v)
                tree.append((u, v))
                queue.append(v)
    return tree


def assign_tiles_from_visibility_drawing(configurations, tree, tiling):
    tiles = list(tiling.values())
    random.shuffle(tiles)

    i = 0
    for v in configurations:
        configurations[v].tiles.add(tiles[i])
        i += 1


def scale_polygon_to_area(polygon, target_area):
    return polygon  # dummy


def generate_small_translations():
    return [(0,0), (1,0), (0,1), (-1,0), (0,-1)]


def translate_polygon(poly, offset):
    return poly


def select_tiles_with_max_overlap(_, tiling, k):
    return set(random.sample(list(tiling.values()), k))


def symmetric_difference_area(A, B):
    return len(A.symmetric_difference(B))


def force_directed_move(guiding_shapes, GM):
    return False  # convergence immediately for toy


def adjacency_preserved(configurations):
    return True  # relaxed in toy model


def improves_max_delta(u, v, configurations, guiding_shapes):
    return True


def tile_distance(Su, Sv):
    return abs(len(Su) - len(Sv))


def attraction_force(Su, Sv, dist):
    return (0.01 * dist, 0.01 * dist)


def repulsion_force(u, v, overlap, guiding_shapes):
    return (-0.02 * len(overlap), -0.02 * len(overlap))


def translate_guiding_shape(S, dx, dy):
    # guiding shapes are tile sets → no-op here
    pass


def solve_min_cost_flow(flow_network):
    return []  # dummy flow


def apply_flow_to_configurations(flow, configurations):
    pass


def boundary_tiles(C, tiling):
    return C.tiles


def total_negative_supply(configurations):
    return 0


def adjacent_boundary_pairs(configurations):
    return []


def compute_transfer_cost(s, t, guiding_shapes):
    return 0


def build_dual_graph(map_data, weights):
    GM = DualGraph()
    GM.vertices = list(weights.keys())

    for u, v in map_data:
        GM.add_edge(u, v)

    return GM


def try_set_tile(tile, from_v, to_v, configurations, guiding_shapes):
    """
    Section 4.2 – Reshaping operation (set(t, v))

    Attempts to move `tile` from configuration `from_v` to `to_v`.
    Returns True if the move is accepted.
    """
    C_from = configurations[from_v]
    C_to = configurations[to_v]

    if tile not in C_from.tiles:
        return False

    # Apply tentative move
    C_from.tiles.remove(tile)
    C_to.tiles.add(tile)

    # Validity checks (simplified)
    valid = (
        C_from.is_connected() and
        C_to.is_connected()
    )

    if valid:
        return True

    # Rollback if invalid
    C_to.tiles.remove(tile)
    C_from.tiles.add(tile)
    return False


def build_flow_network(configurations, guiding_shapes, tiling):
    """
    Section 4.3 – Build a (simplified) flow network

    This toy version only records size mismatches.
    """
    network = MinCostFlow()

    # Compute size differences
    size_diff = {}
    for v, C in configurations.items():
        size_diff[v] = len(C.tiles) - len(guiding_shapes[v])

    network.size_diff = size_diff
    return network
