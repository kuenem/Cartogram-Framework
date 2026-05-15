from collections import defaultdict, deque

class Configuration:
    """Represents C(v) in the paper"""
    def __init__(self, vertex_id):
        self.vertex_id = vertex_id
        self.tiles = set()

    def is_connected(self):
        """Check edge-connectivity (BFS)"""
        if not self.tiles:
            return True
        visited = set()
        queue = deque([next(iter(self.tiles))])
        while queue:
            t = queue.popleft()
            visited.add(t)
            for n in t.neighbors:
                if n in self.tiles and n not in visited:
                    queue.append(n)
        return visited == self.tiles