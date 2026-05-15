from collections import defaultdict

class DualGraph:
    def __init__(self):
        self.vertices = []
        self.adj = defaultdict(set)

    def add_edge(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)

    def neighbors(self, v):
        return self.adj[v]
