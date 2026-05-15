class Tile:
    def __init__(self, x, y):
        self.coord = (x, y)          # grid coordinate (x, y)
        self.neighbors = set()      # adjacent tiles

    def __repr__(self):
        return f"T{self.coord}"

    def __hash__(self):
        return hash(self.coord)

    def __eq__(self, other):
        return self.coord == other.coord