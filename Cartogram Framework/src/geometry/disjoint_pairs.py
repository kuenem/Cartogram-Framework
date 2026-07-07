import numpy as np

def disjoint_pairs_horizontal_and_vertical_center(polygons):
    # Use centroids, not bounding boxes
    centroids = [np.mean(np.asarray(p), axis=0) for p in polygons]
    horizontal_pairs = []
    vertical_pairs = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = centroids[j][0] - centroids[i][0]  # positive = j is right of i
            dy = centroids[j][1] - centroids[i][1]  # positive = j is above i

            if abs(dx) >= abs(dy):
                # Primarily horizontal separation
                if dx >= 0:
                    horizontal_pairs.append((i, j))  # i left of j
                else:
                    horizontal_pairs.append((j, i))  # j left of i
            else:
                # Primarily vertical separation
                if dy >= 0:
                    vertical_pairs.append((i, j))    # i below j
                else:
                    vertical_pairs.append((j, i))    # j below i

    return horizontal_pairs, vertical_pairs