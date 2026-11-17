import numpy as np
from graphs.build_graph import build_edges, knn_neighbours, neighbour_unit_vectors, nbr_feature

LAT, LON = np.array([28.6, 28.6, 28.6]), np.array([77.2, 77.1, 77.3])      # station 0 centre, 1 west, 2 east


def test_upwind_neighbour_dominates():
    idx, dist = knn_neighbours(LAT, LON, 2)
    E = neighbour_unit_vectors(LAT, LON, idx)
    pm, u, v = np.array([[0.0, 100.0, 10.0]]), np.array([[5.0, 5.0, 5.0]]), np.zeros((1, 3))   # wind blows eastward
    assert abs(nbr_feature(pm, u, v, idx, E, dist)[0, 0] - 100.0) < 1e-3


def test_edge_weight_prefers_upwind():
    ei, ew, _, _ = build_edges(LAT, LON, np.array([1.0, 0.0]), k=2)
    w = {(int(s), int(d)): float(x) for (s, d), x in zip(ei.T, ew)}
    assert w[(1, 0)] > w[(2, 0)]