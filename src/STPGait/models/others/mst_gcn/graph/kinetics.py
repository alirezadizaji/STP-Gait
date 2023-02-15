import numpy as np
import sys

sys.path.extend(['../'])
from . import tools
import networkx as nx

# Edge format: (origin, neighbor)
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (5, 1), (2, 1),
            (0, 1), (15, 0), (16, 0), (17, 15), (16, 18),
            (1, 8), (8, 9), (8, 12), (9, 10), (12, 13),
            (10, 11), (11, 24), (11, 22), (22, 23),
            (13, 14), (14, 21), (14, 19), (19, 20)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')