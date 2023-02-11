from . import tools

num_node = 25
class Graph:
    self_link = [(i, i) for i in range(num_node)]
    inward = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
            [1, 8], [8, 9], [8, 12], [9, 10], [10, 11], 
            [11, 24], [11, 23], [12, 13], [13, 14], [14, 21], [14, 20],
            [19, 20], [22, 23]]
    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self):
        self.edges = Graph.neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
