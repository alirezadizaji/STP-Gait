import numpy as np
import torch

class Skeleton:
    num_nodes: int = 25

    _one_direction_edges: np.ndarray = np.array([
        [0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [8, 12], [9, 10], [10, 11], 
        [11, 24], [11, 23], [12, 13], [13, 14], [14, 21], [14, 20]
    ])

    edges: np.ndarray = np.concatenate((_one_direction_edges, _one_direction_edges[:, [1, 0]]))
    
    self_loop_edges = np.concatenate([edges, np.tile(np.arange(num_nodes)[np.newaxis], (2, 1))])

    @staticmethod
    def get_simple_interframe_edges(num_frames: int, use_self_loop: bool = True) -> torch.Tensor:
        edges = Skeleton.self_loop_edges if use_self_loop else Skeleton.edges
        
        M = edges[0].size
        edges = np.tile(edges, num_frames)
        start_index = np.arange(num_frames).repeat(M) * Skeleton.num_nodes
        edges = edges + start_index[np.newaxis, :]

        # Build inter frame edges
        node_indices = np.arange(num_frames * Skeleton.num_nodes)
        diff = node_indices[:, np.newaxis] - node_indices[np.newaxis, :]
        row, col = np.nonzero(np.logical_or(np.abs(diff) % Skeleton.num_nodes == 0, np.abs(diff) > 0))
        inter_frame_edges = np.concatenate([row, col])
        edges = np.concatenate([edges, inter_frame_edges])

        return torch.from_numpy(edges)