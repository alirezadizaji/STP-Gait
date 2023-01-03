import numpy as np
import torch

class Skeleton:
    num_nodes: int = 25

    _one_direction_edges: np.ndarray = np.array([
        [0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [8, 12], [9, 10], [10, 11], 
        [11, 24], [11, 23], [12, 13], [13, 14], [14, 21], [14, 20]
    ]).T
    edges: np.ndarray = np.concatenate((_one_direction_edges, _one_direction_edges[[1, 0]]), axis=1)

    self_loop_edges = np.concatenate([edges, np.tile(np.arange(num_nodes), (2, 1))], axis=1)

    upper_part_node_indices = np.concatenate([np.arange(8), np.arange(4) + 15])

    @staticmethod
    def lower_part_edges():
        src, dst = Skeleton.self_loop_edges
        src_mask = src[:, np.newaxis] == Skeleton.upper_part_node_indices[np.newaxis, :]
        src_mask = src_mask.sum(1) == 0

        dst_mask = dst[:, np.newaxis] == Skeleton.upper_part_node_indices[np.newaxis, :]
        dst_mask = dst_mask.sum(1) == 0

        mask = np.logical_or(src_mask, dst_mask)

        return np.stack([src[mask], dst[mask]])

    @staticmethod
    def get_simple_interframe_edges(num_frames: int, dilation: int = 30, use_lower_part: bool = False) -> torch.Tensor:
        edges = Skeleton.self_loop_edges if not use_lower_part else Skeleton.lower_part_edges()
        M = edges[0].size
        edges = np.tile(edges, num_frames)
        start_index = np.arange(num_frames).repeat(M) * Skeleton.num_nodes
        edges = edges + start_index[np.newaxis, :]

        # Build inter frame edges
        if dilation > 1:
            node_indices = np.arange(num_frames * Skeleton.num_nodes)
            diff = node_indices[:, np.newaxis] - node_indices[np.newaxis, :]
            row, col = np.nonzero(np.logical_and(np.abs(diff) % (Skeleton.num_nodes * dilation) == 0, np.abs(diff) > 0))
            inter_frame_edges = np.stack([row, col], axis=0)
            edges = np.concatenate([edges, inter_frame_edges], axis=1)

        return torch.from_numpy(edges)