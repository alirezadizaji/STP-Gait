import numpy as np
import torch

from ..utils import remove_duplicate_edges, replicate_edge_index_in_frames, \
    generate_inter_frames_edge_index_mode1, generate_inter_frames_edge_index_mode2

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
    def lower_part_edges(edge_index: np.ndarray) -> np.ndarray:
        src, dst = edge_index
        src_mask = src[:, np.newaxis] == Skeleton.upper_part_node_indices[np.newaxis, :]
        src_mask = src_mask.sum(1) == 0

        dst_mask = dst[:, np.newaxis] == Skeleton.upper_part_node_indices[np.newaxis, :]
        dst_mask = dst_mask.sum(1) == 0

        mask = np.logical_and(src_mask, dst_mask)
        edges_1d = np.concatenate([src[mask], dst[mask]])
        u, indices = np.unique(edges_1d, return_inverse=True)
        res = np.stack(np.split(indices, 2))

        return res

    @staticmethod
    def get_simple_interframe_edges(T: int, dilation: int = 30, use_lower_part: bool = False) -> torch.Tensor:
        edge_index = Skeleton.self_loop_edges
        if use_lower_part:
            edge_index = Skeleton.lower_part_edges(edge_index)

        V = np.unique(edge_index).size
        edge_index = replicate_edge_index_in_frames(edge_index, T)

        inter_frame_edges = generate_inter_frames_edge_index_mode1(T, V, dilation)
        edge_index = np.concatenate([edge_index, inter_frame_edges], axis=1)

        return torch.from_numpy(edge_index)

    @staticmethod
    def get_interframe_edges_mode2(T: int, I: int = 30, use_lower_part: bool = False) -> torch.Tensor:
        edge_index = Skeleton.self_loop_edges
        if use_lower_part:
            edge_index = Skeleton.lower_part_edges(edge_index)

        V = np.unique(edge_index).size
        edge_index = replicate_edge_index_in_frames(edge_index, T)

        inter_frame_edges = generate_inter_frames_edge_index_mode2(T, V, I)
        edge_index = np.concatenate([edge_index, inter_frame_edges], axis=1)
        edge_index = remove_duplicate_edges(edge_index)
        return edge_index  