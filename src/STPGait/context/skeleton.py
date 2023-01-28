from typing import Optional

import numpy as np
import torch

from ..utils import replicate_edge_index_in_frames, \
    generate_inter_frames_edge_index_mode1, generate_inter_frames_edge_index_mode2

class Skeleton:
    CENTER: int = 8
    LEFT_HEAP: int = 12
    LEFT_KNEE: int = 13
    num_nodes: int = 25

    HEAD_EYES_EARS: np.ndarray = np.array([0, 15, 16, 17, 18])
    SHOULDERS: np.ndarray = np.array([2, 5])
    ELBOWS_HANDS: np.ndarray = np.array([3, 4, 6, 7])

    FOOTS: np.ndarray = np.array([19, 20, 21, 22, 23, 24])
    WRISTS_FOOTS: np.ndarray = np.array([11, 14])
    WHOLE_FOOT: np.ndarray = np.concatenate((FOOTS, WRISTS_FOOTS))
    
    NECK_TO_KNEE: np.ndarray = np.array([1, CENTER, 9, 12, 10, 13])
    
    NON_CRITICAL: np.ndarray = np.concatenate((HEAD_EYES_EARS, SHOULDERS, FOOTS))
    CRITICAL: np.ndarray = np.concatenate((ELBOWS_HANDS, NECK_TO_KNEE, WRISTS_FOOTS))

    UPPER_BODY: np.ndarray = np.array([8, 9, 12, 1, 2, 3, 4, 5, 6, 7, 0, 15, 16, 17, 18])
    LEFT_FOOT: np.ndarray = np.array([14, 20, 21, 22])
    LEFT_KNEE: np.ndarray = np.array([13])
    RIGHT_FOOT: np.ndarray = np.array([11, 23, 24, 19])
    RIGHT_KNEE: np.ndarray = np.array([10])

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
    def get_vanilla_edges(T: int, use_lower_part: bool = False):
        """ Please checkout documentation of `replicate_edge_index_in_frames` function. """
        edge_index = Skeleton.self_loop_edges
        if use_lower_part:
            edge_index = Skeleton.lower_part_edges(edge_index)

        V = np.unique(edge_index).size
        edge_index = replicate_edge_index_in_frames(edge_index, T)

        return edge_index, V

    @staticmethod
    def get_simple_interframe_edges(T: int, dilation: int = 30, use_lower_part: bool = False) -> torch.Tensor:
        r""" Please checkout documentation of `generate_inter_frames_edge_index_mode1` function. """
        
        edge_index, V = Skeleton.get_vanilla_edges(T, use_lower_part)
        
        inter_frame_edges = generate_inter_frames_edge_index_mode1(T, V, dilation)
        edge_index = np.concatenate([edge_index, inter_frame_edges], axis=1)

        return torch.from_numpy(edge_index)

    @staticmethod
    def get_interframe_edges_mode2(T: int, I: int = 30, offset: Optional[int] = None, 
            use_lower_part: bool = False) -> torch.Tensor:
        r""" Please checkout documentation of `generate_inter_frames_edge_index_mode2` function. """
        
        edge_index, V = Skeleton.get_vanilla_edges(T, use_lower_part)

        inter_frame_edges = generate_inter_frames_edge_index_mode2(T, V, I, offset)
        edge_index = np.concatenate([edge_index, inter_frame_edges], axis=1)
        return torch.from_numpy(edge_index.astype(np.int64))  