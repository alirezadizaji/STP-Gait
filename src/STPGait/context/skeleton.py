from typing import Optional

import numpy as np
import torch

from ..enums import Body
from ..utils import replicate_edge_index_in_frames, \
    generate_inter_frames_edge_index_mode1, generate_inter_frames_edge_index_mode2

class Skeleton:
    CENTER: int = 8
    LEFT_HEAP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 10
    num_nodes: int = 25

    HEAD_EYES_EARS: np.ndarray = np.array([0, 15, 16, 17, 18])
    SHOULDERS: np.ndarray = np.array([2, 5])
    ELBOWS_HANDS: np.ndarray = np.array([3, 4, 6, 7])

    LEFT_FOOT: np.ndarray = np.array([14, 19, 20, 21])
    RIGHT_FOOT: np.ndarray = np.array([11, 22, 23, 24])
    FOOTS: np.ndarray = np.array([19, 20, 21, 22, 23, 24])
    WRISTS_FOOTS: np.ndarray = np.array([11, 14])
    WHOLE_FOOT: np.ndarray = np.concatenate((FOOTS, WRISTS_FOOTS))
    
    NECK_TO_KNEE: np.ndarray = np.array([1, CENTER, 9, 12, 10, 13])
    
    NON_CRITICAL: np.ndarray = np.concatenate((HEAD_EYES_EARS, SHOULDERS, FOOTS))
    CRITICAL: np.ndarray = np.concatenate((ELBOWS_HANDS, NECK_TO_KNEE, WRISTS_FOOTS))
    
    UPPER_BODY: np.ndarray = np.array([8, 9, 12, 1, 2, 3, 4, 5, 6, 7, 0, 15, 16, 17, 18])
    # CENTER NODES are exist in both upper and lower parts
    LOWER_BODY: np.ndarray = np.array([8, 9, 12, 10, 11, 13, 14, 19, 20, 21, 22, 23, 24])

    _one_direction_edges: np.ndarray = np.array([
        [0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [8, 12], [9, 10], [10, 11], 
        [11, 24], [11, 23], [12, 13], [13, 14], [14, 21], [14, 20],
        [19, 20], [22, 23],
    ]).T
    edges: np.ndarray = np.concatenate((_one_direction_edges, _one_direction_edges[[1, 0]]), axis=1)

    self_loop_edges = np.concatenate([edges, np.tile(np.arange(num_nodes), (2, 1))], axis=1)

    @staticmethod
    def half_part_edges(edge_index: np.ndarray, use_lower_part: bool = True) -> np.ndarray:
        src, dst = edge_index

        node_indices = Skeleton.LOWER_BODY if use_lower_part else Skeleton.UPPER_BODY
        src_mask = src[:, np.newaxis] == node_indices[np.newaxis, :]
        src_mask = src_mask.sum(1) > 0

        dst_mask = dst[:, np.newaxis] == node_indices[np.newaxis, :]
        dst_mask = dst_mask.sum(1) > 0

        mask = np.logical_and(src_mask, dst_mask)
        edges_1d = np.concatenate([src[mask], dst[mask]])
        u, indices = np.unique(edges_1d, return_inverse=True)
        res = np.stack(np.split(indices, 2))

        return res

    @staticmethod
    def get_vanilla_edges(T: int, body_part: Body = Body.WHOLE):
        """ Please checkout documentation of `replicate_edge_index_in_frames` function. """
        edge_index = Skeleton.self_loop_edges
        if body_part == Body.LOWER:
            edge_index = Skeleton.half_part_edges(edge_index, use_lower_part=True)
        elif body_part == Body.UPPER:
            edge_index = Skeleton.half_part_edges(edge_index, use_lower_part=False)
        else:
            pass
        V = np.unique(edge_index).size
        edge_index = replicate_edge_index_in_frames(edge_index, T)

        return edge_index, V

    @staticmethod
    def get_simple_interframe_edges(T: int, dilation: int = 30, body_part: Body = Body.WHOLE) -> torch.Tensor:
        r""" Please checkout documentation of `generate_inter_frames_edge_index_mode1` function. """
        
        edge_index, V = Skeleton.get_vanilla_edges(T, body_part)
        
        inter_frame_edges = generate_inter_frames_edge_index_mode1(T, V, dilation)
        edge_index = np.concatenate([edge_index, inter_frame_edges], axis=1)

        return torch.from_numpy(edge_index)

    @staticmethod
    def get_interframe_edges_mode2(T: int, I: int = 30, offset: Optional[int] = None, 
            body_part: Body = Body.WHOLE) -> torch.Tensor:
        r""" Please checkout documentation of `generate_inter_frames_edge_index_mode2` function. """
        
        edge_index, V = Skeleton.get_vanilla_edges(T, body_part)

        inter_frame_edges = generate_inter_frames_edge_index_mode2(T, V, I, offset)
        edge_index = np.concatenate([edge_index, inter_frame_edges], axis=1)
        return torch.from_numpy(edge_index.astype(np.int64))  