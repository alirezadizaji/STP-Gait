from typing import Optional

import numpy as np

def remove_duplicate_edges(edge_index: np.ndarray) -> np.ndarray:
    """ It removes duplicate edges from the given graph edges by finding the index of edges 
    having the same source and destination node indices, then it takes the latter edge index
    and removes it from the edge set.

    Args:
        edge_index (np.ndarray): edge set from which duplicate samples are removed. Shape: 2, E

    Returns:
        np.ndarray: edge_index after removing duplicate edges
    """
    edge_index = edge_index.astype(np.uint16)
    src, dst = edge_index
    E = src.size

    # find the source-destination difference between each two edges
    diff_src = src[:, np.newaxis] - src[np.newaxis, :] # E, E

    # find the indices of edges having the same source
    same_src = np.nonzero(diff_src == 0)
    idx_src1, idx_src2 = same_src
    ## To prevent Memory leakage
    del diff_src 

    # checkout if corresponding destinations are equal too
    diff_dst = dst[:, np.newaxis] - dst[np.newaxis, :] # E, E
    mask = diff_dst[idx_src1, idx_src2] == 0
    ## To prevent Memory leakage
    del diff_dst

    # generate a mask to ignore duplicate (latter) edges
    edge_mask = np.ones(E, dtype=np.bool)
    edge_mask[idx_src2[mask]] = False

    # remove duplicate edges
    edge_index = np.stack([src[edge_mask], dst[edge_mask]]) 

    return edge_index

def replicate_edge_index_in_frames(edge_index: np.ndarray, T: int) -> np.ndarray:
    """ It evenly replicates edge sets into multiple frames

    Args:
        edge_index (np.ndarray): A set of edge sets to be replicated into multiple frames
        T (int): Number of frames to operation replication

    Returns:
        np.ndarray: replicated edge_index
    """

    # E, V = number of edges and nodes
    E = edge_index[0].size
    V = np.unique(edge_index).size

    frames_edge_index = np.tile(edge_index, T)
    frames_start_index = np.arange(T).repeat(E) * V
    frames_edge_index = frames_edge_index + frames_start_index[np.newaxis, :]

    return frames_edge_index


def generate_inter_frames_edge_index_mode1(T: int, V: int, dilation: int = 1) -> np.ndarray:
    r""" It generates inter-frame edge indices between non-consecutive constant-coefficient-distant nodes.
    For instance, suppose `T = 100`, and `V = 25` and `dilation = 3`, then nodes with indices `0, 75, 150, ...` will be connected with each other,
    then nodes with indices `1, 76, 151, ...` will be connected with each other and so on.

    Args:
        T (int): Total number of frames each sample sequence would have.
        V (int): Number of nodes within each frame.
        dilation (int, optional): number of dilation to consider for skipping an amount of frames. Defaults to 1.

    Returns:
        Union[np.ndarray, None]: _description_
    """
    assert dilation > 0

    node_indices = np.arange(T * V)
    diff = node_indices[:, np.newaxis] - node_indices[np.newaxis, :]
    row, col = np.nonzero(np.logical_and(np.abs(diff) % (V * dilation) == 0, np.abs(diff) > 0))
    inter_frame_edges = np.stack([row, col], axis=0)
    return inter_frame_edges


def generate_inter_frames_edge_index_mode2(T: int, V: int, I: int = 30, offset: Optional[int] = None) -> np.ndarray:
    r""" It generates inter frames edge index between nodes using second mode: 
    A number of consecutive frames will have inter frames, then skip it for a offset, then go on for the rest.
    For instance, with `I = 30` and `offset = 20`, corresponding nodes within frames with indices of `0, 1, ..., 29` (1st inter-frame chunk) will connect to each other,
    then `20, 21, ..., 49` (2nd interframe chunk) will connect to each other, and so on. If `offset` is None, then it will be equivalent with `I`.

    Args:
        T (int): Total number of frames each sample sequence would have.
        V (int): Number of nodes within each frame.
        I (int, optional): Number of consecutive inter frames to be connected with each other. Defaults to `30`.
        offset (Optional[int], None): It skips connecting consecutive frames for an amount; If `None`, then by default there will be no overlap between two chunks of interframes. Defaults to `None`.

    Returns:
        np.ndarray: generated inter-frame edge set
    """
    if offset is None:
        offset = I
        
    # To make computations easier, let's suppose each inter-frame chunks (EVEN the LAST ONE) has the same number of frames
    remained = T % I
    M = (T - remained + I) // offset

    # node indices within each chunk LOCALLY
    chunk_node_index = np.tile(np.arange(V), I) + np.repeat(np.arange(I), V) # I * V (= L)
    
    # Find inter-frame edge indices within each chunk LOCALLY
    diff = chunk_node_index[:, np.newaxis] - chunk_node_index[np.newaxis, :] # L * L
    diff = np.abs(diff)
    src, dst = np.nonzero(np.logical_and(diff % I == 0, diff > 0))
    chunk_edge_index = np.stack([src, dst])
    E = chunk_edge_index.shape[1]

    # Find inter-frame edge indices within each chunk GLOBALLY
    chunks_start_index = np.repeat(np.arange(M) * offset * V, E) # M * E (= K)
    chunks_edge_index = np.tile(chunk_edge_index, (1, M)) # 2, K
    chunks_edge_index = chunks_edge_index + chunks_start_index[np.newaxis, :] # 2, K

    # Remove edges having forbidden nodes, i.e. nodes not exist in general.
    forbidden_start_node_index = T * V
    src, dst = chunks_edge_index
    mask = np.logical_and(src < forbidden_start_node_index, dst < forbidden_start_node_index)
    chunks_edge_index = np.stack([src[mask], dst[mask]])

    # Due to overlapping chunking of consecutive frames, it is possible to have duplicate edges, therefore remove them.
    if offset < I:
        chunks_edge_index = remove_duplicate_edges(chunks_edge_index)

    return chunks_edge_index