import numpy as np


def _move_non_empty_frames(data: np.ndarray) -> np.ndarray:
    """ Moves non-empty frames at the first of sequence

    Args:
        data (np.ndarray): shape: (N, T, V, C)

    Returns:
        np.ndarray: _description_
    """

    new_data = np.zeros_like(data)

    frames_agg = data.sum(axis=(2, 3)) 
    non_empty_indices = np.nonzero(frames_agg != 0)

    seq_num = non_empty_indices[0]
    _, indices = np.unique(seq_num, return_index=True)
    indices = np.concatenate([indices, seq_num.size])
    sizes = np.diff(indices)
    frame_num = np.concatenate([np.arange(s) for s in sizes]).ravel()

    # Put non-empty frames at first
    new_data[seq_num, frame_num] = data[seq_num, non_empty_indices[1]]
    
    return new_data, sizes


def pad_null_frames(data: np.ndarray) -> np.ndarray:
    """ It repeats the sequence to fill null frames for each skeleton

    Args:
        data (np.ndarray): Skeleton frames; shape: N, C, T, V
    """
    
    T = data.shape[1]
    data = np.transpose(data, [0, 2, 3, 1])  # N, T, V, C

    # First, put non-empty frames at first. Also return the non-empty frames count per sequence
    new_data, sizes = _move_non_empty_frames(data)
    empty_sizes: np.ndarray = T - sizes

    # Second, create a repeated frames for each sequence to fill empty frames
    repeat_num = np.ceil(empty_sizes / sizes).max()
    repeat_data = np.tile(new_data, (1, repeat_num, 1, 1))
    repeat_data, _ = _move_non_empty_frames(repeat_data)

    seq_num = np.arange(empty_sizes.size)
    seq_num = np.repeat(seq_num, empty_sizes)
    frame_num = np.concatenate([np.arange(s) for s in empty_sizes]).ravel()
    frame_num2 = np.concatenate([np.arange(em) + s for em, s in zip(empty_sizes, sizes)]).ravel()

    # Third, fill empty frames using repeated data
    new_data[seq_num, frame_num2] = repeat_data[seq_num, frame_num]

    data = np.transpose(new_data, [0, 3, 1, 2])  # N, T, V, C

    return data