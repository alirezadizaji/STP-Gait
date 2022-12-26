import numpy as np


def pad_empty_frames(data: np.ndarray) -> np.ndarray:
    """ It swaps non-empty frames to the beginning of the sequence at first place, and then fills empty frames

    Args:
        data (np.ndarray): shape: (N, T, V, C)

    Returns:
        np.ndarray: _description_
    """
    
    T = data.shape[1]

    frames_agg = data.sum(axis=(2, 3)) 
    non_empty_indices = np.nonzero(frames_agg != 0)

    seq_num = non_empty_indices[0]
    _, indices = np.unique(seq_num, return_index=True)
    indices = np.concatenate([indices, [seq_num.size]])
    non_empty_sizes = np.diff(indices)

    empty_sizes: np.ndarray = T - non_empty_sizes
    repeat_num = int(np.ceil(empty_sizes / non_empty_sizes).max())

    # All sequences are repeated with same count, though at last only the first T frames are extracted.
    new_data = np.zeros((data.shape[0], T * repeat_num, *data.shape[2:]))

    frame_num_repeated = np.concatenate([np.arange(s * repeat_num) for s in non_empty_sizes]).ravel()
    seq_num_repeated = np.repeat(seq_num, repeat_num)
    
    non_empty_frames_repeated = []
    for start, end in zip(indices[:-1], indices[1:]):
        seq_frame_indices = non_empty_indices[1][start:end]
        non_empty_frames_repeated.append(np.tile(seq_frame_indices, repeat_num).tolist())
    non_empty_frames_repeated = np.concatenate(non_empty_frames_repeated).ravel()
    
    assert seq_num_repeated.size == frame_num_repeated.size == non_empty_frames_repeated.size

    # To prevent CPU leakage, chunk data indexing into ten subset operations
    tmp_indices = np.arange(seq_num_repeated.size)
    for ti in np.array_split(tmp_indices, 10):
        # Put non-empty frames at first; As it fills "repeated", the empty frames are also filled simultaneously.
        new_data[seq_num_repeated[ti], frame_num_repeated[ti]] = \
            data[seq_num_repeated[ti], non_empty_frames_repeated[ti]]

    new_data = new_data[:, :T, ...]
    return new_data