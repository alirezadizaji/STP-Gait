import os
from typing import Tuple
import pickle

import numpy as np
import pandas as pd

from ..enums import WalkDirection
from ..preprocess import preprocessing
from ..utils import timer

@timer
def proc_gait_data(load_dir: str, save_dir: str, fillZ_empty: bool = False, 
        normalize: bool = False) -> None:
    """ Processes Raw gait dataset (CSV file) provided by OpenPose

    Args:
        load_dir (str): CSV raw data directory to be loaded. It must all parts of the file directory, including its name too.
        save_dir (str): Where to save the processed file. The file name will always be `processed.pkl`.
        fillZ_empty (bool): If True, then fill Z dimension as zero, O.W. fill it by processing patient steps information.
        normalize (bool): If True, then normalize patient locations using gaussian and minimax normalization, O.W. leave it intact.
    """
    num_features = 3
    num_nodes = 25

    with open(load_dir, "rb") as f:
        df = pd.read_pickle(f)
    
    raw_data = df['keypoints'].values
    gait_seq = df['gait_sequence'].values
    labels = df['class'].values
    names = df['video_name'].values
    walk_directions = df['walk_direction'].values
    
    num_frames = [r.shape[0] for r in raw_data]
    mean, std = np.mean(num_frames), np.std(num_frames)
    max_frame = int(np.ceil(mean + std))
    num_samples = raw_data.shape[0]   
    data = np.zeros((num_samples, max_frame, num_nodes, num_features)) # N, T, V, C

    for idx, r in enumerate(raw_data):
        sample_num_frames = min(r.shape[0], max_frame)
        r = r[:sample_num_frames]
        sample_feature = np.stack(np.split(r, num_nodes, axis=1), axis=1) # T, V, C - 1
        sample_gait = gait_seq[idx]

        sample_z = np.zeros((sample_num_frames, num_nodes))
        if not fillZ_empty:
            # Seems like the first two steps is when the patient enters to the process :), since it is always NaN
            step_time = np.array(list(sample_gait['STime'].values()))[2:]
            step_len = np.array(list(sample_gait["SLen"].values()))[2:]

            total_time = step_time.sum()
            num_frames_per_sec = sample_num_frames / total_time
            start_frame_idx = 0
            end_len = start_len = 0
            
            # fill Z values
            for length, time in zip(step_len, step_time):
                step_frames = int(time * num_frames_per_sec) + 1
                
                if step_frames + start_frame_idx > sample_num_frames:
                    step_frames = sample_num_frames - start_frame_idx
                
                end_len = start_len + length
                zs = np.linspace(start_len, end_len, step_frames)
                sample_z[start_frame_idx: start_frame_idx + step_frames] = zs[..., None]

                start_frame_idx += step_frames
                start_len = end_len

        sample_feature = np.concatenate([sample_feature, sample_z[..., None]], axis=2)
        data[idx, :sample_num_frames] = sample_feature

    # Revert walk direction when going away from the camera
    away_idxs = np.nonzero(walk_directions == WalkDirection.AWAY)
    data[away_idxs, ..., 2] = data[away_idxs, ..., 2].max((1, 2)) - data[away_idxs, ..., 2]
    
    data, labels, names = preprocessing(data, labels, names, normalize=normalize)

    with open(os.path.join(save_dir, "processed.pkl"), 'wb') as f:
        pickle.dump((data, labels, names), f)