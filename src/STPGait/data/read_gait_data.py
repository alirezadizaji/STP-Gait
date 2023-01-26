from dataclasses import dataclass
import os
from typing import Optional
import pickle

import numpy as np
import pandas as pd

from ..enums import WalkDirection
from ..preprocess import preprocessing
from ..utils import timer
from ..preprocess.main import PreprocessingConfig

@dataclass
class ProcessingGaitConfig:
    fillZ_empty: bool = True
    preprocessing_conf: PreprocessingConfig = PreprocessingConfig(critical_limit=30)

@timer
def proc_gait_data(load_dir: str, save_dir: str, filename: str="processed.pkl", 
        config: ProcessingGaitConfig = ProcessingGaitConfig()) -> None:
    """ Processes Raw gait dataset (CSV file) provided by OpenPose

    Args:
        load_dir (str): CSV raw data directory to be loaded. It must all parts of the file directory, including its name too.
        save_dir (str): Where to save the processed file.
        filename (str, optional): Filename to store processed file with. Default to processed.pkl.
        config (ProcessingGaitConfig, optional): configuration to process gait data with.
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

    # indexes groups
    upper_body_idx = [8, 9, 12, 1, 2, 3, 4, 5, 6, 7, 0, 15, 16, 17, 18]
    left_foot_idx = [14, 20, 21]
    left_knee_idx = [13]
    right_foot_idx = [11, 23, 24]
    right_knee_idx = [10]

    for idx, r in enumerate(raw_data):
        sample_num_frames = num_frames[idx]
        sample_feature = np.stack(np.split(r, num_nodes, axis=1), axis=1) # T, V, C - 1
        sample_gait = gait_seq[idx]

        sample_z = np.zeros((sample_num_frames, num_nodes))
        if not config.fillZ_empty:
            # Seems like the first two steps is when the patient enters to the process :), since it is always NaN
            step_time = np.array(list(sample_gait['STime'].values()))[2:]
            step_len = np.array(list(sample_gait["SLen"].values()))[2:]
            step_foot = np.array(list(sample_gait["Foot"].values()))[2:]

            total_time = step_time.sum()
            num_frames_per_sec = sample_num_frames / total_time
            start_frame_idx = 0
            end_len = start_len = 0
            
            # fill Z values
            for length, time, foot in zip(step_len, step_time, step_foot):
                step_frames = int(time * num_frames_per_sec) + 1
                
                if step_frames + start_frame_idx > sample_num_frames:
                    step_frames = sample_num_frames - start_frame_idx
                
                if foot == 0: #right foot
                    right_foot_z = np.linspace(start_len, start_len + length, step_frames)
                    right_knee_z = np.linspace(start_len, start_len + (3/4)*length, step_frames)
                    left_knee_z = np.linspace(start_len, start_len + (1/4)*length, step_frames)
                    upper_body_z = np.linspace(start_len, start_len + (1/2)*length, step_frames)
                    left_foot_z = np.zeros((step_frames))

                elif foot == 1: #left foot
                    left_foot_z = np.linspace(start_len, start_len + length, step_frames)
                    left_knee_z = np.linspace(start_len, start_len + (3/4)*length, step_frames)
                    right_knee_z = np.linspace(start_len, start_len + (1/4)*length, step_frames)
                    upper_body_z = np.linspace(start_len, start_len + (1/2)*length, step_frames)
                    right_foot_z = np.zeros((step_frames))

                sample_z[start_frame_idx: start_frame_idx + step_frames, right_foot_idx] = right_foot_z[..., None]
                sample_z[start_frame_idx: start_frame_idx + step_frames, left_foot_idx] = left_foot_z[..., None]
                sample_z[start_frame_idx: start_frame_idx + step_frames, right_knee_idx] = right_knee_z[..., None]
                sample_z[start_frame_idx: start_frame_idx + step_frames, left_knee_idx] = left_knee_z[..., None]
                sample_z[start_frame_idx: start_frame_idx + step_frames, upper_body_idx] = upper_body_z[..., None]

                start_frame_idx += step_frames
                start_len = end_len

        sample_num_frames = min(r.shape[0], max_frame)
        r = r[:sample_num_frames]
        sample_feature = np.concatenate([sample_feature, sample_z[..., None]], axis=2)
        data[idx, :sample_num_frames] = sample_feature

    # Revert walk direction when going away from the camera
    away_idxs = np.nonzero(walk_directions == WalkDirection.AWAY)
    data[away_idxs, ..., 2] = data[away_idxs, ..., 2].max((1, 2)) - data[away_idxs, ..., 2]
    
    data, hard_cases_id = preprocessing(data, config.preprocessing_conf)

    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump((data, labels, names, np.array(hard_cases_id)), f)
