from dataclasses import dataclass
import os
import pickle

import numpy as np
import pandas as pd

from ..enums import Step, WalkDirection
from ..preprocess import preprocessing
from ..utils import timer
from ..preprocess.main import PreprocessingConfig
from ..context import Skeleton

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
    if not config.fillZ_empty:
        gait_seq = df['gait_sequence'].values
        walk_directions = df['walk_direction'].values
    labels = df['class'].values
    names = df['video_name'].values
    
    num_frames = [r.shape[0] for r in raw_data]
    mean, std = np.mean(num_frames), np.std(num_frames)
    max_frame = int(np.ceil(mean + std))
    num_samples = raw_data.shape[0]   
    data = np.zeros((num_samples, max_frame, num_nodes, num_features)) # N, T, V, C

    for idx, r in enumerate(raw_data):
        sample_num_frames = num_frames[idx]
        sample_feature = np.stack(np.split(r, num_nodes, axis=1), axis=1) # T, V, C - 1

        sample_z = np.zeros((sample_num_frames, num_nodes))

        # Fill Z values using manual analysis :/
        if not config.fillZ_empty:
            sample_gait = gait_seq[idx]
            # Seems like the first two steps is when the patient enters to the process :), since it is always NaN
            step_time = np.array(list(sample_gait['STime'].values()))[2:]
            step_len = np.array(list(sample_gait["SLen"].values()))[2:]
            step_foot = np.array(list(sample_gait["Foot"].values()))[2:]

            total_time = step_time.sum()
            num_frames_per_sec = sample_num_frames / total_time
            # Frame zero always have Z = 0
            start_frame_idx = 1
            
            wd = walk_directions[idx]

            # fill Z values
            for L, time, foot in zip(step_len, step_time, step_foot):
                step_frames = int(time * num_frames_per_sec) + 1

                if wd == WalkDirection.AWAY:
                    L = -L
                
                if step_frames + start_frame_idx > sample_num_frames:
                    step_frames = sample_num_frames - start_frame_idx
                

                start_len_rf = sample_z[start_frame_idx - 1, Skeleton.RIGHT_FOOT[0]]
                start_len_rk = sample_z[start_frame_idx - 1, Skeleton.RIGHT_KNEE]
                start_len_lf = sample_z[start_frame_idx - 1, Skeleton.LEFT_FOOT[0]]
                start_len_lk = sample_z[start_frame_idx - 1, Skeleton.LEFT_KNEE]
                start_len_ub = sample_z[start_frame_idx - 1, Skeleton.UPPER_BODY[0]]
                
                ub_step = 0.5 * L
                if foot == Step.RIGHT:
                    lf_step = 0
                    lk_step = 0.25 * L
                    rk_step = 0.75 * L
                    rf_step = L
                elif foot == Step.LEFT:
                    rf_step = 0
                    rk_step = 0.25 * L
                    lk_step = 0.75 * L
                    lf_step = L
                else:
                    raise ValueError()

                upper_body_z = np.linspace(start_len_ub, start_len_ub + ub_step, step_frames)
                left_foot_z = np.linspace(start_len_lf, start_len_lf + lf_step, step_frames)
                left_knee_z = np.linspace(start_len_lk, start_len_lk + lk_step, step_frames)
                right_knee_z = np.linspace(start_len_rk, start_len_rk + rk_step, step_frames)
                right_foot_z = np.linspace(start_len_rf, start_len_rf + rf_step, step_frames)
                
                ST, ET = start_frame_idx, start_frame_idx + step_frames
                sample_z[ST: ET, Skeleton.RIGHT_FOOT] = right_foot_z[..., None]
                sample_z[ST: ET, Skeleton.LEFT_FOOT] = left_foot_z[..., None]
                sample_z[ST: ET, Skeleton.RIGHT_KNEE] = right_knee_z
                sample_z[ST: ET, Skeleton.LEFT_KNEE] = left_knee_z
                sample_z[ST: ET, Skeleton.UPPER_BODY] = upper_body_z[..., None]

                start_frame_idx += step_frames

        eligible_num_frames = min(r.shape[0], max_frame)
        sample_feature = np.concatenate([sample_feature, sample_z[..., None]], axis=2)
        data[idx, :eligible_num_frames] = sample_feature[:eligible_num_frames]

    data, hard_cases_id = preprocessing(data, config.preprocessing_conf)

    # Revert walk direction when going away from the camera
    if not config.fillZ_empty:
        away_idxs = np.nonzero(walk_directions == WalkDirection.AWAY)[0]
        data[away_idxs, ..., 2] = data[away_idxs, ..., 2] - data[away_idxs, ..., 2].min((1, 2), keepdims=True)
    
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump((data, labels, names, np.array(hard_cases_id)), f)
