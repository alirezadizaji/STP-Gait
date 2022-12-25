import numpy as np
import pandas as pd

from ..preprocess import preprocessing

def proc_gait_data(path: str) -> np.ndarray:
    """ Processes Raw gait dataset (CSV file) provided by OpenPose

    Args:
        path (str): path to CSV raw data

    Returns:
        np.ndarray: processed CSV file in numpy array format with shape N (number of samples), C (number of features), T (number of frames), V (number of nodes)
    """
    num_features = 3
    num_nodes = 25

    with open(path, "rb") as f:
        df = pd.read_pickle(f)
    
    raw_data = df['keypoints'].values
    gait_seq = df['gait_sequence'].values

    num_frames = [r.shape[0] for r in raw_data]
    mean, std = np.mean(num_frames), np.std(num_frames)
    max_frame = mean + std
    num_samples = raw_data.shape[0]    
    data = np.zeros(num_samples, max_frame, num_nodes, num_features) # N, T, V, C

    for idx, r in enumerate(raw_data):
        sample_num_frames = min(r.shape[0], max_frame)
        data = data[:sample_num_frames]
        sample_feature = np.stack(np.split(data, num_nodes), dim=0) # V * (C - 1) -> V, C - 1

        sample_gait = gait_seq[idx]

        # Seems like the first two steps is when the patient enters to the process :), since it is always NaN
        step_time = np.array(sample_gait['STime'].values())[2:]
        step_len = np.array(sample_gait["SLen"].values())[2:]

        total_time = step_time.sum()
        num_frames_per_sec = sample_num_frames / total_time
        start_frame_idx = 0
        end_len = start_len = 0
        
        # fill Z values
        sample_z = np.zeros(num_nodes)
        for length, time in zip(step_len, step_time):
            step_frames = int(time * num_frames_per_sec) + 1
            
            if step_frames + start_frame_idx > sample_num_frames:
                step_frames = sample_num_frames - start_frame_idx
            
            end_len = start_len + length
            zs = np.linspace(start_len, end_len, step_frames)
            sample_z[start_frame_idx, start_frame_idx + step_frames] = zs

            start_frame_idx += step_frames
            start_len = end_len

        sample_feature = np.concatenate([sample_feature, sample_z[:, None]], dim=1)
        data[idx, :sample_num_frames] = sample_feature
    
    # swap Y and Z features 
    data[..., [1, 2]] = data[..., [2, 1]]
    data = preprocessing(data)

    return data

if __name__ == "__main__":
    proc_gait_data("../Data/output_1.pkl")