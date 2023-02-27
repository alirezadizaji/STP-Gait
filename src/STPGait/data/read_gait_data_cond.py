from dataclasses import dataclass
import os
import pickle

import numpy as np
import pandas as pd

from ..utils import timer
from ..preprocess.main import PreprocessingConfig
from .read_gait_data import proc_gait_data

@dataclass
class ProcessingGaitConfig:
    fillZ_empty: bool = True
    preprocessing_conf: PreprocessingConfig = PreprocessingConfig(critical_limit=30)

@timer
def proc_gait_data_v2(load_dir: str, save_dir: str, filename: str="processed.pkl", 
        config: ProcessingGaitConfig = ProcessingGaitConfig()) -> None:

    proc_gait_data(load_dir, save_dir, filename, config)

    with open(os.path.join(save_dir, filename), 'rb') as f:
        data, labels, names, hard_cases_id = pickle.load(f)

    os.remove(os.path.join(save_dir, filename))
    with open(load_dir, "rb") as f:
        df = pd.read_pickle(f)
    
    condition = df['condition'].values
    sid = df['subjectID'].values
    idu, id_uv = np.unique(sid, return_inverse=True)
    condu, cond_uv = np.unique(condition, return_inverse=True)
    
    N = idu.size
    M = condu.size

    new_data = np.full((N, M, *data.shape[1:]), fill_value=-np.inf)        # N, M, T, V, C
    new_labels = ["unlabeled" for _ in range(N)]                            # N

    for x, y, sid, cond in zip(data, labels, id_uv, cond_uv):
        new_data[sid, cond] = x
        new_labels[sid] = y
        
    new_labels = np.array(new_labels)
    
    
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump((new_data, new_labels, names, np.array(hard_cases_id)), f)