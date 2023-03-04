import os
import pickle
from typing import List

import pandas as pd

from .read_gait_data import proc_gait_data, ProcessingGaitConfig

def read_multi_pkl_gait_data(load_dir: List[str], unite_save_dir: str, save_dir: str, filename: str="processed.pkl", 
        config: ProcessingGaitConfig = ProcessingGaitConfig()):
    pd_list = list()
    for i, l in enumerate(load_dir):
        with open(l, "rb") as f:
            df = pd.read_pickle(f)
            df["pkl_no."] = i
            pd_list.append(df)
    
    pd.concat(pd_list).to_pickle(unite_save_dir)

    proc_gait_data(unite_save_dir, save_dir, filename, config)
    
    with open(unite_save_dir, "rb") as f:
        df = pd.read_pickle(f)
    
    with open(os.path.join(save_dir, filename), 'rb') as f:
        data, labels, names, hard_cases_id = pickle.load(f)

    kfold_no = df["pkl_no."].values

    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump((data, labels, names, hard_cases_id, kfold_no), f)