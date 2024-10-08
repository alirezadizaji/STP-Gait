import os
from argparse import ArgumentParser

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing_extensions import Protocol

from ..context import Skeleton
from ..data import proc_gait_data, ProcessingGaitConfig
from ..preprocess.main import PreprocessingConfig
from ..utils import stdout_stderr_setter


class Func(Protocol):
    def __call__(self, data: np.ndarray, labels: np.ndarray, names: np.ndarray) -> None:
        ...

def initializer(work_dir: str) -> Func:
    
    os.makedirs(work_dir, exist_ok=True)

    def visualize_samples(raw: np.ndarray, processed: np.ndarray, labels: np.ndarray, 
            names: np.ndarray) -> None:

        T = raw.shape[1]

        def _pre_setting_ax(ax, maxes, mines):
            ax.clear()
            ax.view_init(elev=-86, azim=89)

            ax.set_xlim([mines[0], maxes[0]])
            ax.set_xlabel('$X$')
            ax.set_xticks([mines[0], (mines[0] + maxes[0]) / 2, maxes[0]])

            ax.set_ylim([mines[1], maxes[1]])
            ax.set_ylabel('$Y$')
            ax.set_yticks([mines[1], (mines[1] + maxes[1]) / 2, maxes[1]])

            ax.set_zlim([mines[2], maxes[2]])
            ax.set_zlabel('$Z$')
            ax.set_zticks([mines[2], (mines[2] + maxes[2]) / 2, maxes[2]])
        
        def animate(skeleton):
            raw_skeleton, processed_skeleton = np.array_split(skeleton, 2, axis=0)
            raw_skeleton, processed_skeleton = raw_skeleton[0], processed_skeleton[0]

            frame_index = skeleton_index[0]
            _pre_setting_ax(axl, raw_maxes, raw_mines)
            _pre_setting_ax(axr, processed_maxes, processed_mines)
            axl.title.set_text('RAW')
            axr.title.set_text('PROCESSED')
            plt.suptitle(f'Skeleton {index} Frame #{frame_index} of {T} (label: {label})')
            
            for e in Skeleton._one_direction_edges.T:
                rjoint_locs = raw_skeleton[e]
                pjoint_locs = processed_skeleton[e]

                # plot them
                axl.plot(rjoint_locs[:, 0],rjoint_locs[:, 1],rjoint_locs[:, 2], color='blue')
                axr.plot(pjoint_locs[:, 0],pjoint_locs[:, 1],pjoint_locs[:, 2], color='green')
            
            skeleton_index[0] += 1
            return axl, axr

        for index, (raw_skeleton, proc_skeleton, name, label) in enumerate(zip(raw, processed, names, labels)):
            print(f'@@@ Start Visualization: name {name}, Label: {label} @@@', flush=True)
            
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            axl = fig.add_subplot(1, 2, 1, projection='3d')
            axr = fig.add_subplot(1, 2, 2, projection='3d')

            raw_maxes, raw_mines = raw_skeleton.max((0, 1)), raw_skeleton.min((0, 1))
            processed_maxes, processed_mines = proc_skeleton.max((0, 1)), proc_skeleton.min((0, 1))

            _pre_setting_ax(axl, raw_maxes, raw_mines)
            _pre_setting_ax(axr, processed_maxes, processed_mines)

            skeleton_index = [0]
            skeleton = np.stack([raw_skeleton, proc_skeleton], axis=1)
            ani = FuncAnimation(fig, animate, skeleton)
            
            # saving to m4 using ffmpeg writer
            writervideo = animation.FFMpegWriter(fps=30)
            save_dir = os.path.join(work_dir, 'visualization', label)
            os.makedirs(save_dir, exist_ok=True)
            ani.save(os.path.join(save_dir, f'{name}.mp4'), writer=writervideo)

            print(f"@@@Visualization done for {name}@@@", flush=True)

    return visualize_samples

def get_parser():
    parser = ArgumentParser(description='GAIT Skeleton Visualizer.')
    parser.add_argument('-L', '--load-path', type=str, help='Path to load and process (if not exist) the raw data')
    parser.add_argument('-S', '--save-dir', type=str, help='directory to save the processed file')
    parser.add_argument('-F', '--filename', type=str, help='filename to save the processed data.')
    parser.add_argument('-S2', '--save-dir2', type=str, help='directory to save (2nd) the processed file. This is suitable to compare processed (2nd) with raw data (1st).')
    parser.add_argument('-F2', '--filename2', type=str, help='filename to save (2nd) the processed data.')
    parser.add_argument('-D', '--save-vis-dir', type=str, help='directory to save 3D visualizations')
    args = parser.parse_args()
    return args

# @stdout_stderr_setter("./visualize_logs/")
def run_main():
    import warnings
    warnings.filterwarnings("ignore")

    args = get_parser()

    def _read_data(save_dir: str, filename: str, critical_limit: int, non_critical_limit: int):
        save_path = os.path.join(save_dir, filename)
        if not os.path.exists(save_path):
            proc_gait_data(args.load_path, save_dir, filename=filename,
                config=ProcessingGaitConfig(fillZ_empty=False, 
                    preprocessing_conf=PreprocessingConfig(critical_limit=critical_limit, non_critical_limit=non_critical_limit))
            )
        with open(save_path, "rb") as f:
            import pickle
            data, labels, names, hard_cases_id = pickle.load(f)
            if data.ndim == 3:
                data = np.stack(np.array_split(data, 25, axis=2), axis=2)
                data = np.concatenate((data, np.zeros((*data.shape[:3], 1))), axis=3)

            # Fill NaN location same as the center of body
            idx1, idx2, idx3, _ = np.nonzero(np.isnan(data))
            data[idx1, idx2, idx3] = data[idx1, idx2, [Skeleton.CENTER]]

        return data, labels, names, hard_cases_id
    
    raw, labels, names1, _ = _read_data(args.save_dir, args.filename, critical_limit = 120, non_critical_limit = None)
    processed, _, names2, _ = _read_data(args.save_dir2, args.filename2, critical_limit = None, non_critical_limit = None)

    mask1 = np.zeros(raw.shape[0], dtype=np.bool)
    mask2 = np.zeros(raw.shape[0], dtype=np.bool)

    for i1, n1 in enumerate(names1):
        for i2, n2 in enumerate(names2):
            if n2 == n1:
                mask1[i1] = True
                mask2[i2] = True
                break
    
    raw = raw[mask1]
    processed = processed[mask2]
    names = names1[mask1]

    visualizer = initializer(args.save_vis_dir)
    visualizer(raw, processed, labels, names)

if __name__ == "__main__":
    run_main()