from argparse import ArgumentParser
import os
from typing_extensions import Protocol

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np

from ..context import Skeleton
from ..data import proc_gait_data

class Func(Protocol):
    def __call__(self, data: np.ndarray, labels: np.ndarray, names: np.ndarray) -> None:
        ...

def initializer(work_dir: str) -> Func:
    
    os.makedirs(work_dir, exist_ok=True)

    def visualize_samples(data: np.ndarray, labels: np.ndarray, names: np.ndarray) -> None:
        # Swap Y-Z axis for just visualization
        data[..., [1, 2]] = data[..., [2, 1]]

        T = data.shape[1]
        maxes, mines = data.max((0, 1, 2)), data.min((0, 1, 2))

        def _pre_setting_ax():
            ax.clear()
            ax.view_init(elev=5, azim=-87)

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
            frame_index = skeleton_index[0]
            _pre_setting_ax()

            for i, j in Skeleton._one_direction_edges:
                joint_locs = skeleton[[i,j]]
                
                # plot them
                ax.plot(joint_locs[:, 0],joint_locs[:, 1],joint_locs[:, 2], color='green')

            plt.title(f'Skeleton {index} Frame #{frame_index} of {T}\n (label: {label})')
            skeleton_index[0] += 1

            return ax

        for index, (skeleton, name, label) in enumerate(zip(data, names, labels)):
            print(f'Sample name: {name}\nLabel: {label}\n')
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            _pre_setting_ax()

            skeleton_index = [0]
            ani = FuncAnimation(fig, animate, skeleton)
            
            # saving to m4 using ffmpeg writer
            writervideo = animation.FFMpegWriter(fps=30)
            save_dir = os.path.join(work_dir, 'visualization', label)
            os.makedirs(save_dir, exist_ok=True)
            ani.save(os.path.join(save_dir, f'{name}.mp4'), writer=writervideo)

            print(f"@@@\nVisualization done for {name}\n@@@", flush=True)

    return visualize_samples

def get_parser():
    parser = ArgumentParser(description='GAIT Skeleton Visualizer.')
    parser.add_argument('-L', '--load-path', type=str, help='Path to load and process (if not exist) the raw data')
    parser.add_argument('-S', '--save-dir', type=str, help='directory to save the processed file')
    parser.add_argument('-D', '--save-vis-dir', type=str, help='directory to save 3D visualizations')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()

    save_path = os.path.join(args.save_dir, "processed.pkl")
    if not os.path.exists(save_path):
        proc_gait_data(args.load_path, args.save_dir)
    
    with open(save_path, "rb") as f:
        import pickle
        data, labels, names, _ = pickle.load(f)

    visualizer = initializer(args.save_vis_dir)
    visualizer(data, labels, names)