import os
from typing_extensions import Protocol

import matplotlib as mpl
import matplotlib.cm as cm
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
        T = data.shape[1]
        def animate(skeleton):
            frame_index = skeleton_index[0]
            ax.clear()
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-1,1])

            for i, j in Skeleton.edges:
                joint_locs = skeleton[[i,j]]
                
                # plot them
                ax.plot(joint_locs[0],joint_locs[1],joint_locs[2], color='blue')
            
            plt.title(f'Skeleton {index} Frame #{frame_index} of {T}\n (label: {label})')
            skeleton_index[0] += 1

            return ax

        for index, (skeleton, name, label) in enumerate(zip(data, names, labels)):
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-1,1])

            # get data

            print(f'Sample name: {name}\nLabel: {label}\n')   # (C,T,V,M)

            skeleton_index = [0]
            skeleton_frames = skeleton.transpose(1,0,2)
            ani = FuncAnimation(fig, animate, skeleton_frames)
            
            # # saving to m4 using ffmpeg writer
            # writervideo = animation.FFMpegWriter(fps=60)
            # save_dir = os.path.join(work_dir, 'visualization', label)
            # os.makedirs(save_dir, exist_ok=True)
            # ani.save(os.path.join(save_dir, f'{name}.mp4'), writer=writervideo)
            plt.show()

            print(f"@@@\nVisualization done for {name}\n@@@", flush=True)

    return visualize_samples


if __name__ == "__main__":
    proc_gait_data("../../Data/output_1.pkl", "../../Data")
    with open("../../Data/processed.pkl", "rb") as f:
        import pickle
        data, labels, names = pickle.load(f)
    
    # visualizer = initializer("./basic_vis/")
    # visualizer(data, labels, names)