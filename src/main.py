from importlib import import_module
import random
from sys import argv
from typing import TYPE_CHECKING

import numpy as np
import torch

from STPGait.context import Constants
from STPGait.utils.decorators import stdout_stderr_setter
if TYPE_CHECKING:
    from STPGait.entrypoints.core import MainEntrypoint


def global_seed(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.
    Args:
        seed (int): The desired seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_main(entrypoint: 'MainEntrypoint') -> None:
    print(f"%%% RUNNING SEED {Constants.GLOBAL_SEED} %%%", flush=True)
    entrypoint.run()

if __name__ == "__main__":
    Constants.GLOBAL_SEED = int(argv[1])
    
    global_seed(Constants.GLOBAL_SEED)
    script = import_module(f"STPGait.entrypoints.{argv[2]}")
    entrypoint: 'MainEntrypoint' = getattr(script, 'Entrypoint')()

    if entrypoint.conf.save_log_in_file:
        run_main = stdout_stderr_setter(entrypoint.conf.save_dir)(run_main)
    run_main(entrypoint)