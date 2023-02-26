from dataclasses import dataclass, field
import os
from typing import List, Optional, TYPE_CHECKING

from ..enums import Phase
if TYPE_CHECKING:
    from ..enums import Optim


@dataclass
class BaseConfig:
    try_num: int
    """ number of experiment """

    try_name: str
    """ name of experiment """

    training_config: 'TrainingConfig' 
    """ config of your training including optimizer, number of epochs and data shuffling """

    device: str
    """ device to run """

    save_log_in_file: bool
    """ if True then save logs in a file, O.W. in terminal """

    eval_batch_size: int
    """ Evaluation batch size """

    remove_labels: List[str] = field(default_factory=list)
    """ Labels to be removed """

    phase: Phase = Phase.TRAIN
    """ Phase to Run """
    
    @property
    def save_dir(self):
        """ the root location in which log files, visualization outputs and model weights are saved """
        
        folder_name = "_".join([str(self.try_num), self.try_name])
        root = "../results"
        return os.path.join(root, folder_name)

@dataclass
class TrainingConfig:
    num_epochs: int
    optim_type: 'Optim'
    lr: float = 3e-3
    shuffle_training: bool = True
    batch_size: int = 32

    early_stop: Optional[int] = None
    """ If given then the running will be stopped if accuracy has not been improved for that duration (epoch) """