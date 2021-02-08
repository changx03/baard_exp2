import random

import numpy as np
import torch


def set_seeds(random_state):
    random_state = int(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
