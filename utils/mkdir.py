import os
from pathlib import Path


def mkdir(file_name, verbose=True):
    """Make directories with parent (mkdir -p <file_name>)"""
    if not os.path.exists(file_name):
        path = Path(file_name)
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print('Create directory:', file_name)
