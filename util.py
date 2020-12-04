import numpy as np

def get_range(data):
    """Get column-wise min and max values"""
    x_max = np.max(data, axis=0)
    x_min = np.min(data, axis=0)
    return (x_min, x_max)

def normalize(data, xmin, xmax):
    """Scale the data to [0, 1]"""
    return (data - xmin) / (xmax - xmin)

def unnormalize(data, xmin, xmax):
    """Rescale the data to normal range"""
    return data * (xmax - xmin) + xmin
