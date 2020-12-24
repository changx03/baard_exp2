"""
Region-based classification using PyTorch.
"""

def search_r(model, X, y, r0=0 , step_size=0.01):
    r = r0

    return r - step_size