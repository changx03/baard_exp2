

def get_dataset_shape(dataset):
    """Retruns the shape of the data in a PyTorch Dataset object."""
    # shape = list(next(iter(dataset))[0].size())
    X, _ = next(iter(dataset))
    shape = list(X.size())
    shape = [len(dataset)] + shape
    return tuple(shape)
