import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(x, min, max):
    """
    Return a value between 0 and 1, representing where it lies between min and max
    """
    return (x - min) / (max - min)
