import numpy as np

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray
        Input value(s).

    Returns
    ----------
    np.ndarray
        The probability of the values being 1 (sigmoid function).
    """
    return 1 / (1 + np.exp(-x))