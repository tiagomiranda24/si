import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean square error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        Real values of y
    y_pred: np.ndarray
        Predicted values of y

    Returns
    -------
    accuracy: float
        The error value between y_trueand y_pred
    """
    return (np.sum((y_true - y_pred)**2))/len(y_true)
