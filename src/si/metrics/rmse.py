import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the Root Mean Square Error of the model on the given dataset.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset.
    y_pred: np.ndarray
        The predicted labels of the dataset.

    Returns
    -------
    rmse: float
        The Root Mean Square Error of the model.
    """
    
    # Convert to numpy arrays to ensure calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the squared differences between true and predicted values
    error = (y_true - y_pred) ** 2
    
    # Compute the mean of squared errors
    mean_error = np.mean(error)
    
    # Calculate the square root of the mean error
    rmse_value = np.sqrt(mean_error)
    
    return rmse_value