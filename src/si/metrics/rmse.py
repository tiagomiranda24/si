import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true: np.ndarray or list
        The true labels of the dataset.
    y_pred: np.ndarray or list
        The predicted labels of the dataset.

    Returns
    -------
    float
        The Root Mean Square Error (RMSE).
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Compute RMSE by taking the square root of the MSE
    rmse_value = np.sqrt(mse)
    
    return rmse_value

if __name__ == '__main__':
    # Test data
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    
    # Compute RMSE
    rmse_value = rmse(y_true, y_pred)
    print("Calculated RMSE:", rmse_value)
    
    # Verify using explicit MSE and sqrt
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    expected_rmse = np.sqrt(mse)
    print("Expected RMSE:", expected_rmse)
    
    # Check correctness
    assert np.isclose(rmse_value, expected_rmse), "RMSE calculation is incorrect!"
