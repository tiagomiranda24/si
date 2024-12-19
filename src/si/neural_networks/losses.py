from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)
class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss function for multi-class classification.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels, one-hot encoded.
        y_pred: numpy.ndarray
            The predicted probabilities.

        Returns
        -------
        float
            The loss value.
        """
        # Clip predictions to avoid log(0) and division by zero.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute categorical cross-entropy loss.
        # For each sample, we calculate the loss using the formula for categorical cross-entropy.
        # The loss is the negative sum of the true labels multiplied by the log of the predicted probabilities.
        loss_value = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss_value

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the categorical cross-entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels, one-hot encoded.
        y_pred: numpy.ndarray
            The predicted probabilities.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function with respect to y_pred.
        """
        # Clip predictions to avoid division by zero and log(0).
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute the derivative: (y_pred - y_true) / N
        # This is the gradient used in backpropagation for multi-class classification.
        derivative_value = (y_pred - y_true) / y_true.shape[0]
        return derivative_value
