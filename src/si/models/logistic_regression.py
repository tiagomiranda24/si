import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    The LogisticRegression is a linear model for binary classification using the sigmoid function.
    This model solves the classification problem using an adapted Gradient Descent technique.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter to prevent overfitting.
    alpha: float
        The learning rate for gradient descent.
    max_iter: int
        The maximum number of iterations for gradient descent.
    patience: int
        The number of iterations without improvement before stopping the training early.
    scale: bool
        Whether to standardize the dataset (zero mean, unit variance) before training.

    Attributes
    ----------
    theta: np.ndarray
        The model parameters, namely the coefficients of the logistic model.
    theta_zero: float
        The intercept of the logistic model.
    mean: np.ndarray
        The mean values used for scaling the dataset (if scaling is enabled).
    std: np.ndarray
        The standard deviation values used for scaling the dataset (if scaling is enabled).
    cost_history: dict
        A dictionary to store the history of the cost function during training.
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5,
                 scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        self: LogisticRegression
            The fitted model.
        """
        if self.scale:
            # Compute mean and std for scaling
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # Scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # Get dimensions of X
        m, n = X.shape

        # Initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0
        # Gradient descent
        while i < self.max_iter and early_stopping < self.patience:
            # Predicted y using the sigmoid function
            linear_combination = np.dot(X, self.theta) + self.theta_zero
            y_pred = sigmoid_function(linear_combination)

            # Compute the gradient of the loss function with respect to theta
            gradient = (1 / m) * np.dot(X.T, (y_pred - dataset.y)) + (self.l2_penalty / m) * self.theta

            # Compute the gradient for theta_zero
            theta_zero_gradient = (1 / m) * np.sum(y_pred - dataset.y)

            # Update the model parameters
            self.theta -= self.alpha * gradient
            self.theta_zero -= self.alpha * theta_zero_gradient

            # Compute the cost (e.g., log-loss for logistic regression)
            self.cost_history[i] = self.cost(dataset)
            
            # Early stopping condition based on cost improvement
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            i += 1

        return self

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the logistic regression loss (log-loss).

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the loss on.

        Returns
        -------
        loss: float
            The computed log-loss value.
        """
        # Compute the predicted probabilities
        linear_combination = np.dot(dataset.X, self.theta) + self.theta_zero
        y_pred = sigmoid_function(linear_combination)

        # Compute log-loss
        loss = -np.mean(dataset.y * np.log(y_pred + 1e-15) + (1 - dataset.y) * np.log(1 - y_pred + 1e-15))
        # Add L2 penalty term
        loss += (self.l2_penalty / (2 * len(dataset.y))) * np.sum(np.square(self.theta))
        return loss

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the output of the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of.

        Returns
        -------
        predictions: np.ndarray
            The predictions of the dataset (0 or 1).
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        linear_combination = np.dot(X, self.theta) + self.theta_zero
        y_pred = sigmoid_function(linear_combination)
        return (y_pred >= 0.5).astype(int)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the accuracy of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the accuracy on.
        predictions: np.ndarray
            Predictions made by the model.

        Returns
        -------
        accuracy: float
            The accuracy of the model.
        """
        return accuracy(dataset.y, predictions)
