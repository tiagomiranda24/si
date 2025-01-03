import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegression(Model):
    """
    The LassoRegression is a linear model using L1 regularization.
    This model solves the linear regression problem using the coordinate descent technique.

    Parameters
    ----------
    l1_penalty: float
        The L1 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
    mean: np.array
        The mean of the dataset (for every feature)
    std: np.array
        The standard deviation of the dataset (for every feature)
    """

    def __init__(self, l1_penalty: float = 1.0, scale: bool = True):
        """
        Parameters
        ----------
        l1_penalty: float
            The L1 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset, max_iter=1000, tol=1e-4) -> 'LassoRegression':
        """
        Fit the model to the dataset using coordinate descent.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        max_iter: int
            Maximum number of iterations for the coordinate descent.

        tol: float
            Tolerance for convergence.

        Returns
        -------
        self: LassoRegression
            The fitted model
        """
        X = dataset.X
        y = dataset.y

        # Scale the data if required
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            X = (X - self.mean) / self.std

        m, n = X.shape

        # Initialize model parameters
        self.theta = np.zeros(n)
        self.theta_zero = np.mean(y)

        for iteration in range(max_iter):
            theta_prev = self.theta.copy()

            # Update each theta_j
            for j in range(n):
                X_j = X[:, j]
                residual = y - (self.theta_zero + np.dot(X, self.theta) - X_j * self.theta[j])

                # Compute the rho value
                rho = np.dot(X_j, residual)

                # Apply soft thresholding for L1 penalty
                if rho < -self.l1_penalty / 2:
                    self.theta[j] = (rho + self.l1_penalty / 2) / np.sum(X_j ** 2)
                elif rho > self.l1_penalty / 2:
                    self.theta[j] = (rho - self.l1_penalty / 2) / np.sum(X_j ** 2)
                else:
                    self.theta[j] = 0.0

            # Update theta_zero
            self.theta_zero = np.mean(y - np.dot(X, self.theta))

            # Check for convergence
            if np.linalg.norm(self.theta - theta_prev, ord=2) < tol:
                print(f"Converged after {iteration} iterations.")
                break
        else:
            print(f"Did not converge after {max_iter} iterations.")

        return self

    def _predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output for the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of.

        Returns
        -------
        predictions: np.array
            The predictions of the dataset.
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error (MSE) of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on.

        Returns
        -------
        mse: float
            The Mean Square Error of the model.
        """
        predictions = self._predict(dataset)
        return mse(dataset.y, predictions)

if __name__ == '__main__':
    # Create a sample dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset = Dataset(X=X, y=y)

    # Instantiate and fit the Lasso Regression model
    lasso = LassoRegression(l1_penalty=0.1)
    lasso._fit(dataset)

    # Get the coefficients
    print(f"Theta: {lasso.theta}")
    print(f"Theta zero: {lasso.theta_zero}")

    # Predict on the dataset
    predictions = lasso._predict(dataset)
    print(f"Predictions: {predictions}")

    # Compute the score
    mse_score = lasso._score(dataset)
    print(f"MSE: {mse_score}")