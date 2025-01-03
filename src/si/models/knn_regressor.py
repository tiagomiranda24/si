from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
from si.model_selection.split import train_test_split

class KNNRegressor(Model):
    """
    KNN Regressor

    The algorithm K-Nearest Neighbors Regressor is used for regression tasks. 
    It predicts the value of a target variable for a given input by looking at the values of the 
    K nearest neighbors (data points) in the training set. Specifically, it identifies the 
    K data points in the training set that are closest to the input data point based on a 
    distance metric (like Euclidean distance) and then calculates the average (or sometimes 
    a weighted average) of their target values to make the prediction. This allows the model 
    to estimate a continuous value based on the proximity of similar data points.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use (that calculates the distance between a sample and the samples
        in the training dataset).

    Attributes
    ----------
    dataset: Dataset
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None


    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self


    def _get_closest_values(self, sample: np.ndarray) -> np.ndarray:
        """
        Returns the k-nearest target values for a given sample.

        Parameters
        ----------
        sample : np.ndarray
            The sample to get the closest target values for.

        Returns
        -------
        closest_values : np.ndarray
            The k nearest target values corresponding to the sample.
        """
        # Ensure the sample is at least 2D for consistency in distance calculation
        sample = np.atleast_2d(sample)

        # Compute distances between the sample and all points in the dataset
        distances = self.distance(sample, self.dataset.X)

        # Get the indices of the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # Return the target values (y) corresponding to the k nearest neighbors
        return self.dataset.y[k_nearest_neighbors]
    

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the values for the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the values for

        Returns
        -------
        predictions: np.ndarray
            The predicted values for the dataset
        """
        # For each sample in the dataset, get the k-nearest target values and calculate the mean
        predictions = np.apply_along_axis(lambda sample: np.mean(self._get_closest_values(sample)), axis=1, arr=dataset.X)
        return predictions


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the RMSE (Root Mean Squared Error) of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        
        predictions: np.ndarray
            The predicted values

        Returns
        -------
        rmse: float
            The RMSE of the model
        """
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # Load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # Initialize the KNN regressor
    knn_regressor = KNNRegressor(k=3)

    # Fit the model to the train dataset
    knn_regressor.fit(dataset_train)

    # Predict the values for the test dataset
    predictions = knn_regressor.predict(dataset_test)

    # Evaluate the model using RMSE
    rmse_score = knn_regressor.score(dataset_test)
    print(f'The RMSE of the model is: {rmse_score}')