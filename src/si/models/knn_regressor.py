from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

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
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # Parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # Attributes
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

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        return self._fit(dataset)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the target variable for the given test dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to predict the target variable

        Returns
        -------
        predictions: np.ndarray
            An array of predicted values for the testing dataset
        """
        # Initialize an array to hold the predicted values
        predictions = np.zeros(dataset.X.shape[0]) 

        # Calculate the distance between each sample in the test dataset and the training dataset
        for i in range(dataset.X.shape[0]):
            distances = np.zeros(self.dataset.X.shape[0])

            # Calculate the distance between the test sample i and all training samples
            for j in range(self.dataset.X.shape[0]):
                distances[j] = self.distance(dataset.X[i], self.dataset.X[j])

            # Obtain the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Retrieve the corresponding values in Y using the indices
            k_nearest_values = self.dataset.Y[k_indices]

            # Calculate the average of the values
            predictions[i] = np.mean(k_nearest_values)

        return predictions  # Return predictions for the test dataset

    def score(self, dataset: Dataset) -> float:
        """
        It returns the Root Mean Squared Error of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        
        Returns
        -------
        rmse: float
            The error between predictions and actual values
        """
        predictions = self._predict(dataset)  
        return rmse(dataset.Y, predictions)


if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN Regressor
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The Root Mean Squared Error of the model is: {score}')