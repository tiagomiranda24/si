import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    An ensemble classifier that combines multiple models and uses majority voting 
    to predict class labels.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        A list of different models to be included in the ensemble.

    Attributes
    ----------
    models : array-like
        The individual models used in the ensemble.
    """
    def __init__(self, models, **kwargs):
        """
        Initialize the StackingClassifier with the given models.

        Parameters
        ----------
        models : array-like, shape = [n_models]
            A list of models that will be trained in the ensemble.

        kwargs : additional keyword arguments
            Any additional parameters to be passed to the parent Model class.
        """
        super().__init__(**kwargs)  # Initialize the base Model class
        self.models = models  # Store the list of models for stacking

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit each model in the ensemble to the training data.

        Parameters
        ----------
        dataset : Dataset
            The training data used to fit the models.

        Returns
        -------
        self : StackingClassifier
            Returns the fitted StackingClassifier instance.
        """
        for model in self.models:
            model.fit(dataset)  # Fit each model on the training data

        return self  # Return the fitted StackingClassifier

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for the samples in the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The test data for which predictions are to be made.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels for the input data.
        """
        
        # Helper function to compute the majority vote from predictions
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            Calculate the majority vote from an array of predictions.

            Parameters
            ----------
            pred: np.ndarray
                An array of predictions to determine the majority vote.

            Returns
            -------
            majority_vote: int
                The class label that received the most votes.
            """
            # Get unique labels and their counts
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]  # Return the label with the highest count

        # Collect predictions from each model and transpose the results
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # Apply the majority vote function along the rows of the predictions array
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the mean accuracy of the predictions compared to the true labels.

        Parameters
        ----------
        dataset : Dataset
            The test dataset containing the true labels.
        predictions: np.ndarray
            The predicted labels from the ensemble.

        Returns
        -------
        score : float
            The accuracy score of the predictions.
        """
        return accuracy(dataset.y, predictions)  # Calculate and return the accuracy


if __name__ == '__main__':
    # Import necessary components for dataset handling and model training
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression

    # Load a random dataset and split it into training and testing sets
    dataset_ = Dataset.from_random(600, 100, 2)  # Generate a random dataset with 600 samples
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)  # Split the dataset into 80% train and 20% test

    # Initialize KNN and Logistic Regression classifiers
    knn = KNNClassifier(k=3)  # Create a KNN classifier with k=3
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)  # Create a Logistic Regression model

    # Initialize the StackingClassifier with the individual models
    stacking = StackingClassifier([knn, lg])

    # Fit the StackingClassifier on the training dataset
    stacking.fit(dataset_train)

    # Compute and print the score (accuracy) of the model on the test set
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    # Print the predictions for the test dataset
    print(stacking.predict(dataset_test))