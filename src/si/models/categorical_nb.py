import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset

class CategoricalNB(Model):
    """
    Categorical Naive Bayes Classifier

    This classifier is designed for binary features (0 or 1).
    It uses Laplace smoothing to avoid zero probabilities in feature counts.

    Parameters
    ----------
    smoothing : float, default=1.0
        Laplace smoothing parameter to avoid zero probabilities.

    Attributes
    ----------
    class_prior : np.ndarray
        Prior probabilities for each class.
    feature_probs : np.ndarray
        Probabilities of each feature for each class.
    """

    def __init__(self, smoothing: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None

    def _fit(self, dataset: Dataset) -> 'CategoricalNB':
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : CategoricalNB
            The fitted model.
        """
        X, y = dataset.X, dataset.y
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize class counts and feature counts
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))

        # Calculate class counts and feature counts
        for idx, class_label in enumerate(self.classes):
            class_samples = X[y == class_label]
            class_counts[idx] = class_samples.shape[0]
            feature_counts[idx, :] = np.sum(class_samples, axis=0)

        # Calculate class prior probabilities (P(C))
        self.class_prior = (class_counts + self.smoothing) / (n_samples + n_classes * self.smoothing)

        # Apply Laplace smoothing to feature counts and compute feature probabilities (P(F|C))
        self.feature_probs = (feature_counts + self.smoothing) / (class_counts[:, None] + 2 * self.smoothing)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for a given test dataset.

        Parameters
        ----------
        dataset : Dataset
            The test dataset.

        Returns
        -------
        predictions : np.ndarray
            An array of predicted class labels.
        """
        X = dataset.X
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        # Calculate the log probability for each class
        for idx, class_label in enumerate(self.classes):
            # Log probabilities to avoid underflow
            log_prob_class = np.log(self.class_prior[idx])
            log_prob_features = X @ np.log(self.feature_probs[idx]) + (1 - X) @ np.log(1 - self.feature_probs[idx])
            log_probs[:, idx] = log_prob_class + log_prob_features

        # Predict the class with the highest probability
        y_pred = self.classes[np.argmax(log_probs, axis=1)]

        return y_pred

    def _score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy of the model on the given test dataset.

        Parameters
        ----------
        dataset : Dataset
            The test dataset.

        Returns
        -------
        accuracy : float
            The accuracy of the predictions.
        """
        y_pred = self._predict(dataset)
        accuracy = np.mean(y_pred == dataset.y)
        return accuracy


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    # Load the dataset (make sure the file path is correct)
    dataset = read_csv('path', features=True, label=True)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    # Initialize and fit the CategoricalNB model
    nb = CategoricalNB(smoothing=1.0)
    nb._fit(train_dataset)

    # Predict and score the model
    predictions = nb._predict(test_dataset)
    accuracy = nb._score(test_dataset)

    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")