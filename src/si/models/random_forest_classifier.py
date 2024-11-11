import numpy as np
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy


class RandomForestClassifier:
    """
    Random Forest Classifier.
    """
    def __init__(self, n_estimators: int, max_features: int = None, min_sample_split: int = 2, max_depth: int = None, mode: str = 'gini', random_seed: int = 1) -> None:
        """
        Constructor for RandomForestClassifier class.

        Parameters
        ----------
        n_estimators: int
            Number of trees in the forest.
        max_features: int
            Number of features to consider when looking for the best split.
        min_sample_split: int
            Minimum number of samples required to split an internal node.
        max_depth: int
            Maximum depth of the tree.
        mode: str
            The mode to use (e.g., 'gini' or 'entropy').
        random_seed: int
            The seed of the random number generator.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.random_seed = random_seed
        self.trees = []

    def fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Fit the RandomForestClassifier class.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: RandomForestClassifier
            The fitted RandomForestClassifier.
        """
        np.random.seed(self.random_seed)

        # Set max_features if None
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))

        for _ in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = np.random.choice(dataset.X.shape[0], size=dataset.X.shape[0], replace=True)
            X_bootstrap = dataset.X[indices]
            y_bootstrap = dataset.y[indices]

            # Select random subset of features
            feature_indices = np.random.choice(dataset.X.shape[1], size=self.max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]

            # Train decision tree on the bootstrap dataset
            tree = DecisionTreeClassifier(max_depth=self.max_depth, mode=self.mode)
            tree.fit(Dataset(X_bootstrap, y_bootstrap))
            self.trees.append((feature_indices, tree))

        return self

    def most_common(self, sample_predictions):
        """
        Find the most common value in an array.

        Parameters
        ----------
        sample_predictions: np.ndarray
            Array of predictions.

        Returns
        -------
        np.ndarray
            The most common prediction in the array.
        """
        unique_classes, counts = np.unique(sample_predictions, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the values for a given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict.

        Returns
        -------
        predictions: np.ndarray
            The predicted values.
        """
        all_predictions = []

        for feature_indices, tree in self.trees:
            X = dataset.X[:, feature_indices]
            all_predictions.append(tree.predict(Dataset(X)))

        all_predictions = np.array(all_predictions)
        return np.apply_along_axis(self.most_common, axis=0, arr=all_predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy of the model on a given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        score: float
            The accuracy of the model.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


# Example usage
if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    path = r"C:\Users\tiago\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv"
    data = read_csv(path, sep=",", features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_features=4, min_sample_split=2, max_depth=5, mode='gini', random_seed=42)
    model.fit(train)
    print(model.score(test))
