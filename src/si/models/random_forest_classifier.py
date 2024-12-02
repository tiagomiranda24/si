import numpy as np
from typing import List, Tuple, Literal
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy

class RandomForestClassifier:
    """
    Random Forest Classifier.
    """

    def __init__(self, 
                 n_estimators: int, 
                 max_features: int = None, 
                 min_sample_split: int = 2, 
                 max_depth: int = None, 
                 mode: Literal['gini', 'entropy'] = 'gini', 
                 random_seed: int = 1) -> None:
        """
        Initialize the RandomForestClassifier.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_features : int, optional
            Number of features to consider when looking for the best split.
        min_sample_split : int, optional
            Minimum number of samples required to split an internal node.
        max_depth : int, optional
            Maximum depth of the tree.
        mode : Literal['gini', 'entropy'], optional
            The mode to use for splitting (e.g., 'gini' or 'entropy').
        random_seed : int, optional
            Seed for random number generator.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.random_seed = random_seed
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train the RandomForestClassifier.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on.

        Returns
        -------
        self : RandomForestClassifier
            The fitted RandomForestClassifier.
        """
        np.random.seed(self.random_seed)

        if self.max_features == None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))

        for i in range(self.n_estimators):
            # Create a bootstrap dataset by sampling examples with replacement
            indices = np.random.choice(dataset.X.shape[0], size = dataset.shape()[0], replace = True)
            X_bootstrap = dataset.X[indices]
            y_bootstrap = dataset.y[indices]

            # Randomly select a subset of features without replacement
            feature_indices = np.random.choice(dataset.X.shape[1], size=self.max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]

            # Train a decision tree on the bootstrap dataset
            tree = DecisionTreeClassifier(max_depth=self.max_depth, mode=self.mode)
            tree.fit(Dataset(X_bootstrap, y_bootstrap))
            
            # Append a tuple containing the features used and the trained tree
            self.trees.append((feature_indices, tree))

        return self

    def _most_common(self, sample_predictions: np.ndarray) -> int:
        """
        Find the most common value in an array of predictions.

        Parameters
        ----------
        sample_predictions : np.ndarray
            Array of predictions.

        Returns
        -------
        int
            The most common prediction.
        """
        unique_classes, counts = np.unique(sample_predictions, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict.

        Returns
        -------
        predictions : np.ndarray
            The predicted class labels.
        """
        all_predictions = []

        for feature_indices, tree in self.trees:
            X = dataset.X[:, feature_indices]
            all_predictions.append(tree.predict(Dataset(X)))

        all_predictions = np.array(all_predictions)
        return np.apply_along_axis(self._most_common, axis=0, arr=all_predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Compute the accuracy of the model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on.

        Returns
        -------
        accuracy : float
            The accuracy score.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


# Example usage
if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    # Load dataset
    path = r"C:\Users\tiago\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv"
    data = read_csv(path, sep=",", features=True, label=True)

    # Split dataset
    train, test = train_test_split(data, test_size=0.33, random_state=42)

    # Train RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100, 
        max_features=4, 
        min_sample_split=2, 
        max_depth=5, 
        mode='gini', 
        random_seed=42
    )
    model.fit(train)

    # Evaluate model
    print(f"Model accuracy: {model.score(test):.2f}")