from typing import Literal, Tuple, Union, List
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.impurity import gini_impurity, entropy_impurity

class Node:
    """
    Class representing a node in a decision tree.
    """

    def __init__(self, feature_idx: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None,
                 info_gain: float = None, value: Union[float, str] = None) -> None:
        """
        Creates a Node object.

        Parameters
        ----------
        feature_idx: int
            The index of the feature to split on.
        threshold: float
            The threshold value to split on.
        left: Node
            The left subtree.
        right: Node
            The right subtree.
        info_gain: float
            The information gain.
        value: Union[float, str]
            The value of the leaf node.
        """
        # for decision nodes
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf nodes
        self.value = value

class RandomForestClassifier(Model):
    """
    Class representing a random forest classifier.
    """

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_features: int = None, 
                 min_sample_split: int = 2, 
                 max_depth: int = None, 
                 mode: Literal['gini', 'entropy'] = 'gini', 
                 seed: int = None) -> None:
        """
        Creates a RandomForestClassifier object.

        Parameters
        ----------
        n_estimators: int
            Number of decision trees to use.
        max_features: int
            Maximum number of features to use per tree.
        min_sample_split: int
            Minimum number of samples required to split an internal node.
        max_depth: int
            Maximum depth of the trees.
        mode: Literal['gini', 'entropy']
            The mode to use for calculating the information gain.
        seed: int
            Random seed to use to ensure reproducibility.
        
        Attributes
        ----------
        trees: List[Node]
            The trees of the random forest and respective features used for training (initialized as an empty list).
        """
        # parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        
        # estimated parameters
        self.trees: List[Tuple[np.ndarray, Node]] = []  # Initialize trees as an empty list

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Trains the decision trees of the random forest.

        Parameters
        ----------
        dataset: Dataset
            The dataset to train the random forest classifier on.
        
        Returns
        -------
        RandomForestClassifier
            The fitted RandomForestClassifier instance.
        """
        np.random.seed(self.seed)
        n_samples, n_features = dataset.X.shape
        
        # Set max_features to sqrt(n_features) if None
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        
        for _ in range(self.n_estimators):
            # Create bootstrap dataset
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_samples = dataset.X[indices]
            bootstrap_labels = dataset.y[indices]

            # Select max_features random features without replacement
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            bootstrap_samples = bootstrap_samples[:, feature_indices]  # Filter the features
            
            # Create and train a decision tree with the bootstrap dataset
            tree = self._create_decision_tree(Dataset(bootstrap_samples, bootstrap_labels))
            self.trees.append((feature_indices, tree))  # Append the features used and the trained tree
        
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels using the ensemble models.

        Parameters
        ----------
        X: np.ndarray
            The feature data for which predictions are to be made.

        Returns
        -------
        np.ndarray
            The predicted labels for the input data.
        """
        predictions = np.array([tree.predict(X) for _, tree in self.trees])  # Get predictions for each tree
        # Get the most common predicted class for each sample
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])

    def _score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the true labels.

        Returns
        -------
        float
            The accuracy of the model on the provided dataset.
        """
        predictions = self._predict(dataset.X)
        return accuracy(predictions, dataset.y)