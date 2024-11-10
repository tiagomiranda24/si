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
    def __init__(self, feature_idx: int = None, threshold: float = None, left: 'Node' = None, 
                 right: 'Node' = None, info_gain: float = None, value: Union[float, str] = None) -> None:
        """
        Creates a Node object.
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

    def predict(self, x: np.ndarray) -> Union[float, str]:
        """
        Predicts the label for a single sample by traversing the tree.
        """
        if self.value is not None:  # Leaf node
            return self.value
        # Decision node: traverse left or right
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

class RandomForestClassifier(Model):
    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2, 
                 max_depth: int = None, mode: Literal['gini', 'entropy'] = 'gini', seed: int = None) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees: List[Tuple[np.ndarray, Node]] = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        np.random.seed(self.seed)
        n_samples, n_features = dataset.X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_samples = dataset.X[indices]
            bootstrap_labels = dataset.y[indices]
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            bootstrap_samples = bootstrap_samples[:, feature_indices]
            tree = self._create_decision_tree(Dataset(bootstrap_samples, bootstrap_labels))
            self.trees.append((feature_indices, tree))
        return self

    def _create_decision_tree(self, dataset: Dataset) -> Node:
        return self._train_decision_tree(dataset)

    def _train_decision_tree(self, dataset: Dataset, depth: int = 0) -> Node:
        """
        Trains a decision tree on the given dataset.
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to train the decision tree.
        
        Returns
        -------
        Node
            The root node of the trained decision tree.
        """
        # Obtenha os rótulos do conjunto de dados
        labels = dataset.y
        
        # Verifica se os rótulos contêm valores negativos ou não são inteiros
        if labels.min() < 0 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be non-negative integers for np.bincount.")
        
        # Calcula a distribuição de rótulos
        labels = labels.astype(int)
        
        # Define um nó folha se não for possível dividir mais
        if len(set(labels)) == 1 or len(labels) < self.min_sample_split or (self.max_depth and depth >= self.max_depth):
            return Node(value=np.argmax(np.bincount(labels)))

    def _calculate_info_gain(self, parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
        if self.mode == 'gini':
            return gini_impurity(parent) - (len(left_child) / len(parent) * gini_impurity(left_child) +
                                            len(right_child) / len(parent) * gini_impurity(right_child))
        elif self.mode == 'entropy':
            return entropy_impurity(parent) - (len(left_child) / len(parent) * entropy_impurity(left_child) +
                                               len(right_child) / len(parent) * entropy_impurity(right_child))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            tree_preds = [tree.predict(x[feature_indices]) for feature_indices, tree in self.trees]
            predictions.append(np.bincount(tree_preds).argmax())
        return np.array(predictions)

    def _score(self, dataset: Dataset) -> float:
        predictions = self._predict(dataset.X)
        return accuracy(predictions, dataset.y)
