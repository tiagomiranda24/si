from typing import List
import numpy as np
from si.data.dataset import Dataset

def k_fold_cross_validation(model, dataset: Dataset, scoring: str, cv: int, random_state: int) -> List[float]:
    """
    Split the dataset into training and testing sets and perform k-fold cross-validation.

    Parameters
    ----------
    model:
        Model to validate.
    dataset: Dataset
        The dataset to validate the model.
    scoring: str
        The scoring metric to evaluate the model (e.g., 'mean_squared_error').
    cv: int
        The number of folds.
    random_state: int
        Random seed for reproducibility.

    Returns
    -------
    scores: List[float]
        A list of scores for each fold.
    """
    dataset_size = dataset.shape[0]  
    fold_size = dataset_size // cv
    np.random.seed(random_state)
    indices = np.random.permutation(dataset_size)  # Randomly shuffle indices

    scores = []
    for i in range(cv):
        # Split indices for training and testing
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        # Train the model on the training set
        model.fit(dataset.X[train_indices], dataset.y[train_indices])

        # Evaluate the model on the test set using the specified scoring metric
        score = model.score(dataset.X[test_indices], dataset.y[test_indices], scoring=scoring)

        # Append the score for this fold
        scores.append(score)

    return scores
