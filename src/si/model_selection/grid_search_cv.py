from typing import Dict, List
from si.data.dataset import Dataset
import itertools

def grid_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, List], scoring: str, cv: int) -> Dict[str, List[float]]:
    """
    Perform grid search cross-validation to find the best hyperparameters for a given model.

    Parameters
    ----------
    model:
        Model to validate.
    dataset: Dataset
        The dataset to validate the model.
    hyperparameter_grid: Dict[str, List]
        A dictionary where keys are hyperparameter names and values are lists of values to search.
    scoring: str
        The scoring metric to evaluate the model (e.g., 'mean_squared_error').
    cv: int
        The number of folds.

    Returns
    -------
    results: Dict[str, List[float]]
        A dictionary where the keys are hyperparameter combinations and values are lists of scores for each fold.
    """

for hyperparameter in hyperparameter_grid:
    
    if not hasattr(model, hyperparameter):
        raise ValueError(f"Model does not have hyperparameter {hyperparameter}")
 
# Gerar todas as combinações possíveis de hiperparâmetros
combinations = list(itertools.product(*hyperparameter_grid.values()))


for combination in combinations:
    
    for parameter in zip(hiperparameter_grid.keys ,combination)