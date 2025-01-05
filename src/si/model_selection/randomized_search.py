import itertools
from typing import Callable, Tuple, Dict, Any
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation
from si.base.model import Model

def randomized_search_cv(model: Model, dataset: Dataset, hyperparameter_grid: Dict[str, Tuple], scoring: Callable = None,
    cv: int = 5, n_iter: int = 10) -> Dict[str, Any]:
    """
    Performs a randomized hyperparameter search using k-fold cross-validation.

    Arguments:
        - model: The model to be validated.
        - dataset: The dataset to be used for validation.
        - hyperparameter_grid: Dictionary where keys are hyperparameter names, 
          and values are tuples of possible values to search over.
        - scoring: Function to calculate the model score (optional, default is None).
        - cv: Number of folds for k-fold cross-validation (default is 5).
        - n_iter: Number of random hyperparameter combinations to test (default is 10).

    Returns:
        - A dictionary containing cross-validation results for each hyperparameter combination, 
          including the scores, hyperparameters, best hyperparameters, and the best score.
    """

    # Validate hyperparameter names
    for hyperparam in hyperparameter_grid.keys():
        if not hasattr(model, hyperparam):
            raise ValueError(f"The hyperparameter '{hyperparam}' is not valid for the provided model.")

    # Generate all possible combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*hyperparameter_grid.values()))

    # If n_iter is less than the total number of combinations, randomly sample combinations
    if n_iter < len(hyperparameter_combinations):
        sampled_combinations = np.random.choice(len(hyperparameter_combinations), n_iter, replace=False)
        # Select only the sampled combinations
        hyperparameter_combinations = [hyperparameter_combinations[i] for i in sampled_combinations]

    # List to store the results of the search
    results = []

    # Perform k-fold cross-validation for each combination of hyperparameters
    for combination in hyperparameter_combinations:
        # Map the current combination of hyperparameter values to their respective names
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))

        # Set the model parameters using the current hyperparameter combination
        for param, value in hyperparameters.items():
            if hasattr(model, param):
                setattr(model, param, value)
            else:
                raise ValueError(f"Invalid parameter '{param}' for the model.")

        # Apply k-fold cross-validation with the given scoring function
        scores = k_fold_cross_validation(model, dataset, cv=cv, scoring=scoring)

        # Save the results for the current combination
        results.append({
            'hyperparameters': hyperparameters,  # Current hyperparameter combination
            'scores': scores  # Scores from cross-validation
        })

    # Determine the best result based on the average score across folds
    best_result = max(results, key=lambda x: np.mean(x['scores']))
    # Extract the best hyperparameters and their corresponding score
    best_hyperparameters = best_result['hyperparameters']
    best_score = np.mean(best_result['scores'])

    # Return a dictionary with all results and the best configuration
    return {
        'results': results,  
        'best_hyperparameters': best_hyperparameters,  
        'best_score': best_score  
    }


if __name__ == '__main__':

    from si.io.csv_file import read_csv
    from si.models.logistic_regression import LogisticRegression

    path = r"C:\Users\tiago\OneDrive\Documentos\GitHub\si\datasets\breast_bin\breast-bin.csv"
    dataset = read_csv(path, sep=",", features=True, label=True)

    # Create the LogisticRegression model
    model = LogisticRegression()

    # Define the hyperparameter distributions
    hyperparameter_grid = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200).astype(int)  # Ensure max_iter is integer
    }

    # Perform the randomized search
    results = randomized_search_cv(
        model=model,
        dataset=dataset,
        hyperparameter_grid=hyperparameter_grid,
        scoring=None,  # Default scoring
        cv=3,
        n_iter=10
    )

    # Print the results
    print("Results of Randomized Search:")
    for res in results['results']:
        print(f"Hyperparameters: {res['hyperparameters']}, Scores: {res['scores']}")

    print("\nBest Hyperparameters:")
    print(results['best_hyperparameters'])

    print("\nBest Score:")
    print(results['best_score'])