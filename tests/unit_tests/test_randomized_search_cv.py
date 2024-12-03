import numpy as np
from unittest import TestCase
from si.data.dataset import Dataset
from si.base.model import Model 
from si.models.logistic_regression import LogisticRegression
from si.model_selection.cross_validate import k_fold_cross_validation
from si.model_selection.randomized_search import randomized_search_cv
from si.io.csv_file import read_csv


# Load the dataset
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
