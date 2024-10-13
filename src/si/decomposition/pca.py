import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer

class PCA(Transformer):
    def __init__(self, n_components: int):
        """
        PCA class for dimensionality reduction.
        
        Parameters
        ----------
        n_components: int
            The number of principal components to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Estimate the mean, principal components, and explained variance.
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the PCA model.
        
        Returns
        -------
        self: PCA
            The fitted PCA model.
        """
        # Calculate the mean of each feature (column-wise mean)
        self.mean = np.mean(dataset.X, axis=0)
        
        # Center the data (subtract the mean)
        X_centered = dataset.X - self.mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top 'n_components' eigenvalues and eigenvectors
        self.explained_variance = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]

        return self

    def _transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transform the dataset into the reduced space defined by the principal components.
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.
        
        Returns
        -------
        X_reduced: np.ndarray
            The transformed dataset with reduced dimensions.
        """
        # Center the data using the mean calculated during fitting
        X_centered = dataset.X - self.mean

        # Project the data onto the principal components
        X_reduced = np.dot(X_centered, self.components)

        return X_reduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Fit the PCA model and transform the dataset.
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and transform.
        
        Returns
        -------
        X_reduced: np.ndarray
            The transformed dataset with reduced dimensions.
        """
        self._fit(dataset)
        return self._transform(dataset)