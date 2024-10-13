from unittest import TestCase
from datasets import DATASETS_PATH
import numpy as np
import os
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from si.data.dataset import Dataset

class TestPCA(TestCase):
    
    def setUp(self):
        # Load a sample dataset, for example the iris dataset
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
    
    def test_fit(self):
        # Create a PCA instance with a specified number of components
        pca = PCA(n_components=2)
        pca._fit(self.dataset)  # Pass the Dataset object instead of dataset.X
        
        # Verify that the mean is calculated correctly
        expected_mean = np.mean(self.dataset.X, axis=0)
        self.assertTrue(np.allclose(pca.mean, expected_mean))
        
        # Check if the number of components is as expected
        self.assertEqual(pca.components.shape[1], 2)  # Should have 2 components

    def test_transform(self):
        # Create a PCA instance and fit it
        pca = PCA(n_components=2)
        pca._fit(self.dataset)
        
        # Transform the dataset
        X_reduced = pca._transform(self.dataset)
        
        # Check if the reduced dataset has the correct shape
        # The number of rows should remain the same, but the columns should match n_components
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])
        self.assertEqual(X_reduced.shape[1], 2)  # Because n_components = 2
    
    def test_fit_transform(self):
        # Create a PCA instance and use fit_transform
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(self.dataset)
        
        # Check if the reduced dataset has the correct shape
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])
        self.assertEqual(X_reduced.shape[1], 2)
        
        # Check if the mean of the transformed data is approximately zero (data is centered)
        transformed_mean = np.mean(X_reduced, axis=0)
        self.assertTrue(np.allclose(transformed_mean, 0, atol=1e-6))