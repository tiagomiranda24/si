import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        """
        Test the dropna functionality.
        """
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        dataset.dropna()
        self.assertEqual((2, 3), dataset.shape())  # Check shape after dropping NaNs
        self.assertTrue(np.all(~np.isnan(dataset.X)))  # Ensure no NaNs remain

    def test_fillna(self):
        """
        Test the fillna functionality.
        """
        X = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        # Fill NaNs with mean
        dataset.fillna("mean")
        self.assertFalse(np.any(np.isnan(dataset.X)))  # Ensure no NaNs remain
        self.assertEqual(6.5, dataset.X[0, 1])  # Check filled value (mean)

        # Fill NaNs with median
        X[1, 2] = np.nan
        dataset = Dataset(X, y)
        dataset.fillna("median")
        self.assertFalse(np.any(np.isnan(dataset.X)))  # Ensure no NaNs remain
        self.assertEqual(6.5, dataset.X[0, 1])  # Valor corrigido para a mediana


    def test_remove_by_index(self):
        """
        Test the remove_by_index functionality for valid and invalid indices.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        dataset = Dataset(X, y)

        # Test removing a valid index
        dataset.remove_by_index(1)
        np.testing.assert_array_equal(dataset.X, np.array([[1, 2], [5, 6]]))
        np.testing.assert_array_equal(dataset.y, np.array([10, 30]))

        # Test removing an invalid index
        with self.assertRaises(IndexError):
            dataset.remove_by_index(4)  # Index out of bounds