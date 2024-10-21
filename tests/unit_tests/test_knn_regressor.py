from unittest import TestCase
import numpy as np
import os
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor 
from si.model_selection.split import train_test_split

class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3)

        # Fit the model to the dataset
        knn.fit(self.dataset)

        # Check if the training data is correctly stored in the model
        self.assertTrue(np.all(self.dataset.X == knn.dataset.X)) 
        self.assertTrue(np.all(self.dataset.Y == knn.dataset.Y)) 

    def test_predict(self):
        knn = KNNRegressor(k=1)

        # Split the dataset into train and test sets
        train_dataset, test_dataset = train_test_split(self.dataset)

        # Fit the model on the training dataset
        knn.fit(train_dataset)

        # Make predictions on the test dataset
        predictions = knn.predict(test_dataset)

        # Check if the predictions array has the same number of samples as test dataset
        self.assertEqual(predictions.shape[0], test_dataset.Y.shape[0])

        # Check if predictions are in the expected range (assuming a simple case for example)
        self.assertTrue(np.all(predictions >= 0))  

    def test_score(self):
        knn = KNNRegressor(k=3)

        # Split the dataset into train and test sets
        train_dataset, test_dataset = train_test_split(self.dataset)

        # Fit the model on the training dataset
        knn.fit(train_dataset)

        # Get the score (RMSE) on the test dataset
        score = knn.score(test_dataset)

        # The score should be a float value (you may define an expected range based on the dataset)
        self.assertIsInstance(score, float)

        # Optionally, you can check that the RMSE is not too large (define a threshold as needed)
        self.assertLess(score, 10) 

if __name__ == '__main__':
    import unittest
    unittest.main()
