import os
import numpy as np
from unittest import TestCase
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression 
from datasets import DATASETS_PATH

class TestLassoRegression(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        # Splitting the dataset into training and testing sets
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        # Creating an instance of LassoRegression
        lasso = LassoRegression(l1_penalty=1.0, scale=True)
        
        # Fitting the model to the training data
        lasso._fit(self.train_dataset)

        # Checking if the number of coefficients is equal to the number of features
        self.assertEqual(lasso.theta.shape[0], self.train_dataset.shape()[1])

        # Checking if the intercept (theta_zero) is no longer None
        self.assertNotEqual(lasso.theta_zero, None)

        # Checking if the means were calculated when scale=True
        self.assertNotEqual(len(lasso.mean), 0)
        self.assertNotEqual(len(lasso.std), 0)

    def test_predict(self):
        # Creating an instance of LassoRegression and fitting it
        lasso = LassoRegression(l1_penalty=1.0, scale=True)
        lasso._fit(self.train_dataset)

        # Making predictions on the test data
        predictions = lasso._predict(self.test_dataset)

        # Checking if the number of predictions is equal to the number of test samples
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

    def test_score(self):
        # Creating an instance of LassoRegression and fitting it
        lasso = LassoRegression(l1_penalty=1.0, scale=True)
        lasso._fit(self.train_dataset)

        # Calculating the mean squared error (MSE) on the test data
        mse_ = lasso._score(self.test_dataset)

        # Checking if the MSE is within an expected range (Adjust the value according to the expectations for your dataset)
        self.assertTrue(0 <= mse_ < 10000)  