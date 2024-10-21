from unittest import TestCase

import numpy as np
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.metrics.rmse import rmse
from si.statistics.sigmoid_function import sigmoid_function

class TestMetrics(TestCase):

    def test_accuracy(self):

        y_true = np.array([0,1,1,1,1,1,0])
        y_pred = np.array([0,1,1,1,1,1,0])

        self.assertTrue(accuracy(y_true, y_pred)==1)

    def test_mse(self):

        y_true = np.array([0.1,1.1,1,1,1,1,0])
        y_pred = np.array([0,1,1.1,1,1,1,0])

        self.assertTrue(round(mse(y_true, y_pred), 3)==0.004)
        

    def test_rmse(self):
        # Test 1: Perfect prediction (RMSE should be 0)
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        self.assertEqual(rmse(y_true, y_pred), 0.0)

        # Test 2: Larger differences (RMSE should reflect larger error)
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        self.assertEqual(round(rmse(y_true, y_pred), 3), 1.0)

    def test_sigmoid_function(self):
        # Test: Sigmoid function should return values between 0 and 1
        Y = np.array([1, 2, 3])

        self.assertTrue(all(sigmoid_function(Y) > 0))
        self.assertTrue(all(sigmoid_function(Y) < 1))