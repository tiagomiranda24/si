import os
import numpy as np
from unittest import TestCase
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy


class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        # Dataset for multiclass classification tests
        self.y_true_one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot labels
        self.y_pred_probabilities = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])  # Predictions

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_categorical_cross_entropy_loss(self):
        
        # Tests the calculation of the loss for categorical cross-entropy
        error = CategoricalCrossEntropy().loss(self.y_true_one_hot, self.y_pred_probabilities)
        self.assertGreaterEqual(error, 0)  # The loss must be non-negative


    def test_categorical_cross_entropy_derivative(self):
        
        # Tests the calculation of the derivative for categorical cross-entropy
        derivative_error = CategoricalCrossEntropy().derivative(self.y_true_one_hot, self.y_pred_probabilities)
        self.assertEqual(derivative_error.shape, self.y_true_one_hot.shape)  # The dimension must be consistent