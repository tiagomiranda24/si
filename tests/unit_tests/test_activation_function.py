import os
import numpy as np 
from unittest import TestCase
from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.activation import ReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation

class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestTanhLayer(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        # Initialize the Tanh activation function
        tanh_layer = TanhActivation()
        
        # Apply the activation function to the dataset
        result = tanh_layer.activation_function(self.dataset.X)
        
        # Check that all values are in the range [-1, 1]
        self.assertTrue(all([-1 <= i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))

    def test_derivative(self):
        # Initialize the Tanh activation function
        tanh_layer = TanhActivation()
        
        # Compute the derivative of the activation function
        derivative = tanh_layer.derivative(self.dataset.X)
        
        # Ensure the derivative has the same shape as the input data
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestSoftmaxLayer(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        # Initialize the Softmax activation function
        softmax_layer = SoftmaxActivation()
        
        # Apply the activation function to the dataset
        result = softmax_layer.activation_function(self.dataset.X)
        
        # Compute the sum of each row (since Softmax outputs probabilities summing to 1)
        row_sums = np.sum(result, axis=1)
        
        # Check if the sum of each row is approximately 1 (with a tolerance of 1e-6)
        self.assertTrue(np.allclose(row_sums, 1, atol=1e-6))

    def test_derivative(self):
        # Initialize the Softmax activation function
        softmax_layer = SoftmaxActivation()
        
        # Compute the derivative of the activation function
        derivative = softmax_layer.derivative(self.dataset.X)
        
        # Ensure the derivative has the same shape as the input data
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])
