import numpy as np
from unittest import TestCase
from si.neural_networks.layers import DropoutLayer as Dropout

class TestDropoutLayer(TestCase):
    def setUp(self):
        """
        Set up the test case with consistent random seed and test data.
        """
        np.random.seed(0)  # For reproducible results
        self.dropout_layer = Dropout(probability=0.3)
        self.dropout_layer.set_input_shape((4,))  # Example input with 4 features
        self.input_data = np.random.rand(5, 4)  # Batch of 5 examples, each with 4 features

    def test_forward_training(self):
        """
        Test forward propagation in training mode.
        """
        output_train = self.dropout_layer.forward_propagation(input=self.input_data, training=True)
        # Assert the output shape is the same as input shape
        self.assertEqual(output_train.shape, self.input_data.shape)

        # Assert some values are zero (due to dropout)
        self.assertTrue(np.any(output_train == 0))

        # Assert scaling (check the mean roughly matches expected scale)
        scaling_factor = 1 / (1 - self.dropout_layer.probability)
        scaled_mean = np.mean(self.input_data) * scaling_factor
        output_mean = np.mean(output_train[output_train != 0])  # Non-zero mean
        self.assertAlmostEqual(output_mean, scaled_mean, delta=0.1)

    def test_forward_inference(self):
        """
        Test forward propagation in inference mode.
        """
        output_infer = self.dropout_layer.forward_propagation(input=self.input_data, training=False)
        # Assert the output is the same as the input
        np.testing.assert_array_equal(output_infer, self.input_data)

    def test_backward(self):
        """
        Test backward propagation.
        """
        self.dropout_layer.forward_propagation(input=self.input_data, training=True)  # Generate mask
        output_error = np.ones_like(self.input_data)  # Arbitrary error for testing
        input_error = self.dropout_layer.backward_propagation(output_error=output_error)

        # Assert the input error has the same shape as the input
        self.assertEqual(input_error.shape, self.input_data.shape)

        # Assert input error is zero where mask is zero
        mask = self.dropout_layer.mask
        np.testing.assert_array_equal(input_error[mask == 0], 0)