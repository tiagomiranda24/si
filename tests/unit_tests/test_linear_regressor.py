from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv

from si.metrics.mse import mse


class RidgeRegression(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        ridge = RidgeRegression()
        ridge.fit(self.train_dataset)

        self.assertEqual(ridge.theta.shape[0], self.train_dataset.get_shape()(1))
        self.assertNotEqual(ridge.theta_zero, None)
        
        estimates the theta and theta_zero coefficients, mean, std and cost_history



        knn = KNNClassifier(k=3)

        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):

        -
_predict – predicts the dependent variable (y) using the estimated theta coefficients

        knn = KNNClassifier(k=1)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.all(predictions == test_dataset.y))

    def test_score(self):

        -
_score – calculates the error between the real and predicted y values

        knn = KNNClassifier(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertEqual(score, 1)


    def _cost(self):
       
cost – calcultes the cost function betweentherealandpredictedyvalues



parameters:
-
l2_penalty –L2 regularization parameter
-
alpha – thelearning rate
-
max_iter – maximum number of iterations
-
patience – maximum number of iterations without improvement allowed
-
scale – wheter to scale the data or not
•
estimated parameters:
-
theta – the coefficients of the model for every feature
-
theta_zero – the zero coefficient (y intercept)
-
mean – mean of the dataset (for every feature)
-
std – standard deviation of the dataset (for every feature)
-
cost_history – cost function value at each iteration (dictionary iteration: cost)
•
