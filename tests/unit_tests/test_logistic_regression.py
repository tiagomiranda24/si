from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.statistics.sigmoid_function import sigmoid_function
from si.models.logistic_regression import LogisticRegression

class TestLogisticRegression(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        logistic = LogisticRegression()
        logistic.fit(self.train_dataset)

        self.assertEqual(logistic.theta.shape[0], self.train_dataset.shape()[1])
        self.assertNotEqual(logistic.theta_zero, None)
        self.assertNotEqual(len(logistic.cost_history), 0)
        self.assertNotEqual(len(logistic.mean), 0)
        self.assertNotEqual(len(logistic.std), 0)

    def test_predict(self):
        logistic = LogisticRegression()
        logistic.fit(self.train_dataset)

        predictions = logistic.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        logistic = LogisticRegression()
        logistic.fit(self.train_dataset)
        mse_ = logistic.score(self.test_dataset)

        self.assertEqual(round(mse_, 2), 9971.19)