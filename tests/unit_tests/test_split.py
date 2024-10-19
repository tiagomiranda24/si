from unittest import TestCase
from datasets import DATASETS_PATH
import os
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):
        # Perform the stratified split with test_size of 0.2 and random_state 123
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
    
         # Calculate the expected size of the test set
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
    
        # Check if the size of the test set is as expected
        self.assertEqual(test.shape()[0], test_samples_size)
        
        # Check if the size of the training set is the remainder of the data
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
        
        # Verify that the class proportions are maintained
        unique_classes, train_counts = np.unique(train.y, return_counts=True)
        _, test_counts = np.unique(test.y, return_counts=True)
        _, total_counts = np.unique(self.dataset.y, return_counts=True)
        
        # Compare the proportions between training, testing, and total sets
        train_proportions = train_counts / train.shape()[0]
        test_proportions = test_counts / test.shape()[0]
        total_proportions = total_counts / self.dataset.shape()[0]
        
        # Check if the proportions of each class are similar across the sets
        for i in range(len(unique_classes)):
            self.assertAlmostEqual(train_proportions[i], total_proportions[i], places=2)
            self.assertAlmostEqual(test_proportions[i], total_proportions[i], places=2)
