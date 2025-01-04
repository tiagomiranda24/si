from unittest import TestCase
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier 
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset


class TestRandomForestClassifier(TestCase):
    def setUp(self):
        """
        Set up the test environment by loading the dataset and splitting it into train and test sets.
        """
        # Define the path to the dataset
        path = r"C:\Users\tiago\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv"
        self.dataset = read_csv(path, sep=",", features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.33, random_state=42)

    def test_fit(self):
        """
        Test the fit method of RandomForestClassifier.
        """
        # Create a RandomForestClassifier with a specific max_depth
        random_forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=10)  
        random_forest_classifier.fit(self.train_dataset)

        # Verify that the parameters are correctly set
        self.assertEqual(random_forest_classifier.min_sample_split, 2)
        self.assertEqual(random_forest_classifier.max_depth, 10)
        self.assertEqual(len(random_forest_classifier.trees), 10)  

    def test_predict(self):
        """
        Test the predict method of RandomForestClassifier.
        """
        # Create and train RandomForestClassifier
        random_forest_classifier = RandomForestClassifier(n_estimators=10)
        random_forest_classifier.fit(self.train_dataset)

        # Get predictions for the test dataset
        predictions = random_forest_classifier.predict(self.test_dataset)

        # Verify that predictions have the same number of instances as the test dataset
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        """
        Test the score method of RandomForestClassifier.
        """
        # Create and train RandomForestClassifier
        random_forest_classifier = RandomForestClassifier(n_estimators=10)
        random_forest_classifier.fit(self.train_dataset)

        # Calculate the accuracy on the test dataset
        accuracy_score = random_forest_classifier.score(self.test_dataset)

        # Verify that accuracy is greater than or equal to 0.90 (adjustable)
        self.assertGreaterEqual(round(accuracy_score, 2), 0.90)

    def test_most_common(self):
        """
        Test the most_common method of RandomForestClassifier.
        """
        # Create a set of predictions to test most_common
        sample_predictions = np.array([0, 1, 1, 0, 1])

        # Create a classifier to access the method
        random_forest_classifier = RandomForestClassifier(n_estimators=10)
        most_common_value = random_forest_classifier._most_common(sample_predictions)

        # The most common value should be 1
        self.assertEqual(most_common_value, 1)