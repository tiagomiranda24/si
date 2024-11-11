from unittest import TestCase
import os
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_ import RandomForestClassifier
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset

class TestRandomForestClassifier(TestCase):
    def setUp(self):
        # Setup the path and read the dataset
        path = r"C:\Users\tiago\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv"
        self.dataset = read_csv(path, sep=",", features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.33, random_state=42)

    def test_fit(self):
        # Create RandomForestClassifier with a specific max_depth
        random_forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=10)  
        random_forest_classifier.fit(self.train_dataset)

        # Verify that the parameters were set correctly
        self.assertEqual(random_forest_classifier.min_sample_split, 2)
        self.assertEqual(random_forest_classifier.max_depth, 10)
        self.assertEqual(len(random_forest_classifier.trees), 10)  # Ensure that 10 trees were created

    def test_predict(self):
        # Create RandomForestClassifier and fit the model
        random_forest_classifier = RandomForestClassifier(n_estimators=10)
        random_forest_classifier.fit(self.train_dataset)

        # Get predictions for the test dataset
        predictions = random_forest_classifier.predict(self.test_dataset)

        # Ensure that predictions have the same number of instances as the test set
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        # Create RandomForestClassifier and fit the model
        random_forest_classifier = RandomForestClassifier(n_estimators=10)
        random_forest_classifier.fit(self.train_dataset)

        # Compute accuracy score on the test dataset
        accuracy_score = random_forest_classifier.score(self.test_dataset)

        # Check that the accuracy is above a threshold (for example, 0.90)
        self.assertGreaterEqual(round(accuracy_score, 2), 0.90)

    def test_most_common(self):
        # Create a small set of predictions for testing most_common function
        sample_predictions = np.array([0, 1, 1, 0, 1])
        most_common_value = RandomForestClassifier.most_common(self, sample_predictions)

        # The most common value is 1
        self.assertEqual(most_common_value, 1)

    def test_no_negative_labels(self):
        # Check if a ValueError is raised for a dataset with negative labels
        invalid_dataset = Dataset(X=self.train_dataset.X, y=np.array([-1, 0, 1, 1, 0]))  # Invalid labels

        # Create RandomForestClassifier and ensure it raises ValueError
        random_forest_classifier = RandomForestClassifier(n_estimators=10)
        with self.assertRaises(ValueError):
            random_forest_classifier.fit(invalid_dataset)
