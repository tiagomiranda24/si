from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.random_ import RandomForestClassifier

class TestRandomForestClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        random_forest_classifier = RandomForestClassifier(max_depth=10)  
        random_forest_classifier.fit(self.train_dataset)

        self.assertEqual(random_forest_classifier.min_sample_split, 2)
        self.assertEqual(random_forest_classifier.max_depth, 10)

    def test_predict(self):
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(self.train_dataset)

        predictions = random_forest_classifier.predict(self.test_dataset.X)  

        # Correct shape access
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])  

    def test_score(self):
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(self.train_dataset)

        # Dynamic accuracy threshold
        accuracy_ = random_forest_classifier.score(self.test_dataset)
        self.assertGreaterEqual(round(accuracy_, 2), 0.90)  # Example threshold