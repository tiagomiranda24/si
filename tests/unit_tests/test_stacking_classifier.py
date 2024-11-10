import unittest
import numpy as np
from si.data.dataset import Dataset
from si.ensemble.stacking_classifier import StackingClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split

class TestStackingClassifier(unittest.TestCase):
    def setUp(self):
        # Create a random dataset
        self.dataset = Dataset.from_random(600, 100, 2)
        self.dataset_train, self.dataset_test = train_test_split(self.dataset, test_size=0.2)

        # Initialize individual models
        self.knn = KNNClassifier(k=3)
        self.lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

        # Initialize the StackingClassifier with the individual models
        self.stacking = StackingClassifier([self.knn, self.lg])

    def test_fit(self):
        # Test fitting the StackingClassifier
        self.stacking.fit(self.dataset_train)
        for model in self.stacking.models:
            self.assertTrue(hasattr(model, 'is_fitted'))
            self.assertTrue(model.is_fitted)

    def test_predict(self):
        # Test predicting with the StackingClassifier
        self.stacking.fit(self.dataset_train)
        predictions = self.stacking.predict(self.dataset_test)
        self.assertEqual(len(predictions), len(self.dataset_test.y))
        self.assertTrue(np.all(np.isin(predictions, np.unique(self.dataset.y))))

    def test_score(self):
        # Test scoring the StackingClassifier
        self.stacking.fit(self.dataset_train)
        score = self.stacking.score(self.dataset_test)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()