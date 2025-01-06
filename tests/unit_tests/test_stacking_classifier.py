import os
from unittest import TestCase
from si.data.dataset import Dataset
from si.ensemble.stacking_classifier import StackingClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.model_selection.split import stratified_train_test_split
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy


class TestStackingClassifier(TestCase):

    def setUp(self):
        """
        Set up the environment for tests by loading the dataset
        and splitting it into training and testing sets.
        """
        self.csv_file = "C:\\Users\\tiago\\OneDrive\\Documentos\\GitHub\\si\\datasets\\breast_bin\\breast-bin.csv"

        # Ensure the dataset file exists and load it
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Dataset file not found at {self.csv_file}")
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = stratified_train_test_split(self.dataset, test_size=0.3)

        # Initialize base models and final model
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()
        knn_final = KNNClassifier()

        # Create the StackingClassifier
        self.stacking_classifier = StackingClassifier(
            models=[knn, logistic_regression, decision_tree], final_model=knn_final
        )

    def test_fit(self):
        """
        Test the `fit` method of StackingClassifier to ensure that the models
        are correctly trained and the new dataset (new_dataset) is created properly.
        """
        # Fit the StackingClassifier with the training set
        self.stacking_classifier.fit(self.train_dataset)

        # Assertions to validate the new dataset's shape
        if hasattr(self.stacking_classifier, "new_dataset"):
            # Check if the new dataset is a numpy array and validate its shape
            new_dataset_shape = self.stacking_classifier.new_dataset.X.shape  # Acesse o X do Dataset
            train_dataset_shape = self.train_dataset.X.shape

            # Ensure the number of rows matches and the number of models matches the columns
            self.assertEqual(new_dataset_shape[0], train_dataset_shape[0])  # Acesse a dimensão correta

            # Check that the number of columns in new_dataset is equal to the number of models
            self.assertEqual(new_dataset_shape[1], len(self.stacking_classifier.models))


    def test_predict(self):
        """
        Test the `predict` method to ensure that predictions have the
        correct shape with respect to the test dataset.
        """
        # Fit the StackingClassifier
        self.stacking_classifier.fit(self.train_dataset)

        # Generate predictions
        predictions = self.stacking_classifier.predict(self.test_dataset)

        # Ensure predictions and test dataset have `shape` accessible
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        """
        Test the `score` method to ensure that the accuracy calculated
        matches the expected accuracy using the accuracy metric function.
        """
        # Fit the StackingClassifier
        self.stacking_classifier.fit(self.train_dataset)

        # Calculate the accuracy score
        accuracy_ = self.stacking_classifier.score(self.test_dataset)

        # Expected accuracy based on actual labels and predictions
        expected_accuracy = accuracy(self.test_dataset.y, self.stacking_classifier.predict(self.test_dataset))

        # Compare the predicted accuracy and expected accuracy rounded with 2 decimal places
        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))