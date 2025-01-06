import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):
    """
    An ensemble classifier that combines predictions from multiple base models 
    using a meta-model to generate final predictions.

    Parameters
    ----------
    models : list
        Array-like of base models to be combined in the ensemble.
        Each model should be an instance of a Model class.
    
    final_model : Model
        The meta-model used to generate final predictions based on base model predictions.
    """
    def __init__(self, models: list, final_model, **kwargs):
        """
        Initialize the StackingClassifier with base models and a meta-model.

        Parameters
        ----------
        models : list
            List of base models to be included in the ensemble.
        
        final_model : Model
            The meta-model used to combine the predictions of the base models.

        kwargs : dict
            Additional parameters for the base class.
        """
        super().__init__(**kwargs)  # Inherit parameters from the parent Model class
        self.models = models  # Assign the base models to the classifier
        self.final_model = final_model  # Assign the meta-model (final model)
        self.new_dataset = None  # Stores predictions from base models

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Train the ensemble models and the final model with the predictions of the initial set of models.

        Parameters
        ----------
        dataset : Dataset
            The training data to fit the models.
        
        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        # Fit each base model with the training dataset
        for model in self.models:
            model.fit(dataset)
        
        # Generate predictions from the base models
        base_predictions_list = [model.predict(dataset) for model in self.models]
        
        # Stack the predictions column-wise to create a 2D array (each column corresponds to a model's predictions)
        base_predictions = np.column_stack(base_predictions_list)

        # Create a new dataset using the base models' predictions
        # The new dataset has X as the predictions and y as the original labels
        self.new_dataset = Dataset(X=base_predictions, y=dataset.y, label=dataset.label)  # Defina new_dataset corretamente

        # Train the final model (meta-model) using the predictions from the base models
        self.final_model.fit(self.new_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data to generate predictions on.
        
        Returns
        -------
        final_predictions : np.ndarray
            The predicted labels from the meta-model.
        """
        # Generate predictions from each base model
        base_predictions_list = [model.predict(dataset) for model in self.models]
        
        # Stack the predictions column-wise to create a 2D array (predictions from all models)
        base_predictions = np.column_stack(base_predictions_list)

        # Use the final model (meta-model) to predict based on the predictions of the base models
        final_predictions = self.final_model.predict(Dataset(X=base_predictions, y=None))

        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    # Import necessary modules and classes for data loading and model training
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    # Load the dataset
    filename = "C:\\Users\\tiago\\OneDrive\\Documentos\\GitHub\\si\\datasets\\breast_bin\\breast-bin.csv"
    breast = read_csv(filename, sep=",", features=True, label=True)
    
    # Split the dataset into training and test sets
    train_dataset, test_dataset = stratified_train_test_split(breast, test_size=0.20, random_state=42)

    # Initialize the base models (KNN, Logistic Regression, and Decision Tree)
    knn1 = KNNClassifier(k=3)
    logreg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    dtree = DecisionTreeClassifier()

    # Initialize the final model (KNN) to combine the base models' predictions
    knn2 = KNNClassifier(k=3)

    # Initialize the StackingClassifier with the base models and final model
    stacking = StackingClassifier(models=[knn1, logreg, dtree], final_model=knn2)

    # Train the StackingClassifier model using the training dataset
    stacking.fit(train_dataset)

    # Compute the score on the test dataset
    score = stacking._score(test_dataset)  # Directly use _score here
    print(f"StackingClassifier Score: {score}")

    # Get predictions on the test dataset
    predictions = stacking._predict(test_dataset)
    print("Predictions:", predictions)
