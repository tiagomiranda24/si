import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):
    """
    An ensemble classifier that combines multiple models and uses majority voting 
    to predict class labels.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        A list of different models to be included in the ensemble.

    Attributes
    ----------
    models : array-like
        The individual models used in the ensemble.
    """
    def __init__(self, models, **kwargs):
        """
        Initialize the StackingClassifier with the given models.

        Parameters
        ----------
        models : array-like, shape = [n_models]
            A list of models that will be trained in the ensemble.

        kwargs : additional keyword arguments
            Any additional parameters to be passed to the parent Model class.
        """
        super().__init__(**kwargs)  
        self.models = models  


    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit each model in the ensemble to the training data.
        """
        # Create an empty list to hold the predictions
        all_predictions = []

        for model in self.models:
            model.fit(dataset)  
            predictions = model.predict(dataset)
            all_predictions.append(predictions)
        
        # Store the predictions in new_dataset
        self.new_dataset = np.column_stack(all_predictions)

        return self
            

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for the samples in the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The test data for which predictions are to be made.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels for the input data.
        """
        
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            Calculate the majority vote from an array of predictions.

            Parameters
            ----------
            pred: np.ndarray
                An array of predictions to determine the majority vote.

            Returns
            -------
            majority_vote: int
                The class label that received the most votes.
            """
            # Get unique labels and their counts
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)] 

        # Collect predictions from each model and transpose the results
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # Apply the majority vote function along the rows of the predictions array
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the mean accuracy of the predictions compared to the true labels.

        Parameters
        ----------
        dataset : Dataset
            The test dataset containing the true labels.
        predictions: np.ndarray
            The predicted labels from the ensemble.

        Returns
        -------
        score : float
            The accuracy score of the predictions.
        """
        return accuracy(dataset.y, predictions)  


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    filename = "C:\\Users\\tiago\\OneDrive\\Documentos\\GitHub\\si\\datasets\\breast_bin\\breast-bin.csv"
    breast = read_csv(filename, sep=",", features=True, label=True)
    train_dataset, test_dataset = stratified_train_test_split(breast, test_size=0.20, random_state=42)

    # Initialize the base models
    knn1 = KNNClassifier(k=3)
    logreg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    dtree = DecisionTreeClassifier()

    # Initialize the final model
    knn2 = KNNClassifier(k=3)

    # Initialize the StackingClassifier with the base models and final model
    stacking = StackingClassifier(models=[knn1, logreg, dtree], final_model=knn2)

    # Train the StackingClassifier model
    stacking.fit(train_dataset)

    # Compute the score on the test dataset
    score = stacking.score(test_dataset)
    print(f"StackingClassifier Score: {score}")

    # Get predictions on the test dataset
    predictions = stacking.predict(test_dataset)
    print("Predictions:", predictions)