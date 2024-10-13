from typing import Callable
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Select features according to a specified percentile of highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values).
    percentile: float
        Percentile of top features to select (between 0 and 100).

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        
        """
        Select features according to a specified percentile of highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values).
        percentile: float, default=10
            Percentile of top features to select (between 0 and 100).
        """
        
        super().__init__(**kwargs)
        
        # Adicionando validação para o percentual
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in the range (0, 100].")
        
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None
        

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the top features according to the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the selected features.
        """
        # Calculate the number of features to select based on the percentile
        num_features = int(np.ceil(self.percentile / 100 * len(self.F)))
        
        # Get the indices of the top features based on the F scores
        idxs = np.argsort(self.F)[-num_features:]
        
        # Select the corresponding features
        features = np.array(dataset.features)[idxs]
        
        # Create a new dataset with the selected features
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    # Example dataset
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    # Initialize the selector with a percentile of 50%
    selector = SelectPercentile(percentile=50)
    selector = selector.fit(dataset)
    transformed_dataset = selector.transform(dataset)
    
    # Print the selected features
    print(transformed_dataset.features)