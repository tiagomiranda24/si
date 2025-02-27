from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)
    

    def dropna(self) -> 'ClassType':
        """
        Removes rows containing NaN values from X and y.

        Returns
        -------
        self : ClassType
            Returns the object itself after removing NaN rows.
        """
        # 1. Identify the rows that do not contain NaN values
        non_nan_mask = ~np.any(np.isnan(self.X), axis=1)

        # 2. Filter the rows in X that do not have NaN values
        self.X = self.X[non_nan_mask]

        # 3. Update the y vector by removing the entries corresponding to the removed rows
        self.y = self.y[non_nan_mask]

        return self

    def fillna(self, value) -> 'ClassType':
        """
        Replaces NaN values in X with the specified value.

        Parameters
        ----------
        value : str or float
            If "mean", replaces NaN with the mean of the column.
            If "median", replaces NaN with the median of the column.
            If a float, replaces NaN with the specified value.

        Returns
        -------
        self : ClassType
            Returns the object itself after replacing NaN values.
        """
        # For each column in X, replace NaN values with the appropriate value
        for i in range(self.X.shape[1]):
            if np.any(np.isnan(self.X[:, i])):
                # If the value is "mean", calculate the mean ignoring NaN
                if value == "mean":
                    fill_value = np.nanmean(self.X[:, i])
                # If the value is "median", calculate the median ignoring NaN
                elif value == "median":
                    fill_value = np.nanmedian(self.X[:, i])
                else:
                    fill_value = value

                # Replace NaN in column i with the calculated value
                self.X[:, i] = np.where(np.isnan(self.X[:, i]), fill_value, self.X[:, i])

        return self

    def remove_by_index(self, index: int) -> 'ClassType':
        """
        Removes a row from X and y by index.

        Parameters
        ----------
        index : int
            The index of the row to be removed.

        Returns
        -------
        self : ClassType
            Returns the object itself after removing the row.
        
        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        # Check if the index is valid
        if index < 0 or index >= self.X.shape[0]:
            raise IndexError("Index out of bounds")

        # Create a mask that keeps all rows except the one being removed
        mask = np.ones(self.X.shape[0], dtype=bool)
        mask[index] = False

        # Apply the mask to filter X and y
        self.X = self.X[mask]
        self.y = self.y[mask]

        return self



if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
