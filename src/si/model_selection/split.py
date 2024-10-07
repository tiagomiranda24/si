from si.data.dataset import Dataset
import numpy as np



def train_test_split(dataset : Dataset, test_size : float, random_state : 1234 ):

    np.random.seed(random_state)

    permutation = np.random.permutation(dataset.X.shape()(0)) 

    test_sample_size = int(dataset.shape()(0)*test_size)

    test_idx = permutation[:test_sample_size]
    train_idx = permutation[test_sample_size:]

    train_dataset = Dataset(X=dataset.X[train_idx, :], y=dataset.y)