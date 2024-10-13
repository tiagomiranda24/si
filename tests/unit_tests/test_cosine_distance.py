from unittest import TestCase
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from si.statistics.cosine_distance import cosine_distance

class TestCosineDistance(TestCase):
    
    def setUp(self):
        # Define example data for testing
        self.x = np.array([1, 2, 3])
        self.y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    def test_cosine_distance(self):
        # Calculate cosine distance using our function
        our_distance = cosine_distance(self.x, self.y)
        
        # Calculate cosine distance using sklearn
        sklearn_distance = cosine_distances(self.x.reshape(1, -1), self.y).flatten()
        
        # Assert that the results are almost equal
        assert np.allclose(our_distance, sklearn_distance), \
            f"Our distances: {our_distance}, Sklearn distances: {sklearn_distance}"
