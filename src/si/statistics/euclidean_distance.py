import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the Euclidean distance of a point (x) to a set of points y.
        distance_x_y1 = sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
        distance_x_y2 = sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
        ...

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Euclidean distance for each point in y.
    """
    x = np.atleast_2d(x)  # Ensure x is 2D
    # Compute the Euclidean distance using broadcasting
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

if __name__ == '__main__':
    # Test euclidean_distance
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = euclidean_distance(x, y)
    
    # Using sklearn
    from sklearn.metrics.pairwise import euclidean_distances
    sklearn_distance = euclidean_distances(x.reshape(1, -1), y)
    
    # Assert the distances are close
    assert np.allclose(our_distance, sklearn_distance)
    
    print("Our Distance:", our_distance)
    print("Sklearn Distance:", sklearn_distance)