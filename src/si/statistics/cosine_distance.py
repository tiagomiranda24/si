import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine distance between a point x and a set of points y.
    
    Parameters
    ----------
    x: np.ndarray
        A single sample.
    y: np.ndarray
        Multiple samples.
    
    Returns
    -------
    np.ndarray
        An array containing the cosine distances between x and each sample in y.
    """
    # Calculate the dot product between x and each vector in y
    dot_product = np.dot(y, x)
    
    # Calculate the norms (magnitudes) of x and each vector in y
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)
    
    # Compute cosine similarity
    cosine_similarity = dot_product / (norm_x * norm_y)
    
    # Compute cosine distance
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance


if __name__ == '__main__':
    # Example usage
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Calculate cosine distance
    distances = cosine_distance(x, y)
    print("Cosine distances:", distances)