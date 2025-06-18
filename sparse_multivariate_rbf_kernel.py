import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def sparse_multivarite_rbf_kernel(X: np.ndarray, Y: np.ndarray, h: float, k: int) -> csr_matrix:
    """
    Computes a sparse multivariate RBF kernel matrix by finding the k-nearest neighbors.

    For each row in X, the RBF kernel is computed only for its k-nearest neighbors
    in Y based on Euclidean distance. This is highly efficient for large datasets.

    The RBF kernel is defined as: K(x, y) = exp(-||x - y||^2 / (2 * h^2)).

    Args:
        X (np.ndarray): A 2D array of shape (n_samples_X, n_features).
            The first set of data points.
        Y (np.ndarray): A 2D array of shape (n_samples_Y, n_features).
            The second set of data points.
        h (float): The bandwidth parameter (a.k.a. length scale) of the RBF kernel.
            Must be a positive number.
        k (int): The number of nearest neighbors in Y to consider for each point in X.

    Returns:
        csr_matrix: A sparse matrix of shape (n_samples_X, n_samples_Y) containing
                    the computed RBF kernel values.
    
    Raises:
        ValueError: If h is not positive, or if k is not a positive integer.
    """
    # --- 1. Input Validation ---
    if h <= 0:
        raise ValueError("Bandwidth parameter h must be positive.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("Number of neighbors k must be a positive integer.")
    
    n_samples_X, n_features_X = X.shape
    n_samples_Y, n_features_Y = Y.shape

    if n_features_X != n_features_Y:
        raise ValueError("X and Y must have the same number of features.")

    # Ensure k is not larger than the number of available points in Y
    k = min(k, n_samples_Y)

    # --- 2. Find K-Nearest Neighbors ---
    # Use scikit-learn's NearestNeighbors for efficiency.
    # 'auto' algorithm will choose the best method (e.g., BallTree, KDTree)
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
    nn.fit(Y)
    
    # distances: Euclidean distances to the neighbors
    # indices: Column indices (in Y) of the neighbors
    distances, indices = nn.kneighbors(X)

    # --- 3. Calculate RBF Kernel Values ---
    # The RBF formula requires the squared Euclidean distance.
    squared_distances = distances**2
    
    # Apply the RBF kernel formula to the non-zero distances.
    # This is our 'data' for the sparse matrix.
    kernel_values = np.exp(-squared_distances / (2 * h**2))

    # --- 4. Construct the Sparse Matrix ---
    # Create the row pointers for the sparse matrix. For each of the n_samples_X,
    # we have k corresponding non-zero values.
    # Example: if n_samples_X=3, k=2, this creates [0, 0, 1, 1, 2, 2]
    row_indices = np.arange(n_samples_X).repeat(k)
    
    # The 'indices' array from kneighbors already gives us the column indices.
    # We just need to flatten it to a 1D array.
    col_indices = indices.flatten()

    # The 'kernel_values' also needs to be flattened.
    data = kernel_values.flatten()
    
    # Create the CSR matrix
    kernel_matrix = csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(n_samples_X, n_samples_Y)
    )
    
    return kernel_matrix