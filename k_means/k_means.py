import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self,
                number_of_clusters : int = 2,
                max_iterations : int = 10000,
                initial_centroids : np.ndarray = None):
        """Create an instance of the KMeans algorithm, with hyperparameters.

        Args:
            number_of_clusters (int, optional): The number of clusters to find
                centroids for. Defaults to 2.
            max_iterations (int, optional): Maximum number of iterations the
                algorithm should use. If the algorithm doesn't converge until
                max_iterations has been reached, then it uses the centroids
                it found. Defaults to 10000.
            initial_centroids (np.ndarray, optional): Initial centroids to use
                when initializing the algorithm. Used primarily / only for
                debugging. If None, random initial centroids will be computed
                when .fit is called. Defaults to None.
        """
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations
        self.initial_centroids = initial_centroids
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X_np = np.array(X)  # To be sure we are always working with same type
        n_samples, n_features = X.shape

        # Initialize centroids
        if self.initial_centroids is None:
            rng = np.random.default_rng()
            centroids = rng.random((self.number_of_clusters, n_features))
        else:
            centroids = np.array(self.initial_centroids)

        # Initialize assignments, in order to finish early if it doesn't change
        prev_assignments = np.empty(0)

        for _ in range(self.max_iterations):
            # Get closest centroid for each point
            assignments = get_assignments(X, centroids)

            if np.array_equal(assignments, prev_assignments):
                print("Finished early: No change in centroid assignment")
                self.centroids = centroids
                return


            # Update each centroid to be the mean of its points
            for centroid in range(len(centroids)):
                assigned_points = X[assignments == centroid]
                centroids[centroid] = np.mean(assigned_points, axis=0)

            # Update prev_assignments, to be able to finish early
            prev_assignments = assignments
        
        self.centroids = centroids

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement 
        raise NotImplementedError()
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        # TODO: Implement 
        raise NotImplementedError()
    

### Own utility functions ###
def get_assignments(X : np.ndarray, centroids : np.ndarray) -> np.ndarray:
    """Get closest centroid for each point in X.

    Args:
        X (np.ndarray): Points in the dataset which will get assigned one 
            (closest) centroid each.
        centroids (np.ndarray): Centroids to get the closest centroid from.

    Returns:
        np.ndarray: 1D-array with length = number of datapoints, where each
            value is the index of datapoint[i]'s closest centroid
    """
    distances = cross_euclidean_distance(X, centroids)
    return np.argmin(distances, axis=1)
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))


if __name__ == '__main__':
    kmeans = KMeans()
    kmeans.fit(np.array([[0, 0], [0, 1], [3, 2], [3, 3]]))