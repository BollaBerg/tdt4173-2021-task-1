import numpy as np
from numpy.lib.arraysetops import isin 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import random
from typing import Union, Iterable


class KMeans:
    
    def __init__(self,
                number_of_clusters : int = 2,
                max_iterations : int = 10000,
                initial_centroids : Union[np.ndarray, str] = "random",
                Processing = None):
        """Create an instance of the KMeans algorithm, with hyperparameters.

        Args:
            number_of_clusters (int, optional): The number of clusters to find
                centroids for. Defaults to 2.
            max_iterations (int, optional): Maximum number of iterations the
                algorithm should use. If the algorithm doesn't converge until
                max_iterations has been reached, then it uses the centroids
                it found. Defaults to 10000.
            initial_centroids (np.ndarray | "kmeans++" | "random", optional):
                Method used for selecting initial centroids.
                If initial_centroids is np.ndarray -> initial_centroids will be
                    used as initial centroids in the algorithm (mostly used for
                    debugging).
                If initial_centroids == "kmeans++" -> Initial centroids will be
                    computed using the k-means++ algorithm.
                If initial_centroids == "random" or anything other than the 
                    above options -> Initial centroids will be randomly 
                    selected from the dataset.
            Processing (Class, optional): Processing class that MUST have
                two functions - processing.preprocess (which preprocesses the
                data) and processing.unprocess (which undoes the processing
                done earlier, to get the correct centroids out after!)
                preprocessing will be applied to X (input to .fit and .predict)
                before running the algorithm. If None, no processing will be
                applied. Defaults to None.
        """
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations

        if (isinstance(initial_centroids, Iterable)
            and not isinstance(initial_centroids, str)
        ):
            def get_supplied_centroids(
                    X : np.ndarray,
                    number_of_centroids : int) -> np.ndarray:
                return np.array(initial_centroids)
            self.get_initial_centroids = get_supplied_centroids

        elif initial_centroids in ("kmeans++", "k-means++", "k-means ++"):
            self.get_initial_centroids = k_means_plusplus

        else:
            self.get_initial_centroids = random_centroids
        

        if Processing is None:
            self.Processing = None
        else:
            self.Processing = Processing()
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # To be sure we are always working with same type
        X_np = np.array(X).astype(np.float)

        # Apply preprocessing
        if self.Processing is not None:
            X_np = self.Processing.preprocess(X_np)

        # Initialize centroids
        centroids = self.get_initial_centroids(X_np, self.number_of_clusters)

        # Initialize assignments, in order to finish early if it doesn't change
        prev_assignments = np.empty(0)

        for epoch in range(self.max_iterations):
            # Get closest centroid for each point
            assignments = get_assignments(X_np, centroids)

            if np.array_equal(assignments, prev_assignments):
                print(
                    f"Finished early (epoch {epoch}): "
                    + "No change in centroid assignment"
                )
                self.centroids = centroids
                return


            # Update each centroid to be the mean of its points
            for centroid in range(len(centroids)):
                assigned_points = X_np[assignments == centroid]
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
        X_np = np.array(X)
        if self.Processing is not None:
            X_np = self.Processing.preprocess(X_np)
        return get_assignments(X_np, self.centroids)
    
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
        if self.Processing is not None:
            return self.Processing.unprocess(self.centroids)

        return self.centroids
    

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
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    distances = cross_euclidean_distance(X, centroids)
    return np.argmin(distances, axis=1)


#### Initialization functions ####
def k_means_plusplus(X : np.ndarray, number_of_centroids : int) -> np.ndarray:
    """Compute centroids using the K-means++ algorithm.

    For more info about K-means++, see 
        https://en.wikipedia.org/wiki/K-means%2B%2B

    Args:
        X (np.ndarray): Array of datapoints to get clusters for
        number_of_centroids (int): Number of centroids to compute

    Returns:
        np.ndarray: 2D array with number_of_centroids centroids.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n_samples, n_features = X.shape
    
    # Choose one random center from the datapoints
    rng = np.random.default_rng()
    centroids = np.empty((number_of_centroids, n_features))
    centroids[0] = rng.choice(X)

    # Select rest of centroids randomly, with weights = distance from closest
    # already selected centroid
    for i in range(1, number_of_centroids):
        distances = cross_euclidean_distance(X, centroids[0:i])

        closest_distances = np.min(distances, axis=1)
        
        # rng.choice demands probabilities that sum to 1
        probabilities = closest_distances/closest_distances.sum()

        new_centroid = rng.choice(X, p=probabilities, axis=0)
        centroids[i] = new_centroid
    
    return centroids


def random_centroids(X : np.ndarray, number_of_centroids : int) -> np.ndarray:
    """Select random datapoints to use as centroids.

    A more random, but faster, way of selecting centroids than k-means++

    Args:
        X (np.ndarray): Array of datapoints to get clusters for
        number_of_centroids (int): Number of centroids to compute

    Returns:
        np.ndarray: 2D array with number_of_centroids centroids.
    """
    rng = np.random.default_rng()
    centroids = rng.choice(
        X, size=number_of_centroids, replace=False
    )
    return centroids



#### Pre- and postprocessing ####
class ProcessingNormalize:
    def __init__(self):
        self.max_ = None
        self.min_ = None
        self.diff = None

    def preprocess(
            self,
            X : np.ndarray,
            floor : float = 0.0,
            roof : float = 1.0) -> np.ndarray:
        """Normalize data in X, to be between floor and roof on both axes

        Args:
            X (np.ndarray): Array of data. If not of type np.ndarray, it will be
                converted to an np.ndarray.
            floor (float, optional): Min value for the dataset. Defaults to 0.0.
            roof (float, optional): Max value for the dataset. Defaults to 1.0.

        Returns:
            np.ndarray: Array with normalized data from X
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.max_ is None:
            max_ = np.max(X, axis=0)
            self.max_ = max_
        else:
            max_ = self.max_
        
        if self.min_ is None:
            min_ = np.min(X, axis=0) - floor
            self.min_ = min_
        else:
            min_ = self.min_

        if self.diff is None:
            diff = (max_ - min_) / roof
            self.diff = diff
        else:
            diff = self.diff

        for column in range(X.shape[1]):
            # Make the column be in [floor, ...)
            X[:, column] -= min_[column]

            # Make the column be in [floor, roof]
            X[:, column] /= diff[column]
        
        return X
    
    def unprocess(self, centroids : np.ndarray) -> np.ndarray:
        """Undo preprocessing from centroids, to get the original centroids

        Args:
            centroids (np.ndarray): Array of computed (processed) centroids

        Returns:
            np.ndarray: Array of unprocessed centroids
        """
        for column in range(centroids.shape[1]):
            # Make column be in [floor, ...)
            centroids[:, column] *= self.diff[column]

            # Make column be in [original floor, original roof]
            centroids[:, column] += self.min_[column]
        
        return centroids

    
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