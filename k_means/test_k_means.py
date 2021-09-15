import numpy as np

from k_means import KMeans

def test_KMeans_computes_centroids_correctly():
    """Test that KMeans computes centroids correctly
    
    With max_iterations = 1 and known initial centroids, we can know beforehand
    what the centroids should be after the one iteration.
    """
    kmeans = KMeans(max_iterations=1, initial_centroids=[[1., 1.], [4., 4.]])

    input_data = np.array([[0, 0], [0, 1], [5, 5], [4, 5]])
    kmeans.fit(input_data)

    precomputed_centroids = np.array([[0, 0.5], [4.5, 5]])

    np.testing.assert_array_equal(kmeans.centroids, precomputed_centroids)