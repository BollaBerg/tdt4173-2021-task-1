import numpy as np

from k_means import KMeans, ProcessingNormalize

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


def test_preprocessing_normalize():
    """Test that the ProcessingNormalize function works correctly"""
    input_values = np.array([[0, 0.1], [0.2, 5.0], [-0.2, 5.0]])
    processing = ProcessingNormalize()
    output_values = processing.preprocess(input_values, floor=0.0, roof=1.0)

    expected_output_values = np.array([[0.5, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(output_values, expected_output_values)


    un_normalized_values = processing.unprocess(output_values)
    np.testing.assert_array_equal(un_normalized_values, input_values)
