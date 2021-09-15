import numpy as np
from pytest import approx

from logistic_regression import LogisticRegression, sigmoid

def test_predict_multiplies_weights_correctly():
    """Test that LogisticRegression.predict() correctly multiplies weights"""
    reg = LogisticRegression(has_intercept=False)
    reg.weights = np.array([-1, 2])
    predictions = reg.predict(np.array([[1, 2], [3, 4], [11, 22]]))

    actual_predictions = sigmoid(np.array([-1 + 4, -3 + 8, -11 + 44]))

    np.testing.assert_array_equal(
        predictions, actual_predictions,
        "Weights were not multiplied correctly"
    )