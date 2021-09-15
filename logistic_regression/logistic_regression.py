import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, initial_learning_rate=0.01):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.initial_learning_rate = initial_learning_rate

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        n_samples, n_features = X.shape
        learning_rate = self.initial_learning_rate

        # Initialize random weights
        rng = np.random.default_rng()
        self.weights = rng.random((n_features,))

        get_sample = _create_function_get_sample(X)

        for i in range(n_samples):
            sample = np.array(get_sample(X, i))
            predicted_result = self.predict(sample)
            actual_result = get_sample(y, i)

            # Update weights
            self.weights += (
                learning_rate
                * (actual_result - predicted_result)
                * sample
            )

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        sigmoid_inputs = np.dot(X, self.weights)
        return sigmoid(sigmoid_inputs)
        

# Own utility functions
def _create_function_get_sample(X_input):
    if isinstance(X_input, np.ndarray):
        def get_sample(X, index):
            return X[index]
        return get_sample
    
    elif isinstance(X_input, pd.DataFrame):
        def get_sample(X, index):
            return X.iloc[index]
        return get_sample
    
    else:
        raise NotImplementedError(
            "X_input is not numpy.ndarray or pandas.DataFrame."
            + "Other lists are not implemented!"
        )

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        
if __name__ == '__main__':
    reg = LogisticRegression()
    reg.weights = np.array([1, 2])
    print(reg.predict(np.array([[1, 2], [3, 4], [11, 22]])))