import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

from typing import Callable


class LogisticRegression:
    
    def __init__(self,
                initial_learning_rate : float = 0.01,
                epochs : int = 10000,
                has_intercept : bool = True,
                preprocessing : Callable[[pd.DataFrame], pd.DataFrame] = None):
        """Initialize an instance of Logistic Regression, with optional
        hyperparameters.

        Args:
            initial_learning_rate (float, optional): Learning rate (alpha) the
                regression should use. Defaults to 0.01.
            epochs (int, optional): Number of epochs the regression should
                use to learn. Defaults to 10000.
            has_intercept (bool, optional): Whether the regression should have
                a different intercept than (0,0). If False, there is no 0-th 
                weight, meaning that the 50/50 split will go through origo.
                Defaults to True.
            preprocessing (Callable[(pd.DataFrame) -> pd.DataFrame], optional):
                Preprocessing function to apply to the data before running
                logistic regression on it. Will be applied to input data for
                both .fit and .predict. If None, no preprocessing will be done
                to the data. Defaults to None.
        """
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.initial_learning_rate = initial_learning_rate
        self.epochs = epochs
        self.has_intercept = has_intercept
        self.preprocessing = preprocessing

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        _, n_features = X.shape
        learning_rate = self.initial_learning_rate

        if self.preprocessing is not None:
            X = self.preprocessing(X)

        if self.has_intercept:
            n_features += 1
            X = add_intercept_column(X)

        # Initialize random weights
        rng = np.random.default_rng()
        self.weights = rng.random((n_features,))

        for _ in range(self.epochs):
            predicted_results = sigmoid(np.dot(X, self.weights))
            actual_results = y

            # Calculate how much the prediction missed
            prediction_difference = actual_results - predicted_results

            # Calculate how much to change the weights
            delta_weights = learning_rate * np.dot(
                prediction_difference, X
            )

            # Update the weights
            self.weights += delta_weights

    
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
        if self.preprocessing is not None:
            X = self.preprocessing(X)

        if self.has_intercept:
            X = add_intercept_column(X)

        sigmoid_inputs = np.dot(X, self.weights)
        return sigmoid(sigmoid_inputs)
        

# Own utility functions
def add_intercept_column(X : pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with values from X and an Intercept column.

    This is used when X is supplied as input to either .fit or .predict, if the
    hyperparameter has_intercept is True. If the parameter is True, the 
    prediction from X can have its intercept in a different spot than origo.

    If X already is a Pandas DataFrame with a column named "Intercept", then
    the method simply returns X early.

    If X already is a Pandas DataFrame without a column named "Intercept", then
    the column is added and the DataFrame is returned.

    If X is not a Pandas DataFrame, the method converts it to a DataFrame,
    before adding a column named "Intercept" to it.

    Args:
        X (pd.DataFrame): Input DataFrame that should get an Intercept column.

    Returns:
        pd.DataFrame: DataFrame with original data, and an Intercept column.
    """
    X_df = pd.DataFrame(X)

    if "Intercept" in X_df.columns:
        return X_df
    
    n_samples = X_df.shape[0]
    X_df.insert(0, "Intercept", np.ones((n_samples,)))
    return X_df


def cartesian_to_polar_coordinates(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    """Convert cartesian coordinates to polar coordinates

    Args:
        x (np.ndarray): Array of distances along x-axis
        y (np.ndarray): Array of distances along y-axis

    Returns:
        np.ndarray: Distances from origo
        np.ndarray: Angles from x-axis
    """
    distance_from_origo = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return distance_from_origo, angle


def preprocessing_cartesian_to_polar_coordinates(X : pd.DataFrame) -> pd.DataFrame:
    """Preprocessing step to convert a DataFrame of cartesian (x,y)-coordinates
    to a DataFrame of polar coordinates

    NOTE: Assumes that X has columns 'x0' and 'x1', which will be used as 
    cartesian coordinates (x, y).

    Args:
        X (pd.DataFrame): Input DataFrame with cartesian coordinates: (x0, x1).
            IMPORTANT: MUST CONTAIN COLUMNS 'x0' and 'x1'!

    Returns:
        pd.DataFrame: DataFrame with polar coordinates: (distance, angle)
    """
    distances, angles = cartesian_to_polar_coordinates(X['x0'], X['x1'])
    return pd.DataFrame({"Distances": distances, "Angles": angles})


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
    # Partition data into independent (feature) and depended (target) variables
    data_1 = pd.read_csv('/mnt/c/Users/andre/Documents/Skole/Skole 21.2 Høst/TDT4173 Maskinlæring/tdt4173-2021-task-1/logistic_regression/data_1.csv')
    X = data_1[['x0', 'x1']]
    y = data_1['y']

    # Create and train model.
    model_1 = LogisticRegression() # <-- Should work with default constructor  
    model_1.fit(X, y)

    # Calculate accuracy and cross entropy for (insample) predictions 
    y_pred = model_1.predict(X)
    print(f'Accuracy: {binary_accuracy(y_true=y, y_pred=y_pred, threshold=0.5) :.3f}')
    print(f'Cross Entropy: {binary_cross_entropy(y_true=y, y_pred=y_pred) :.3f}')