import numpy as np
import matplotlib.pyplot as plt

def mse(y, predictions):
    """
    Compute the Mean Squared Error (MSE) between true values and predictions.

    Parameters:
    y (array-like): Array of true target values.
    predictions (array-like): Array of predicted values.

    Returns:
    float: The Mean Squared Error.
    """
    return np.mean(np.square(y - predictions))


def accuracy(y, predictions):
    """
    Compute a ratio-based accuracy metric. This is not a standard metric and
    may not be appropriate for all problems. Typically used in classification
    problems with binary outcomes.

    Parameters:
    y (array-like): Array of true target values.
    predictions (array-like): Array of predicted values.

    Returns:
    float: The ratio-based accuracy metric.
    """
    return np.mean(predictions / y)


def train_test_split(X, y, split=0.2, shuffle=False):
    """
    Split data into training and testing sets.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    split (float): Fraction of data to be used as the test set (default is 0.2).
    shuffle (bool): Whether to shuffle the data before splitting (default is False).

    Returns:
    tuple: (X_train, X_test, y_train, y_test) - Split data.
    """
    if shuffle:
        np.random.shuffle(X)
        np.random.shuffle(y)

    split_index = int(len(X) * (1 - split))
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


class LinearRegression:
    def __init__(self, iterations=1000, learning_rate=0.01):
        """
        Initialize the LinearRegression model.

        Parameters:
        iterations (int): Number of iterations for training (default is 1000).
        learning_rate (float): Learning rate for gradient descent (default is 0.01).
        """
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.weights = None
        self.bias = 0

        self.history = []

    def fit(self, X, y):
        """
        Train the LinearRegression model using gradient descent.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.

        Raises:
        ValueError: If X or y is empty or if the number of samples in X and y do not match.
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X or y is empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")

        self.X = X
        self.y = y

        self.weights = np.zeros(X.shape[1])

        for i in range(self.iterations):
            if i % 100 == 0:
                print("Iteration: ", i)
            self.update_weights()
            y_pred = self.predict(self.X)
            self.history.append(mse(self.y, y_pred))

    def update_weights(self):
        """
        Update weights and bias using gradient descent.
        """
        y_pred = self.predict(self.X)
        errors = y_pred - self.y

        # Calculate gradients
        dW = (1 / self.X.shape[0]) * np.dot(self.X.T, errors)
        dB = (1 / self.X.shape[0]) * np.sum(errors)

        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * dB

    def predict(self, X):
        """
        Predict target values for given feature matrix X.

        Parameters:
        X (array-like): Feature matrix.

        Returns:
        array: Predicted values.
        """
        return self.bias + np.dot(X, self.weights)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the test data.

        Parameters:
        X_test (array-like): Test feature matrix.
        y_test (array-like): Test target vector.

        Returns:
        tuple: (mse_score, accuracy_score) - Mean Squared Error and ratio-based accuracy metric.
        """
        y_pred = self.predict(X_test)
        mse_score = mse(y_test, y_pred)
        accuracy_score = accuracy(y_test, y_pred)

        return mse_score, accuracy_score
