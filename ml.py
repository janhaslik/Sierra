import numpy as np


def mse(y, predictions):
    return np.mean(np.square(y - predictions))


def accuracy(y, predictions):
    return np.mean(predictions / y)


def train_test_split(X, y, split=0.2, shuffle=False):
    split_index = int(1 - split * len(X))
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    if shuffle:
        np.random.shuffle(X_train)
        np.random.shuffle(y_train)
        np.random.shuffle(X_test)
        np.random.shuffle(y_test)

    return X_train, X_test, y_train, y_test


class LinearRegression:
    def __init__(self, iterations=1000, learning_rate=0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.weights = None
        self.bias = 0

        self.history = []

    def fit(self, X, y):
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
        y_pred = self.predict(self.X)
        errors = y_pred - self.y

        # Calculate gradients
        dW = (1 / self.X.shape[0]) * np.dot(self.X.T, errors)
        dB = (1 / self.X.shape[0]) * np.sum(y_pred - self.y)

        self.weights = self.weights - self.learning_rate * dW
        self.bias = self.bias - self.learning_rate * dB

    def predict(self, X):
        return self.bias + np.dot(X, self.weights)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse_score = mse(y_test, y_pred)
        accuracy_score = accuracy(y_test, y_pred)

        return mse_score, accuracy_score
