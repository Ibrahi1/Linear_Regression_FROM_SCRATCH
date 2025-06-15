import numpy as np


class LinearRegression:
    def  __init__(self, lr=0.001, n_iters=500):
        self.lr = lr
        self.n_iters=n_iters
        self.weighs = None
        self.bias=None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weighs= np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weighs) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weighs -=self.lr*dw
            self.bias -=self.lr*db

    def predict(self, X):
        y_predicted = np.dot(X, self.weighs) + self.bias
        return y_predicted

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)