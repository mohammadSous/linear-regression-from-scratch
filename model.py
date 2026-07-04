import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01,n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []
        self.weights = None
        self.bias = None
   
    def fit(self,X_train,y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        updated_weights, updated_bias = self._gradient_descent(X_train, y_train)

        self.weights = updated_weights
        self.bias = updated_bias

        return self
    
    def predict(self, X_test):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        return (X_test @ self.weights) + self.bias
    


    def _compute_cost(self, y_true, y_pred):
        n_samples = len(y_true)
        return (1 / (2 * n_samples)) * np.sum((y_pred - y_true) ** 2)
    

    
    def _gradient_descent(self, X_train, y_train):
        weights = self.weights
        bias = self.bias
        n_samples = X_train.shape[0]

        for _ in range(self.n_iterations):

            y_pred = X_train @ weights + bias

            cost = self._compute_cost(y_train, y_pred)
            self.cost_history.append(cost)
               
            dw = (1 / n_samples) * (X_train.T @ (y_pred - y_train))
            weights = weights - self.learning_rate * dw

            db = (1 / n_samples) * np.sum(y_pred - y_train)
            bias = bias - self.learning_rate * db

        return weights, bias
    

    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_total)

        return {
            "mae": mae,
            "mse": mse,
            "r2": r2
        }