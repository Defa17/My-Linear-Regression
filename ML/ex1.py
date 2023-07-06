import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter, learning_rate, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights


    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X, y, verbose=False):
        """
        :param X: все фичи в виде датафрейма пандаса
        :param y: целевая переменная в виде пандасовской серии
        :param verbose: указывает на какой итерации выводить лог

        """

        X.insert(0, 'ones', [1] * X.shape[0], allow_duplicates=True)
        vector_weights = [1] * X.shape[1]

        for i in range(self.n_iter):
            y_pred = np.dot(X, vector_weights)
            mse = np.mean((y_pred - y) ** 2)
            gradient_descent = 2 * (y_pred - y) @ X / X.shape[0]
            vector_weights -= self.learning_rate * gradient_descent


            if verbose and i % 100 == 0:
                print(f'Iteration {i}: MSE={mse}')

        self.weights = vector_weights

    def get_coef(self):
        return np.array(self.weights[1:])

    def predict(self, X):
        X.insert(0, 'ones', [1] * X.shape[0], allow_duplicates=True)
        y_pred = np.dot(X, self.weights)
        return y_pred

