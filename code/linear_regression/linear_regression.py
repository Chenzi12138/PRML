# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 11:00
# @Author  : Ming
# @File    : linear_regression.py
import numpy as np
from base_regression import Regression


class LinearRegression(Regression):
    """
    Linear regression model.
    y = x @ w  -> {w ~(1,D),x(D,1),X~(N,D)}
    t ~ N(t|x @ w,var)
    """

    def __init__(self):
        self.w = None
        self.var = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Perform least-squares fitting.
        :param x_train: [np.ndarray] training independent variable (N,D)
        :param y_train: [np.ndarray] training dependent variable (N,)
        :return:
        """
        inv = np.linalg.inv(x_train.T @ x_train) @ x_train.T
        self.w = inv @ y_train
        # or use "np.linalg.pinv" to compute the (Moore-Penrose) pseudo-inverse of a matrix, -> (D,N)
        self.var = np.mean(np.square(x_train @ self.w - y_train))

    def predict(self, x: np.ndarray):
        """
        Return prediction value given input
        :param x: [np.ndarray] samples to predict(N,D)
        :return: [np.ndarray] prediction of each sample (N,)
        """
        y = x @ self.w
        return y
def test():
    x_train = np.array([-1, 0, 1]).reshape(-1, 1)
    y_train = np.array([-2, 0, 2])
    model = LinearRegression()
    model.fit(x_train, y_train)
    actual = model.predict(np.array([[3]]))
    print(actual)


if __name__ == "__main__":
    test()