# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 19:48
# @Author  : Ming
# @File    : linear_regression_regularization.py
import numpy as np
from base_regression import Regression


class LrRegularization(Regression):
    def __init__(self, p: int = 1, alpha: float = 1.):
        """
        Initialize linear regression model with regularization. p=1 is the lasso;p=2 is the ridge
        :param p:
        :param alpha: [float] coefficient of the regularization term, by default 1
        """
        self.alpha = alpha
        self.p = p
        self.w = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
         Perform least-squares fitting.
         :param x_train: [np.ndarray] training independent variable (N,D)
         :param y_train: [np.ndarray] training dependent variable (N,)
         :return:
        """
        eye = np.eye(np.size(x_train, 1))
        inv = np.linalg.inv(self.alpha * eye + np.transpose(x_train) @ x_train) @ np.transpose(x_train)
        self.w = inv @ y_train

    def predict(self, x: np.ndarray):
        """
        Return the prediction value given input x.
        :param x: [np.ndarray] input variable (N,D)
        :return: [np.ndarray] prediction value of each sample (N,)
        """
        return x @ self.w


def test():
    x_train = np.array([-1, 0, 1]).reshape(-1, 1)
    y_train = np.array([-2, 0, 2])
    model = LrRegularization(p=2)
    model.fit(x_train, y_train)
    actual = model.predict(np.array([[3]]))
    print(actual)


if __name__ == "__main__":
    test()
