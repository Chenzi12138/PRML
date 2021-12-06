# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 11:18
# @Author  : Ming
# @File    : test_linear_regression.py
import unittest
import numpy as np
from linear_regression.linear_regression import LinearRegression
from sklearn.datasets import load_iris

iris = load_iris()


class TestLinearRegression(unittest.TestCase):
    def test_fit(self):
        x_train = iris.data
        y_train = iris.target
        model = LinearRegression()
        model.fit(x_train, y_train)
        # self.assertTrue(np.allclose(model.w, x_train[1]))

    def test_predict(self):
        x_train = iris.data
        y_train = iris.target
        model = LinearRegression()
        model.fit(x_train, y_train)

        actual = model.predict(x_train[101:])
        # self.assertTrue(np.allclose(actual, x_train[1]))


if __name__ == "__main__":
    unittest.main()
