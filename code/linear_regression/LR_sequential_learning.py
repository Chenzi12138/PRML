# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 19:15
# @Author  : Ming
# @File    : LR_sequential_learning.py
import numpy as np
from base_regression import Regression


class LrSequentialLearning(Regression):
    def __init__(self, max_iteration=500, learning_rate=0.01, epsilon=0.1):
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.w = None
