# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 21:07
# @Author  : Ming
# @File    : label_transformer.py
import numpy as np


class LabelTransformer(object):
    """
    Label encoder & decoder (one-hot encoding encoder and decoder)
    Attr:
        n_classes: [int] number of classes, K
    """

    def __init__(self, n_classes: int = None):
        self.n_classes = n_classes

    @property
    def n_classes(self):
        return self.__n_classes

    @n_classes.setter
    def n_classes(self, K):
        self.__n_classes = K
        self.__encoder = None if K is None else np.eye(K)

    @property
    def encoder(self):
        return self.__encoder

    def encode(self, class_index: np.ndarray):
        """
        encode class index into one-of-k code
        :param class_index: [np.ndarray] non-negative class index (N,), k \in [0,n_classes)
        :return: (N,K) np.ndarray, one-of-k encoding of input
        """
        if self.n_classes is None:
            self.n_classes = np.max(class_index) + 1
        return self.encoder[class_index]

    @staticmethod
    def decode(one_hot: np.ndarray):
        """
        decode one-of-k code into class index
        :param one_hot: [np.ndarray] one-of-k code, (N,K)
        :return: (N,) np.ndarray, class index
        """
        return np.argmax(one_hot, axis=1)


def test():
    transformer = LabelTransformer()
    print(transformer.encode(np.array([0, 1, 2, 3, 4])))
    print(transformer.decode(transformer.encode(np.array([0, 1, 2, 3]))))


if __name__ == "__main__":
    test()
