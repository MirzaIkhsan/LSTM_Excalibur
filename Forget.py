import math
import numpy as np

from ActivationFunction import ActivationFunction


class Forget():
    def __init__(self, U, W, bias=1):
        self.U = U
        self.W = W
        self.bias = bias

    def _sigmoid(self, number):
        return ActivationFunction.sigmoid(number)

    def score(self, x, h_prev):
        return self._sigmoid(np.matmul(self.U, x) + np.matmul(self.W, h_prev) + self.bias)
