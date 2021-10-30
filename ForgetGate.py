import math
import numpy as np

from ActivationFunction import ActivationFunction

class ForgetGate():
    def __init__(self, U, W, bias=1):
        self.U = U
        self.W = W
        self.bias = bias

    def _sigmoid(self, number):
        #return ActivationFunction.sigmoid(number)
        return 1 / (1 + np.exp(-number))

    def score(self, x, h_prev):
        self.value = np.array(self._sigmoid(np.matmul(self.U, x) + np.matmul(self.W, h_prev) + self.bias))

        return self.value
