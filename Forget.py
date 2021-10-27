import math
import numpy as np

class Forget():
    def __init__(self, U, W, bias=1):
        self.U = U
        self.W = W
        self.bias = bias

    def _sigmoid(self, number):
        return 1/(1+math.exp(-number))

    def score(self, x, h_prev):
        return self._sigmoid(np.matmul(self.U, x) + np.matmul(self.W, h_prev) + self.bias)
