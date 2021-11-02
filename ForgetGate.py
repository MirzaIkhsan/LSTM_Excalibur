import math
import numpy as np

from ActivationFunction import ActivationFunction

class ForgetGate():
    def __init__(self, U, W, bias):
        self.U = U
        self.W = W
        self.bias = bias

    def _sigmoid(self, number):
        #return ActivationFunction.sigmoid(number)
        return 1 / (1 + np.exp(-number))

    def score(self, x, h_prev):
        self.value = np.array(self._sigmoid(np.matmul(self.U, x.transpose()) + np.matmul(self.W, h_prev) + self.bias))
        return self.value

if __name__ == "__main__":
    xt = np.array([[1, 2]])
    uf = np.array([[0.7, 0.45]])
    wf = np.array([[0.1]])
    h_prev = np.array([[0]])
    bf = 0.15

    forget_gate = ForgetGate(uf, wf, bf)
    
    print("Forget Value: ", forget_gate.score(xt, h_prev))

