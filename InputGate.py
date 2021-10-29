import math
import numpy as np

from ActivationFunction import ActivationFunction

class InputGate():
    def __init__(self, U, W, bi=1, bct=1):
        self.U = U
        self.W = W
        self.bi = bi
        self.bct = bct

    def it(self, x, h_prev):
        self.it = ActivationFunction.sigmoid(np.matmul(self.U, x.transpose()) + np.matmul(self.W, h_prev) + self.bi)
        return self.it

    def ct(self, x, h_prev):
        self.ct = np.tanh(np.matmul(self.U, x.transpose()) + np.matmul(self.W, h_prev) + self.bct)
        return self.ct


if __name__ == "__main__":
    xt = np.array([1, 2])
    ui = np.array([0.6, 0.4])
    wi = np.array([0.25])
    h_prev = np.array([0])
    bi = np.array([0.1])
    bct = np.array([0.1])
    input_example = InputGate(ui, wi, bi, bct)
    print("Input: ", input_example.it(xt, h_prev))
    print("Candidate: ", input_example.ct(xt, h_prev))