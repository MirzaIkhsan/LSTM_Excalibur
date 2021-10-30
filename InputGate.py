import math
import numpy as np

from ActivationFunction import ActivationFunction

class InputGate():
    def __init__(self, Ui, Wi, Uc, Wc, bi=1, bct=1):
        self.Ui = Ui
        self.Wi = Wi
        self.Uc = Uc
        self.Wc = Wc
        self.bi = bi
        self.bct = bct

    def it(self, x, h_prev):
        res = np.matmul(self.Ui, x.transpose()) + np.matmul(self.Wi, h_prev) + self.bi
        # if type(res) == np.float64:
        #     print(res)
        #     res = np.array(res)
        self.it = ActivationFunction.sigmoid_num(res)
        return self.it

    def ct(self, x, h_prev):
        self.ct = np.tanh(np.matmul(self.Uc, x.transpose()) + np.matmul(self.Wc, h_prev) + self.bct)
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