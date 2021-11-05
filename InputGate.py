import math
import numpy as np

from ActivationFunction import ActivationFunction

class InputGate():
    def __init__(self, Ui, Wi, Uc, Wc, bi, bct):
        '''
        Konstruktor untuk Input Gate.

        Parameters:
            Ui: Matriks bobot untuk menghitung input ke-t dari nilai x ke-t.
            Wi: Matriks bobot untuk menghitung input ke-t dari nilai h pada timestep sebelumnya.
            Uc: Matriks bobot untuk menghitung candidate ke-t dari nilai x ke-t.
            Wc: Matriks bobot untuk menghitung candidate ke-t dari nilai h pada timestep sebelumnya.
            bi: bias untuk input ke-t
            bct: bias untuk candidate
        '''
        self.Ui = Ui
        self.Wi = Wi
        self.Uc = Uc
        self.Wc = Wc
        self.bi = bi
        self.bct = bct

    def score_it(self, x, h_prev):
        '''
        Menghitung nilai i untuk timestep t.

        Parameters:
            x:      Nilai x (data masukan) untuk timestep t.
            h_prev: Nilai hidden (h) dari timestep sebelumnya. 

        Returns:
            Numpy array berisi hasil dari perhitungan.
        '''
        res = np.matmul(self.Ui, x.transpose()) + np.matmul(self.Wi, h_prev) + self.bi
        # if type(res) == np.float64:
        #     print(res)
        #     res = np.array(res)
        self.it = np.array(ActivationFunction.sigmoid_num(res))
        return self.it

    def score_ct(self, x, h_prev):
        '''
        Menghitung nilai candidate untuk timestep t.

        Parameters:
            x:      Nilai x (data masukan) untuk timestep t.
            h_prev: Nilai hidden (h) dari timestep sebelumnya. 

        Returns:
            Numpy array berisi hasil dari perhitungan.
        '''
        self.ct = np.tanh(np.matmul(self.Uc, x.transpose()) + np.matmul(self.Wc, h_prev) + self.bct)
        return self.ct


if __name__ == "__main__":
    xt = np.array([[1, 2]])
    ui = np.array([[0.95, 0.8]])
    wi = np.array([[0.8]])
    uc = np.array([[0.45, 0.25]])
    wc = np.array([[0.15]])
    h_prev = np.array([[0]])
    bi = 0.65
    bct = 0.2
    input_example = InputGate(ui, wi, uc, wc, bi, bct)
    print("Input: ", input_example.score_it(xt, h_prev))
    print("Candidate: ", input_example.score_ct(xt, h_prev))