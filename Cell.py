from ForgetGate import ForgetGate
from OutputGate import OutputGate
from InputGate import InputGate
import numpy as np

class Cell:
    def __init__(self,
                Uf, Wf,
                Ui, Wi ,
                Uc, Wc,
                Uo, Wo,
                prev_cell_state,
                prev_hidden,
                bf, 
                bi,
                bct,
                bo,
                ):
        self.forget_gate = ForgetGate(Uf, Wf, bf)
        self.input_gate = InputGate(Ui, Wi, Uc, Wc, bi, bct)
        self.prev_cell_state = prev_cell_state
        self.output_gate = OutputGate(Uo, Wo, bo)
        self.prev_hidden = prev_hidden

    # def calculate_forget(self, U, W, bias):
    #     self.forget_gate.score()
    
    # def calculate_output(self, x, h_prev):
    #     pass

    def calculate_cell(self, x):
        self.cell_state = np.dot(self.forget_gate.score(x, self.prev_hidden), 
                                self.prev_cell_state) + np.dot(self.input_gate.score_it(x, self.prev_hidden), 
                                                        self.input_gate.score_ct(x, self.prev_hidden))
        return self.cell_state

    def calculate_hidden(self, x):
        self.hidden = self.output_gate.score_ht(self.cell_state, x, self.prev_hidden)
        
        return self.hidden

    def calculate_output(self, x):
        self.output = self.output_gate.score_ot(x, self.prev_hidden)

        return self.output

if __name__ == "__main__":
    xt = np.array([[1, 2]])
    uf = np.array([[0.7, 0.45]])
    wf = np.array([[0.1]])
    bf = 0.15
    ui = np.array([[0.95, 0.8]])
    wi = np.array([[0.8]])
    bi = 0.65
    uc = np.array([[0.45, 0.25]])
    wc = np.array([[0.15]])
    bct = 0.2
    uo = np.array([[0.6, 0.4]])
    wo = np.array([[0.25]])
    bo = 0.1
    prev_cell_state = np.array([[0]])
    prev_hidden = np.array([[0]])

    cell_example = Cell(uf, wf, ui, wi, uc, wc, uo, wo, prev_cell_state, prev_hidden, bf, bi, bct, bo)

    print("Cell state: ", cell_example.calculate_cell(xt))

    print("Cell hidden: ", cell_example.calculate_hidden(xt))

    print("Cell output: ", cell_example.calculate_output(xt))
