from ForgetGate import ForgetGate
from Output import Output
from InputGate import InputGate
import numpy as np

class Cell:
    def __init__(self,
                Uf, Wf,
                Ui, Wi,
                Uc, Wc,
                Uo, Wo):
        self.forget_gate = ForgetGate(Uf, Wf)
        self.input_gate = InputGate(Ui, Wi, Uc, Wc)
        self.cell_state = [0]
        self.output = Output(Uo, Wo)
        self.hidden = [0]

    # def calculate_forget(self, U, W, bias):
    #     self.forget_gate.score()
    
    # def calculate_output(self, x, h_prev):
    #     pass

    def calculate_cell(self, x):
        self.cell_state = np.dot(self.forget_gate.score(x, self.hidden), 
                                self.cell_state) + np.dot(self.input_gate.score_it(x, self.hidden), 
                                                        self.input_gate.score_ct(x, self.hidden))
        
        return self.cell_state

    def calculate_hidden(self, x):
        self.hidden = self.output.score_ht(self.cell_state, x, self.hidden)
        
        return self.hidden