import numpy as np
from numpy import random
from Cell import Cell

from ActivationFunction import ActivationFunction


class LSTM:
    def __init__(self, units, activation='tanh', input_shape=None) -> None:
        if(input_shape is None):
            raise Exception('Input shape can\'t be None')

        if(type(input_shape) is not tuple):
            raise Exception('Input shape must be a tuple')

        if(units == 0):
            raise Exception('Units can\'t be zero')

        self.type = 'LSTM'
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.cell_state = 0
        self.hidden_state = 0

        # Forget matrices
        self.Uf = self._random_init(
            n_range=self.units * self.input_shape[1],
            shape=(self.units, self.input_shape[1])
        )
        self.Wf = self._random_init(
            n_range=self.units * self.units,
            shape=(self.units, self.units)
        )

        # Input matrices
        self.Ui = self._random_init(
            n_range=self.units * self.input_shape[1],
            shape=(self.units, self.input_shape[1])
        )
        self.Wi = self._random_init(
            n_range=self.units * self.units,
            shape=(self.units, self.units)
        )

        # Cell state??? matrices
        self.Uc = self._random_init(
            n_range=self.units * self.input_shape[1],
            shape=(self.units, self.input_shape[1])
        )
        self.Wc = self._random_init(
            n_range=self.units * self.units,
            shape=(self.units, self.units)
        )

        # Output matrices
        self.Uo = self._random_init(
            n_range=self.units * self.input_shape[1],
            shape=(self.units, self.input_shape[1])
        )
        self.Wo = self._random_init(
            n_range=self.units * self.units,
            shape=(self.units, self.units)
        )

        self.cells = []
        for i in range(self.units):
            self.cells.append(Cell(self.Uf[i], self.Wf[i],
                                    self.Ui[i], self.Wi[i],
                                    self.Uc[i], self.Wc[i],
                                    self.Uo[i], self.Wo[i]))

    def _random_init(self, n_range, shape, add_bias=False):
        if(add_bias):
            return np.array([random.uniform(0, 1)
                             for _ in range(n_range + self.units)])\
                .reshape(shape[0], shape[1] + 1)
        return np.array([random.uniform(0, 1)
                         for _ in range(n_range)])\
            .reshape(shape)

    def process_timestep(self, data):
        for cell in self.cells:
            cell.calculate_cell(data)
            cell.calculate_hidden()
            #input = self.hidden_state + data
            # input_gate = ActivationFunction.sigmoid(input)
            # self.cell_state = self.cell_state * input_gate
            # forget_gate = input_gate * np.tanh(input)
            # self.cell_state += forget_gate
            # output_gate = input_gate * np.tanh(self.cell_state)
        # return self.cell_state, output_gate

    def forward(self, input):
        if(type(input) is not np.ndarray):
            input = np.array(input)

        if(input.shape != self.input_shape):
            raise Exception(
                'The input shape is not the same as the shape of the input')

        for each_row_idx in range(len(input)):
            res = self.process_timestep(input[each_row_idx])

            if (each_row_idx == len(input)-1):
                print(res)


if __name__ == "__main__":
    input = np.arange(0, 100).reshape(50, 2)

    lstm = LSTM(1, input_shape=(50, 2))
    lstm.forward(input)
    # print(input[0])
    #lstm.process_timestep(input[0])

    #print(lstm.Uf)
    #print(lstm.Wf)

    pass
