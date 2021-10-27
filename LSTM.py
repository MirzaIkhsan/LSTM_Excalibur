import numpy as np
from numpy import random


class LSTM:
    def __init__(self, units, activation='tanh', input_shape=None) -> None:
        if(input_shape is None):
            raise Exception('Input shape can\'t be None')

        if(type(input_shape) is not tuple):
            raise Exception('Input shape must be a tuple')

        if(units == 0):
            raise Exception('Units can\'t be zero')

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

    def _random_init(self, n_range, shape, add_bias=False):
        if(add_bias):
            return np.array([random.uniform(0, 1)
                             for _ in range(n_range + self.units)])\
                .reshape(shape[0], shape[1] + 1)
        return np.array([random.uniform(0, 1)
                         for _ in range(n_range)])\
            .reshape(shape)

    def process_timestep(self, data):
        for each_unit in range(self.units):
            input = self.hidden_state + data
            
        pass

    def forward(self, input):
        if(type(input) is not np.ndarray):
            input = np.array(input)

        if(input.shape != self.input_shape):
            raise Exception(
                'The input shape is not the same as the shape of the input')

        for each_row_idx in range(len(input)):

            for each_unit in range(self.units):

                pass
            pass


if __name__ == "__main__":
    input = np.arange(0, 100).reshape(50, 2)

    lstm = LSTM(1, input_shape=(50, 2))
    # lstm.forward(input)
    # print(input[0])
    lstm.process_timestep(input[0])

    print(lstm.Uf)
    print(lstm.Wf)

    pass
