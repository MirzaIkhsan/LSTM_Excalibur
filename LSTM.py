import numpy as np
from numpy import random
from Cell import Cell

from ActivationFunction import ActivationFunction


class LSTM:
    def __init__(self, units, activation='tanh', input_shape=None) -> None:
    # units adalah jumlah neuron pada hidden layernya
    # input shape berisi tuple (x, y) dimana x adalah jumlah timestep dan y adalah jumlah fitur
        if(input_shape is None):
            raise Exception('Input shape can\'t be None')

        if(type(input_shape) is not tuple):
            raise Exception('Input shape must be a tuple')

        if(units == 0):
            raise Exception('Units can\'t be zero')

        self.layerType = 'LSTM'
        self.units = units
        self.input_shape = input_shape
        self.shape = (None, units)
        # self.activation = activation -> kita kyknya gk perlu pake activation ya soalnya udah automatis pake tanh dan sigmoid itu? yang di cellstatenya
        # self.cell_state = np.array([[0]])
        # self.hidden_state = np.array([[0]])

        self.prev_cell_state = self._zero_init(n_range=self.units * 1, shape=(self.units, 1))
        self.prev_hidden_state = self._zero_init(n_range=self.units * 1, shape=(self.units, 1))

        # Forget matrices
        self.Uf = self._random_init(
            n_range=self.units * self.input_shape[1],
            shape=(self.units, self.input_shape[1])
        )
        self.Wf = self._random_init(
            n_range=self.units * self.units,
            shape=(self.units, self.units)
        )
        self.bf = self._random_init(
            n_range=self.units * 1,
            shape=(self.units, 1)
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
        self.bi = self._random_init(
            n_range=self.units * 1,
            shape=(self.units, 1)
        )

        # Cell state matrices
        self.Uc = self._random_init(
            n_range=self.units * self.input_shape[1],
            shape=(self.units, self.input_shape[1])
        )
        self.Wc = self._random_init(
            n_range=self.units * self.units,
            shape=(self.units, self.units)
        )
        self.bct = self._random_init(
            n_range=self.units * 1,
            shape=(self.units, 1)
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
        self.bo = self._random_init(
            n_range=self.units * 1,
            shape=(self.units, 1)
        )

        self._generate_cells()

    def _generate_cells(self):
        self.cells = []
        for i in range(self.units):
            Ufi = self.Uf[i].reshape(1, self.Uf[i].shape[0])
            Wfi = self.Wf[i].reshape(1, self.Wf[i].shape[0])
            bfi = self.bf[i].reshape(1, 1)
            Uii = self.Ui[i].reshape(1, self.Ui[i].shape[0])
            Wii = self.Wi[i].reshape(1, self.Wi[i].shape[0])
            bii = self.bi[i].reshape(1, 1)
            Uci = self.Uc[i].reshape(1, self.Uc[i].shape[0])
            Wci = self.Wc[i].reshape(1, self.Wc[i].shape[0])
            bcti = self.bct[i].reshape(1, 1)
            Uoi = self.Uc[i].reshape(1, self.Uo[i].shape[0])
            Woi = self.Wc[i].reshape(1, self.Wo[i].shape[0])
            boi = self.bo[i].reshape(1, 1)
            prev_cell_state_i = self.prev_cell_state[i].reshape(1, 1)
            self.cells.append(Cell(Ufi, Wfi,
                                    Uii, Wii,
                                    Uci, Wci,
                                    Uoi, Woi, 
                                    prev_cell_state_i, self.prev_hidden_state, bfi,
                                    bii, bcti, boi))

    def _random_init(self, n_range, shape, add_bias=False):
        if(add_bias):
            return np.array([random.uniform(0, 1)
                             for _ in range(n_range + self.units)])\
                .reshape(shape[0], shape[1] + 1)
        return np.array([random.uniform(0, 1)
                         for _ in range(n_range)])\
            .reshape(shape)

    def _zero_init(self, n_range, shape):
        return np.array([0 for _ in range(n_range)])\
            .reshape(shape)

    def process_timestep(self, data):
        #iterasi per units -> dapetin h_prev dan cell state  
        for i in range(self.input_shape[0]):
            output_value = []
            xi = data[i].reshape(1,data[i].shape[0])
            for j in range(self.units):
                self.cells[j].calculate_cell(xi) 
                self.cells[j].calculate_hidden(xi)
                output_value.append(self.cells[j].calculate_output(xi)[0][0]) #[0][0] karena hasilnya kan dalam shape (1,1) jadi ambil valuenya     caranya gini

            self.output_value = np.array(output_value).reshape(self.units, 1)

            # print("======Prev cell state========")
            # print(self.prev_cell_state)
            # print("======Prev hidden state======")
            # print(self.prev_hidden_state)
            # print("======Prev output============")
            # print(self.output_value)

    def forward(self, data):
        self.process_timestep(data)
        # print("Output untuk dikasih ke Dense")
        # print(self.output_value)
        return self.output_value        
    # def forward(self, input):
    #     if(type(input) is not np.ndarray):
    #         input = np.array(input)

    #     if(input.shape != self.input_shape):
    #         raise Exception(
    #             'The input shape is not the same as the shape of the input')

    #     for each_row_idx in range(len(input)):
    #         res = self.process_timestep(input[each_row_idx])

    #         if (each_row_idx == len(input)-1):
    #             for cell in res:
    #                 print(cell.hidden)


if __name__ == "__main__":
    input_data = np.arange(0, 6).reshape(3, 2) #3 timestep dan 2 fitur

    lstm = LSTM(10, input_shape=(3, 2))

    lstm.forward(input_data)

    # for i in range(lstm.units):
    #     xi = input_data[i].reshape(1,input_data[i].shape[0])
    #     print("==========Cell State============")
    #     print(lstm.cells[i].calculate_cell(xi))
    #     print("==========Hidden State============")
    #     print(lstm.cells[i].calculate_hidden(xi))
    #     print("==========Output============")
    #     print(lstm.cells[i].calculate_output(xi))

    # for i in range(lstm.units):
    #     xi = input_data[i].reshape(1,input_data[i].shape[0])
    #     print("==========Cell State============")
    #     print(lstm.cells[i].calculate_cell(xi))
    #     print("==========Hidden State============")
    #     print(lstm.cells[i].calculate_hidden(xi))
    #     print("==========Output============")
    #     print(lstm.cells[i].calculate_output(xi))
    # print("=====Forget Gate=====")
    # print(lstm.Uf.shape)
    # print(lstm.Wf.shape)
    # print(lstm.bf.shape)
    # print("=====================")
    # print("=====Input Gate=====")
    # print(lstm.Ui.shape)
    # print(lstm.Wi.shape)
    # print(lstm.bi.shape)
    # print("=====================")
    # print("=====Cell Gate=====")
    # print(lstm.Uc.shape)
    # print(lstm.Wc.shape)
    # print(lstm.bc.shape)
    # print("=====================")
    # print("=====Output Gate=====")
    # print(lstm.Uo.shape)
    # print(lstm.Wo.shape)
    # print(lstm.bo.shape)
    # print("=====================")
    # lstm.forward(input_data)
    # print(input[0])
    #lstm.process_timestep(input[0])

    #print(lstm.Uf)
    #print(lstm.Wf)

    # pass
