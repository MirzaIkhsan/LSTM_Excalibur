import numpy as np
import math
from ActivationFunction import ActivationFunction
from utils import paddMatrix

class Dense:
    '''
    Kelas yang mengimplementasikan Dense layer 
    pada ANN.
    '''

    def __init__(self, n_neuron=None, weight=None, activation='linear', bias=1):
        '''
        Konstruktor kelas Dense

        Parameters:
            n_neuron: int
                Banyak neuron pada layer.
            weight: array of number
                Weight untuk seluruh neuron dalam layer.
            activation: str
                Nama fungsi aktivasi yang dipakai. ('sigmoid'/'relu'/'linear'/'softmax')
            bias: number
                Bias untuk layer.
        '''
        self.activation = ActivationFunction.get_by_name(activation)
        self.layerType = "Dense"
        self.output = []
        self.shape = (None, n_neuron)
        self.n_neuron = n_neuron
        self.bias = bias
        if weight is not None:
            self.weight = np.array(weight)
        else:
            self.weight = None

    def get_activation_name(self):
        '''
        Mengembalikan nama dari fungsi aktivasi layer.

        Returns:
            Nama fungsi aktivasi (str).
        '''
        return self.activation.__name__


    def forward(self, input): 
        '''
        Fungsi yang melakukan forward propagation 
        pada input dalam Dense layer.

        Parameters:
            input: array of array of number
                Input yang akan diproses dengan forward propagation.
            
        Returns:
            Array of array of number berisi hasil forward propagation pada
            Dense layer.
        '''
        if self.weight.shape[1] == self.n_neuron:
            temp_output = np.full(shape=(self.n_neuron), fill_value=0).astype(float)

            for i in range(len(list(input))+1):
                # Iterate through neuron
                for j in range(self.n_neuron):
                    if i == 0:
                        temp_output[j] += self.weight[i][j] * self.bias
                    else:
                        temp_output[j] += self.weight[i][j] * float(input[i-1])
            return self.activation(temp_output)
        else:
            print("Input attributes dimension doesn't match with weight dimension")

    def backprop_output(self, target, prev_layer):
        '''
        Fungsi yang melakukan update error term dan derivatives untuk layer output 

        Parameters:
            target: nilai target misalkan untuk batch dengan dua data
                untuk sigmoid: [[0],[1]] 
                untuk softmax: [[0,0,0,1],[0,1,0,0]]
            prev_layer: layer sebelum layer terakhir (index: -2)
        Result:
            self.error_term (dE/dNet) dari layer output yang adalah numpy array berisi angka sesuai dengan jumlah neuron pada output layer
            self.derivatives (dNet/dW) dari layer output yang adalah numpy array berisi angka dengan shape sesuai dengan shape bobot dari output layer
        '''
        if(self.activation == ActivationFunction.get_by_name("sigmoid")):
            final_output = self.output.reshape(100,1)
            target = target.reshape(100,1)
            previous_output = prev_layer.output
            padded_previous_output = []
            for i in range(previous_output.shape[0]):
              padded_previous_output.append(np.append(previous_output[i],prev_layer.bias))
            padded_previous_output = np.array(padded_previous_output)
            error_term = final_output*(1-final_output)*(target-final_output) #dE/dNet
            self.error_term = error_term
            self.derivatives = np.matmul(np.transpose(padded_previous_output), error_term) #de/dW = dE/dNet * dNet/dW
            self.derivatives = np.average(self.derivatives, axis=0)
            self.derivatives = np.expand_dims(self.derivatives, axis=1) #dE/dW
        
        if(self.activation == ActivationFunction.get_by_name("softmax")):
            final_output = self.output
            self.error_term = np.copy(final_output)
            previous_output = prev_layer.output

            print("===Target===")
            print(target.shape)
            print("============")
            padded_previous_output = []
            for i in range(previous_output.shape[0]):
              padded_previous_output.append(np.append(previous_output[i],prev_layer.bias))
            padded_previous_output = np.array(padded_previous_output)

            #dE/dNet
            for i in range(final_output.shape[0]):
                for j in range(final_output.shape[1]):
                    if(target[i][j] == 1):
                        try:
                            self.error_term[i][j] = math.log(final_output[i][j], math.exp(1))
                        except:
                            self.error_term[i][j] = 1    
            #dE/dW

            self.derivatives = []
            for i in range(self.error_term.shape[0]):
                self.derivatives.append(np.multiply.outer(padded_previous_output[i], self.error_term[i])) #de/dW = dE/dNet * dNet/dW
            self.derivatives = np.array(self.derivatives)
            self.derivatives = np.average(self.derivatives, axis=0)

    def backprop_hidden(self, prev_layer, next_layer):
        '''
        Fungsi yang melakukan update error term dan derivatives untuk layer hidden 

        Parameters:
            prevLayer: layer sebelum hidden layer ini
            nextLayer: layer setelah hidden layer ini
        Result:
            self.error_term (dE/dNet) dari layer hidden yang adalah numpy array berisi angka sesuai dengan jumlah neuron pada hidden layer
            self.derivatives (dNet/dW) dari layer hidden yang adalah numpy array berisi angka dengan shape sesuai dengan shape bobot dari hidden layer
        '''
        #Hitung error term dari layer
        error_term=[]
        error_term_next_layer_batch = next_layer.error_term
        output_current_layer_batch = self.output
        weight_next_layer = next_layer.weight[1:] #buang weight untuk bias

        # Iterasi terhadap data dalam batch
        for i in range(error_term_next_layer_batch.shape[0]):
          error_term_next_layer = error_term_next_layer_batch[i] # Ambil error term layer setelahnya untuk data_batch ke i
          output_current_layer = output_current_layer_batch[i] # Ambil output untuk data_batch ke i
          final_output_current_layer = output_current_layer*(1-output_current_layer) # Turunan fungsi aktivasi thd net
          error_term_temp = -1*np.matmul(error_term_next_layer, final_output_current_layer*np.transpose(weight_next_layer)) # Error term per neuron dari layer tsb
          error_term.append(error_term_temp)
        error_term = np.array(error_term)
        self.error_term = error_term #dE/dnet
        
        #Hitung nilai turunan errornya terhadap bobot (format turunan mengikuti format bobot jadi nnt mengupdatenya gampang)
        prev_layer_output = prev_layer.output
        padded_prev_layer_output = paddMatrix(prev_layer_output)
        derivatives = []
        for i in range(padded_prev_layer_output.shape[0]):
            derivatives.append(np.multiply.outer(padded_prev_layer_output[i], error_term[i])) #dE/dnet(error term) * dnet/dW
        derivatives = np.array(derivatives)
        derivatives = np.mean( derivatives, axis=0 )
        self.derivatives = derivatives

'''
    def forward(self, input): 
        
        #This function will receive input array of array of number, and
        #return array of array of number. The first dimension iterate through the amount of data,
        #and the second dimension iterate through data attributes
        
        # attr_size = len(input[0])
        # print(self.weight)
        # print(self.weight.shape[1])
        # print(self.n_neuron)
        print(input)
        if self.weight.shape[1] == self.n_neuron:
            temp_all = []
            #Iterate through data
            for items in input:
                temp_output = np.full(shape=(self.n_neuron), fill_value=0)
                # Iterate through input matrix and bias
                # print(items)
                for i in range(len(list(input))+1):
                    # Iterate through neuron
                    for j in range(self.n_neuron):
                        if j == 0:
                            temp_output[i][j] += self.weight[i][j] * self.bias
                        else:
                            temp_output[i][j] += self.weight[i][j] * items[i-1]
                temp_all.append(np.array(temp_output))
            self.output = self.activation(temp_all)
        # self.output = self.activation(np.matmul(input, self.weight) + self.bias_weight*1)
        else:
            print("Input attributes dimension doesn't match with weight dimension")

        return self.output
'''