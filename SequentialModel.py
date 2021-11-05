from Dense import Dense
from Input import Input
from LSTM import LSTM
import random
import numpy as np
import math
from ActivationFunction import ActivationFunction
import re

class Sequential:
  def __init__(self, layers=[]):
    '''
    Konstruktor untuk Sequential model.

    Parameters:
        layers (optional): layer yang akan dimasukkan dalam model.
   '''
    self.layers = []
    if (len(layers) > 0):
      # Pushing layers in the param into stack.
      for layer in layers:
        self.add(layer)


  def _get_layers_(self):
    '''
    Mendapatkan layer-layer dalam model.

    Returns:
        array of objects (layer dalam model).
    '''
    return self.layers

  def add(self, layer):
    '''
    Menambahkan layer baru pada stack.
    
    Restrictions:
    - First layer could only be LSTM or Input.
    - Input and LSTM layer could only be inserted as the first layer.
    '''
    ## Add input layer
    if (type(layer) is Input):
      #### (consider if input is not the first layer)
      if (len(self.layers) > 0):
        print("Input layer hanya bisa ditambah di awal.")
        return

    ## Add LSTM layer
    elif (type(layer) is LSTM):
      #### (consider if LSTM is not the first layer)
      if (len(self.layers) > 0):
        print("LSTM layer hanya bisa ditambah di awal.")
        return
    
    ## Add other layers (not an Input nor LSTM layer)
    elif (len(self.layers) == 0):
      print("Layer pertama hanya bisa berupa input atau LSTM layer.")
      return
    else:
      ## Add dense layer
      if (type(layer) is Dense):
        if layer.weight is None:
          weight_temp=[[random.uniform(0, 1) for i in range(layer.n_neuron)]] # bobot bias
          for i in range(self.layers[-1].shape[1]): #shape layer sebelumnya
            weight_temp.append([random.uniform(0, 1) for j in range(layer.n_neuron)]) #jumlah neuron pada layer ini 
          layer.weight = np.array(weight_temp)
        # layer.bias_weight = np.array([1 for i in range(layer.n_neuron)]) #weight untuk bias


    # Push the new layer into stack.
    self.layers.append(layer)
    print("Menambahkan layer ", layer.layerType)

  def pop_layer(self):
    """
    Mengeluarkan layer terakhir dalam list layer model dan mengembalikannya
    sebagai hasil fungsi.

    Returns:
      Object layer terakhir (jika list tidak kosong).
    """
    return self.layers.pop()

  def clear_layers(self):
    """
      Mengosongkan list of layers model.
    """
    self.layers.clear()

  def predict(self, input):
    '''
      Fungsi yang melakukan forward 1 data

      Parameters:
        input: data untuk dimasukkan ke arsitektur neural network untuk menghasilkan output

      Returns:
        output dari layer terakhir yang adalah list angka dengan jumlah angka sejumlah neuron pada layer terakhir
    '''
    # Panggil forward untuk layer pertama (layer input atau LSTM)
    if (type(self.layers[0]) is Input):
      self.layers[0].output = input
    else: #LSTM
      self.layers[0].output = self.layers[0].forward(input)
    # Panggil forward untuk layer sisanya
    for i in range(1, len(self.layers)):
      self.layers[i].output = self.layers[i].forward(self.layers[i-1].output)
    # Return output layer terakhir
    return self.layers[-1].output


  def summary(self):
    '''
      Fungsi yang mencetak summary dari aristektur neural network meliputi output shape dan jumlah parameter dari masing-masing layer
    '''
    print ("{:<30} {:<30} {:<30}".format('Layer (type)','Output Shape','Param #'))
    #iterate all layers
    total_params = 0
    for i in range(len(self.layers)):
      weightTemp = 0
      if (type(self.layers[i]) is Dense):
        weightTemp = self.layers[i].weight.shape[0] * self.layers[i].weight.shape[1]
      elif (type(self.layers[i]) is LSTM):
        # (m+n+1)*4*n
        # n: unit LSTM
        # m: dimensi input
        weightTemp = (self.layers[i].units + self.layers[i].input_shape[1] + 1)* 4 * self.layers[i].units
      total_params += weightTemp  
      print("{:<30} {:<30} {:<30}".format(str(type(self.layers[i])), str(self.layers[i].shape), str(weightTemp)))
    print("Total params: ", total_params)


if __name__ == "__main__":
  import pandas as pd
  import numpy as np
  import keras
  # from keras import Sequential
  import keras.layers as klayers
  # from keras.layers import LSTM, Dense

  df = pd.read_csv('bitcoin_price_Training - Training.csv')  
  print(df.head())

  #Hapus karakter "," pada string angka pemisah ribuan
  df['Volume'] = df['Volume'].str.replace(",","")
  # print(df['Volume'].head())

  df['Market Cap'] = df['Market Cap'].str.replace(",","")
  # print(df['Market Cap'].head())

  data = df[['Open','High','Low','Close','Volume','Market Cap']]
  X = data.values[0:32]
  X = np.asarray(X).astype('float32')
  X = X.reshape(1, X.shape[0], X.shape[1])
  print(X.shape)

  # Keras Implementation
  # model = keras.Sequential() #initialize sequential model
  # model.add(klayers.LSTM(10, input_shape=(32,6))) #LSTM layer with 10 neurons
  # model.add(klayers.Dense(6, activation='linear')) #Dense output layer with 1 neuron, linear activation
  # print(model.predict(X[:1]))
  # model.summary()

  # Self-Implementation
  selfmodel = Sequential()
  selfmodel.add(LSTM(10, input_shape=(32,6)))
  selfmodel.add(Dense(6, activation='linear'))
  # print(selfmodel.predict(X[:1])) Belum bisa
  selfmodel.summary()
  