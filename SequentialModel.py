from Dense import Dense
from Input import Input
# from Flatten import Flatten
# from Conv import Conv
# from Pooling import Pooling
from LSTM import LSTM
import random
import numpy as np
import math
# import utils
# from utils import paddMatrix
from ActivationFunction import ActivationFunction
import re

class Sequential:
  def __init__(self, layers=[]):
    '''Sequential model constructor'''
    
    self.layers = []
    if (len(layers) > 0):
      # Pushing layers in the param into stack.
      for layer in layers:
        self.add(layer)


  def _get_layers_(self):
    return self.layers

  def add(self, layer):
    '''
    Push a new layer into stack
    
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

      ## Add flatten layer
      # if (type(layer) is Flatten):
      #   temp = 1
      #   for shapeEl in self.layers[-1].shape[1:]:
      #     temp *= shapeEl
      #   layer.shape = (None, temp)
      
      ## Add convo layer
      # if (type(layer) is Conv):
      #   wRow = self.layers[-1].shape[2]
      #   wCol = self.layers[-1].shape[3]
      #   fRow = layer.kernel_size[1]
      #   fCol = layer.kernel_size[2]
      #   nFilter = layer.filter
      #   p = layer.padding
      #   sRow = layer.strides[1]
      #   sCol = layer.strides[2]
      #   VRow = math.floor((wRow-fRow+2*p)/sRow) + 1
      #   VCol = math.floor((wCol-fCol+2*p)/sCol) + 1
      #   layer.shape = (None, nFilter, VRow, VCol)
      
      ## Add pooling layer
      # if (type(layer) is Pooling):
      #   wRow = self.layers[-1].shape[2]
      #   wCol = self.layers[-1].shape[3]
      #   fRow = layer.filter_size[0]
      #   fCol = layer.filter_size[1]
      #   nFilter = self.layers[-1].shape[1]
      #   sRow = layer.strides[0]
      #   sCol = layer.strides[1]
      #   VRow = math.floor((wRow-fRow)/sRow) + 1
      #   VCol = math.floor((wCol-fCol)/sCol) + 1
      #   layer.shape = (None, nFilter, VRow, VCol)

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


  # def forward_batch(self, input):
  #   '''
  #     Fungsi yang melakukan forward batch data

  #     Parameters:
  #       input: data batch untuk dimasukkan ke arsitektur neural network untuk menghasilkan output 

  #     Returns:
  #       output dari layer terakhir yang adalah list dari list angka dengan jumlah angka sejumlah neuron pada layer terakhir sejumlah data pada batch data
  #   '''
  #   # Format input = list of input (list sebanyak batch)
  #   # Panggil forward untuk setiap batch
  #   # Init output setiap layer dengan list kosong
    
  #   print("Menghitung prediksi...")

  #   for i in range(len(self.layers)):
  #     self.layers[i].output = []

  #   # Fill output list dengan output dari setiap data dari batch
  #   for i in range(input.shape[0]):
  #     # Panggil forward untuk layer pertama (layer input)
  #     self.layers[0].output.append(input[i])

  #     # Panggil forward untuk layer sisanya
  #     for j in range(1, len(self.layers)):
  #       self.layers[j].output.append(np.array(self.layers[j].forward(self.layers[j-1].output[i])))
  #   #Convert into numpy array
  #   for i in range(len(self.layers)):
  #     self.layers[i].output = np.array(self.layers[i].output)

  #   return self.layers[-1].output

  # def forward_batch_class(self, input):
  #   '''
  #     Fungsi yang melakukan forward batch data

  #     Parameters:
  #       input: data batch untuk dimasukkan ke arsitektur neural network untuk menghasilkan output 

  #     Returns:
  #       output dari layer terakhir yang adalah list dari list angka dengan jumlah angka sejumlah neuron pada layer terakhir sejumlah data pada batch data
  #       dengan jika fungsi aktivasinya sigmoid maka akan dibulatkan, jika fungsi aktivasinya softmax maka akan mencari
  #       index dimana nilai tertinggi ditemukan
  #   '''
  #   # Format input = list of input (list sebanyak batch)
  #   # Panggil forward untuk setiap batch
  #   # Init output setiap layer dengan list kosong
    
  #   print("Menghitung prediksi...")

  #   for i in range(len(self.layers)):
  #     self.layers[i].output = []

  #   # Fill output list dengan output dari setiap data dari batch
  #   for i in range(input.shape[0]):
  #     # Panggil forward untuk layer pertama (layer input)
  #     self.layers[0].output.append(input[i])

  #     # Panggil forward untuk layer sisanya
  #     for j in range(1, len(self.layers)):
  #       self.layers[j].output.append(np.array(self.layers[j].forward(self.layers[j-1].output[i])))
  #   #Convert into numpy array
  #   for i in range(len(self.layers)):
  #     self.layers[i].output = np.array(self.layers[i].output)

  #   # if(self.layers[-1].activation == ActivationFunction.get_by_name('sigmoid')):
  #   #   return round(self.layers[-1].output)
  #   prediction = []

  #   for i in range(self.layers[-1].output.shape[0]):
  #     if(self.layers[-1].activation == ActivationFunction.get_by_name('sigmoid')):
  #       prediction.append(round(self.layers[-1].output[i][0]))
  #     if(self.layers[-1].activation == ActivationFunction.get_by_name('softmax')):
  #       prediction.append(np.argmax(self.layers[-1].output[i]))
    
  #   prediction = np.array(prediction)
  #   return prediction

  # def get_batch_dense(self, input, batch_size):
  #   #Untuk mendapatkan data hasil pembagian dari batch

  #   if len(input) % batch_size != 0:
  #     raise Exception('Error batch size')
    
  #   res = []
  #   idx = 0
  #   for i in range(idx, len(input), batch_size):
  #     res.append(input[i:i+batch_size])
  #     idx += batch_size
  #   return res

  # def backpropagation(self, input, target): #ini untuk masing-masing batch
  #   '''
  #     Fungsi yang melakukan backpropagation yang meliputi tahap forward, menghitung error term dan gradient bobot, dan update bobot

  #     Parameters:
  #       input: data batch untuk melakukan training
  #       target: nilai target untuk masing-masing data pada batch

  #     Result:
  #       Bobot masing-masing weight dari arsitektur terupdate sesuai dengan gradient dari masing-masing bobot
  #   '''
  #   #Panggil forward
  #   self.forward_batch(input) #diposisi ini harusnya semua layer udah punya outputnya -> outputnya ini berarti list of output (karena batch)
  #   # self.layers[-1].output = np.array([[0.5],[0.75]])
  #   #Panggil backprop layer terakhir
  #   self.layers[-1].backprop_output(target, self.layers[-2]) #menghasilkan derivatives untuk layer output (-1 -> layer output, -2 -> layer sebelum output)
  #   for i in range(len(self.layers)-2,0,-1):
  #     self.layers[i].backprop_hidden(self.layers[i-1], self.layers[i+1]) 

  #   #Setelah kode-kode di atas dijalanin -> u/setiap layer bakalan ada atribut derivatives
  #   #Derivatives ini misalkan cara aksesnya seq.layers[1].derivatives -> ini shapenya udah sama kayak weightnya 
  #   #Update weightnya bisa langsung seq.layers[1].weight = seq.layers[1].weight - 0.2*seq.layers[1].derivatives, untuk layerType flatten dan pooling nggak usah panggil update bobot
  #   #Tambahin fungsi update bobot
  #   for each_layer in self.layers:
  #     if(type(each_layer) is Dense):
  #       each_layer.weight = each_layer.weight - self.learning_rate * each_layer.derivatives
      
  #     elif(type(each_layer) is Conv):
  #       for k in range(each_layer.kernel.shape[0]):
  #         for z in range(each_layer.kernel.shape[1]):
  #           for row in range(each_layer.kernel.shape[2]):
  #             for col in range(each_layer.kernel.shape[3]):
  #               each_layer.kernel[k][z][row][col] = each_layer.kernel[k][z][row][col] - self.learning_rate * each_layer.derivatives[k][z][row][col]

  # def compile(self, learning_rate, loss, metrics):
  #   self.learning_rate = learning_rate
  #   pass

  # def fit(self, input, target, batchsize, epoch=1):
  #   for i in range(epoch):
  #     print("EPOCH", i)
  #     input_batches = self.get_batch_dense(input, batchsize)
  #     target_batches = self.get_batch_dense(target, batchsize)
  #     for i in range(len(input_batches)):
  #       print("Batch ke-", i)
  #       self.backpropagation(input_batches[i], target_batches[i])
  #       utils.calculateConfusionMatrix(self.layers[-1].output, self.layers[-1].activation, target_batches[i])

  def summary(self):
    '''
      Fungsi yang mencetak summary dari aristektur neural network meliputi output shape dan jumlah parameter dari masing-masing layer

      Parameters:

      Return:
        
    '''
    print ("{:<30} {:<30} {:<30}".format('Layer (type)','Output Shape','Param #'))
    #iterate all layers
    total_params = 0
    for i in range(len(self.layers)):
      weightTemp = 0
      if (type(self.layers[i]) is Dense):
        weightTemp = self.layers[i].weight.shape[0] * self.layers[i].weight.shape[1]
      # elif (type(self.layers[i]) is Conv):
      #   #Get all kernel dimensions
      #   kernel_dimensions=1
      #   for j in range(len(self.layers[i].kernel_size)):
      #     kernel_dimensions *= self.layers[i].kernel_size[j]
      #   weightTemp = self.layers[i].filter * (self.layers[i-1].shape[1] * kernel_dimensions + 1) 
      elif (type(self.layers[i]) is LSTM):
        #TODO
        pass
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
  