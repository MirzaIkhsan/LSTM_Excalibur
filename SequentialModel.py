from Dense import Dense
from Input import Input
from Flatten import Flatten
from Conv import Conv
from Pooling import Pooling
import random
import numpy as np
import math
import utils
from utils import paddMatrix
import ActivationFunction
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
    '''Push a new layer into stack'''
    ## Add input layer
    if (type(layer) is Input):
      #### (consider if input is not the first layer)
      if (len(self.layers) > 0):
        print("Input layer hanya bisa ditambah di awal.")
        return
    elif (len(self.layers) == 0):
      print("Layer pertama hanya bisa berupa input layer.")
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
      if (type(layer) is Flatten):
        temp = 1
        for shapeEl in self.layers[-1].shape[1:]:
          temp *= shapeEl
        layer.shape = (None, temp)
      
      ## Add convo layer
      if (type(layer) is Conv):
        wRow = self.layers[-1].shape[2]
        wCol = self.layers[-1].shape[3]
        fRow = layer.kernel_size[1]
        fCol = layer.kernel_size[2]
        nFilter = layer.filter
        p = layer.padding
        sRow = layer.strides[1]
        sCol = layer.strides[2]
        VRow = math.floor((wRow-fRow+2*p)/sRow) + 1
        VCol = math.floor((wCol-fCol+2*p)/sCol) + 1
        layer.shape = (None, nFilter, VRow, VCol)
      
      ## Add pooling layer
      if (type(layer) is Pooling):
        wRow = self.layers[-1].shape[2]
        wCol = self.layers[-1].shape[3]
        fRow = layer.filter_size[0]
        fCol = layer.filter_size[1]
        nFilter = self.layers[-1].shape[1]
        sRow = layer.strides[0]
        sCol = layer.strides[1]
        VRow = math.floor((wRow-fRow)/sRow) + 1
        VCol = math.floor((wCol-fCol)/sCol) + 1
        layer.shape = (None, nFilter, VRow, VCol)

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
    # Panggil forward untuk layer pertama (layer input)
    self.layers[0].output = input
    # Panggil forward untuk layer sisanya
    for i in range(1, len(self.layers)):
      self.layers[i].output = self.layers[i].forward(self.layers[i-1].output)
    # Return output layer terakhir
    return self.layers[-1].output


  def forward_batch(self, input):
    '''
      Fungsi yang melakukan forward batch data

      Parameters:
        input: data batch untuk dimasukkan ke arsitektur neural network untuk menghasilkan output 

      Returns:
        output dari layer terakhir yang adalah list dari list angka dengan jumlah angka sejumlah neuron pada layer terakhir sejumlah data pada batch data
    '''
    # Format input = list of input (list sebanyak batch)
    # Panggil forward untuk setiap batch
    # Init output setiap layer dengan list kosong
    
    print("Menghitung prediksi...")

    for i in range(len(self.layers)):
      self.layers[i].output = []

    # Fill output list dengan output dari setiap data dari batch
    for i in range(input.shape[0]):
      # Panggil forward untuk layer pertama (layer input)
      self.layers[0].output.append(input[i])

      # Panggil forward untuk layer sisanya
      for j in range(1, len(self.layers)):
        self.layers[j].output.append(np.array(self.layers[j].forward(self.layers[j-1].output[i])))
    #Convert into numpy array
    for i in range(len(self.layers)):
      self.layers[i].output = np.array(self.layers[i].output)

    return self.layers[-1].output

  def forward_batch_class(self, input):
    '''
      Fungsi yang melakukan forward batch data

      Parameters:
        input: data batch untuk dimasukkan ke arsitektur neural network untuk menghasilkan output 

      Returns:
        output dari layer terakhir yang adalah list dari list angka dengan jumlah angka sejumlah neuron pada layer terakhir sejumlah data pada batch data
        dengan jika fungsi aktivasinya sigmoid maka akan dibulatkan, jika fungsi aktivasinya softmax maka akan mencari
        index dimana nilai tertinggi ditemukan
    '''
    # Format input = list of input (list sebanyak batch)
    # Panggil forward untuk setiap batch
    # Init output setiap layer dengan list kosong
    
    print("Menghitung prediksi...")

    for i in range(len(self.layers)):
      self.layers[i].output = []

    # Fill output list dengan output dari setiap data dari batch
    for i in range(input.shape[0]):
      # Panggil forward untuk layer pertama (layer input)
      self.layers[0].output.append(input[i])

      # Panggil forward untuk layer sisanya
      for j in range(1, len(self.layers)):
        self.layers[j].output.append(np.array(self.layers[j].forward(self.layers[j-1].output[i])))
    #Convert into numpy array
    for i in range(len(self.layers)):
      self.layers[i].output = np.array(self.layers[i].output)

    # if(self.layers[-1].activation == ActivationFunction.get_by_name('sigmoid')):
    #   return round(self.layers[-1].output)
    prediction = []

    for i in range(self.layers[-1].output.shape[0]):
      if(self.layers[-1].activation == ActivationFunction.get_by_name('sigmoid')):
        prediction.append(round(self.layers[-1].output[i][0]))
      if(self.layers[-1].activation == ActivationFunction.get_by_name('softmax')):
        prediction.append(np.argmax(self.layers[-1].output[i]))
    
    prediction = np.array(prediction)
    return prediction

  def get_batch_dense(self, input, batch_size):
    #Untuk mendapatkan data hasil pembagian dari batch

    if len(input) % batch_size != 0:
      raise Exception('Error batch size')
    
    res = []
    idx = 0
    for i in range(idx, len(input), batch_size):
      res.append(input[i:i+batch_size])
      idx += batch_size
    return res

  def backpropagation(self, input, target): #ini untuk masing-masing batch
    '''
      Fungsi yang melakukan backpropagation yang meliputi tahap forward, menghitung error term dan gradient bobot, dan update bobot

      Parameters:
        input: data batch untuk melakukan training
        target: nilai target untuk masing-masing data pada batch

      Result:
        Bobot masing-masing weight dari arsitektur terupdate sesuai dengan gradient dari masing-masing bobot
    '''
    #Panggil forward
    self.forward_batch(input) #diposisi ini harusnya semua layer udah punya outputnya -> outputnya ini berarti list of output (karena batch)
    # self.layers[-1].output = np.array([[0.5],[0.75]])
    #Panggil backprop layer terakhir
    self.layers[-1].backprop_output(target, self.layers[-2]) #menghasilkan derivatives untuk layer output (-1 -> layer output, -2 -> layer sebelum output)
    for i in range(len(self.layers)-2,0,-1):
      self.layers[i].backprop_hidden(self.layers[i-1], self.layers[i+1]) 

    #Setelah kode-kode di atas dijalanin -> u/setiap layer bakalan ada atribut derivatives
    #Derivatives ini misalkan cara aksesnya seq.layers[1].derivatives -> ini shapenya udah sama kayak weightnya 
    #Update weightnya bisa langsung seq.layers[1].weight = seq.layers[1].weight - 0.2*seq.layers[1].derivatives, untuk layerType flatten dan pooling nggak usah panggil update bobot
    #Tambahin fungsi update bobot
    for each_layer in self.layers:
      if(type(each_layer) is Dense):
        each_layer.weight = each_layer.weight - self.learning_rate * each_layer.derivatives
      
      elif(type(each_layer) is Conv):
        for k in range(each_layer.kernel.shape[0]):
          for z in range(each_layer.kernel.shape[1]):
            for row in range(each_layer.kernel.shape[2]):
              for col in range(each_layer.kernel.shape[3]):
                each_layer.kernel[k][z][row][col] = each_layer.kernel[k][z][row][col] - self.learning_rate * each_layer.derivatives[k][z][row][col]

  def compile(self, learning_rate, loss, metrics):
    self.learning_rate = learning_rate
    pass

  def fit(self, input, target, batchsize, epoch=1):
    for i in range(epoch):
      print("EPOCH", i)
      input_batches = self.get_batch_dense(input, batchsize)
      target_batches = self.get_batch_dense(target, batchsize)
      for i in range(len(input_batches)):
        print("Batch ke-", i)
        self.backpropagation(input_batches[i], target_batches[i])
        utils.calculateConfusionMatrix(self.layers[-1].output, self.layers[-1].activation, target_batches[i])

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
      elif (type(self.layers[i]) is Conv):
        #Get all kernel dimensions
        kernel_dimensions=1
        for j in range(len(self.layers[i].kernel_size)):
          kernel_dimensions *= self.layers[i].kernel_size[j]
        weightTemp = self.layers[i].filter * (self.layers[i-1].shape[1] * kernel_dimensions + 1) 
      total_params += weightTemp  
      print("{:<30} {:<30} {:<30}".format(str(type(self.layers[i])), str(self.layers[i].shape), str(weightTemp)))
    print("Total params: ", total_params)

  def save_model(self, filename=None):
    '''
    Menyimpan model yang sudah dibuat ke dalam sebuah file txt.
    Jika filename None, akan dibuka prompt yang meminta nama file.
    '''
    #filename = input('Masukan nama file: ')
    if filename is None:
      filename = input('Masukan nama model: ')
    f = open('../saves/' + filename + '.txt', 'w')
    count_layers = len(self.layers)

    f.write(str(count_layers))
    f.write("\n")

    for layer in self.layers:
      f.write(layer.layerType)
      f.write(";")
      if layer.layerType == "Input":
        f.write("({ch},{w},{h})\n".format(
          ch = layer.shape[1],
          w = layer.shape[2],
          h = layer.shape[3]))

      elif layer.layerType == "Conv":
        f.write("{filter};{kernel_size};{strides};{p};{act}\n".format(
          filter = layer.filter,
          kernel_size = layer.kernel_size,
          strides = layer.strides,
          p = layer.padding,
          act = layer.get_activation_name()
        ))

        # Save weights (list of matriks?)
        for i in range(layer.filter):
          f.write("f:{i}\n".format(i=i))
          for depth in range(layer.kernel_size[0]):
            f.write("depth:{depth}\n".format(depth=depth))
            for row in range(layer.kernel_size[1]):
              for col in range(layer.kernel_size[2]):
                f.write("{el};".format(el=layer.kernel[i][depth][row][col]))
              f.write("\n")


      elif layer.layerType == "Pooling":
        f.write("{filter_size};{strides};{mode}\n".format(
          filter_size = layer.filter_size,
          strides = layer.strides,
          mode = layer.mode
        ))
        
      elif layer.layerType == "Flatten":
        # Do nothing
        f.write("\n")

      elif layer.layerType == "Dense":
        f.write("{n_neuron};{act};{bias}\n".format(
          n_neuron = layer.n_neuron,
          act = layer.get_activation_name(),
          bias = layer.bias
        ))

        # Save weights (matriks? Array?)
        f.write("{weightrow}\n".format(weightrow=layer.weight.shape[0]))

        for row in layer.weight:
          for el in row:
            f.write("{el};".format(el=el))
          f.write("\n")

    f.close()
    
  def load_model(self, filename=None):
    '''
    Membaca sebuah file txt dan membuat model berdasarkan data yang tersimpan dalam file.
    Jika filename tidak disediakan, akan muncul prompt yang meminta nama file.
    '''
    self.clear_layers()
    if filename is None:
      filename = input('Masukan nama model: ')

    try:
      f = open('../saves/' + filename + '.txt', 'r')
    except FileNotFoundError:
      print("File model " + filename + "tidak ditemukan. Pastikan file model ada dalam direktori saves dengan ekstensi .txt.")
      return

    print("Load model dari file /saves/" + filename + ".txt...")
    num_layers = int(f.readline())

    for i in range(num_layers):
      line = f.readline().strip("\n").split(";")
      
      if (line[0] == "Input"):
        shape = re.sub('[()]', '', line[1]).split(",")
        self.add(Input(shape=(int(shape[0]),int(shape[1]),int(shape[2]))))

      elif (line[0] == "Conv"):
        filter = line[1]
        kernel_size = re.sub('[()]', '', line[2]).split(",")
        strides = re.sub('[()]', '', line[3]).split(",")
        p = line[4]
        act = line[5]
        
        #Load weights
        filters = []
        filter_idx = -1
        while (filter_idx < (int(filter)-1)):
          line = f.readline().strip("\n").split(":")
          filter_idx = int(line[1])
          filter_temp = []
          for d in range(int(kernel_size[0])):
            line = f.readline()
            depth_temp = []
            for row in range(int(kernel_size[1])):
              row_temp = []
              weights = f.readline().strip("\n").split(";")
              for col in range(int(kernel_size[2])):
                row_temp.append(float(weights[col]))
              depth_temp.append(row_temp)
            filter_temp.append(depth_temp)
          filters.append(filter_temp)

        self.add(Conv(
          filter=int(filter),
          kernel_size=(int(kernel_size[0]),int(kernel_size[1]),int(kernel_size[2])),
          strides=(int(strides[0]),int(strides[1]),int(strides[2])),
          padding=int(p),
          activation=act,
          kernel = filters))

      elif (line[0] == "Pooling"):
        filter_size = re.sub('[()]', '', line[1]).split(",")
        strides = re.sub('[()]', '', line[2]).split(",")
        mode = line[3]
        self.add(Pooling(
          filter_size=(int(filter_size[0]),int(filter_size[1])),
          strides=(int(strides[0]),int(strides[1])),
          mode=mode
        ))

      elif (line[0] == "Flatten"):
        self.add(Flatten())

      elif (line[0] == "Dense"):
        n_neuron = line[1]
        act = line[2]
        bias = line[3]

        #Load weights
        weight_rows = f.readline().strip("\n")
        weight_rows = int(weight_rows)

        weight = []
        for row in range(weight_rows):
          temp_row = []
          line = f.readline().strip("\n").split(";")
          for col in range(int(n_neuron)):
            temp_row.append(float(line[col]))
          weight.append(temp_row)

        self.add(Dense(
          int(n_neuron),
          activation=act,
          bias=int(bias),
          weight=weight
        ))

    print("Loading model " + filename + " selesai.")
    f.close()
