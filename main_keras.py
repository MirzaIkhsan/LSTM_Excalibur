# Main ini masih bikin modelnya pake library keras

import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv('bitcoin_price_Training - Training.csv')  
# print(df.head())

#Hapus karakter "," pada string angka pemisah ribuan
df['Volume'] = df['Volume'].str.replace(",","")
# print(df['Volume'].head())


df['Market Cap'] = df['Market Cap'].str.replace(",","")
# print(df['Market Cap'].head())



data = df[['Open','High','Low','Close','Volume','Market Cap']]
X = np.flip(data.values[0:32], axis=0)
# X = data.values[0:32]
X = np.asarray(X).astype('float32')
X = X.reshape(1, X.shape[0], X.shape[1])
# print(X)


model = Sequential() #initialize sequential model
model.add(LSTM(10, input_shape=(32,6))) #LSTM layer with 10 neurons
model.add(Dense(6, activation='linear')) #Dense output layer with 1 neuron, linear activation

print(model.predict(X[:1]))
