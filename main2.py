import pandas as pd
import numpy as np
from SequentialModel import Sequential
from Dense import Dense
from LSTM import LSTM

df = pd.read_csv('bitcoin_price_Training - Training.csv')  
print(df.head())

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
print(X.shape)


# Self-Implementation
selfmodel = Sequential()
selfmodel.add(LSTM(10, input_shape=(32,6)))
selfmodel.add(Dense(6, activation='linear'))
print("===============Hasil predict==================")
print(selfmodel.predict(X[0]))
print("===============Hasil predict==================")
selfmodel.summary()
