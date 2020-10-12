import sys
import json
import codecs
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Input, Bidirectional, GlobalMaxPool1D, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import matplotlib.pylab as plt

# Input Variables
batch_size = 32
epochs = 30
embedding_size = 128
max_characteristics = 20000
max_len = 80

# Data Base Loading
(xTraining, yTraining), (xTest, yTest) = imdb.load_data(num_words=max_characteristics)
xTraining = sequence.pad_sequences(xTraining, maxlen=max_len)
xTest = sequence.pad_sequences(xTest,maxlen=max_len)

# Model Creation
input = Input(shape=(max_len, ))
x = Embedding(max_characteristics, embedding_size)(input)
x = LSTM(embedding_size, return_sequences=True, activation='relu')(x)
x = Flatten()(x)
x = Dense(1, activation="sigmoid", kernel_initializer='zeros', bias_initializer='zeros')(x)

model = Model(inputs=input, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()

# Training
checkpoint = ModelCheckpoint('deteccion_texto.h5', monitor='val_binary_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
history = model.fit(xTraining, yTraining, batch_size=batch_size, epochs=5, callbacks=[checkpoint], validation_data=(xTest,yTest), shuffle=True, verbose=1)

# Results Visualization
plt.figure(1)
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochcs')
plt.legend(['Test','Entrenamiento'], loc='upper left')

plt.figure(2)
plt.plot(history.history['val_binary_accuracy'])
plt.plot(history.history['binary_accuracy'])
plt.title('Precision del Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Test','Entrenamiento'], loc='upper left')

output = model.predict(xTraining[round(len(xTraining[:, 0]) * 0.9):round(len(xTraining[:, 0])), :])

output[output < 0.5] = 0
output[output >= 0.5] = 1

difference = abs(yTraining[round(len(xTraining[:, 0]) * 0.9):round(len(xTraining[:, 0]))] - np.uint8(output[:, 0]))
punctuation = difference[difference > 0]
score = (1 - (len(punctuation) / len(difference)))

print('Best score is: %f' % score)
plt.show()
