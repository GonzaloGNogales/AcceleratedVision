import tensorflow as tf
import numpy as np
import tensorflow.keras
import math
from pandas import read_csv
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Conv3D, BatchNormalization, ConvLSTM2D, LSTM, GRU
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Vector-Matrix Conversion
def create_data_base(data, look_back=1):
	dataX, dataY = [], []

	for i in range(len(data)-look_back-1):
		a = data[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(data[i + look_back, 0])

	return np.array(dataX), np.array(dataY)


# Load Data
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
data = dataframe.values
data = data.astype('float32')

# Data Normalization
scaling = MinMaxScaler(feature_range=(0, 1))
data = scaling.fit_transform(data)

# Train/Test Division
tsize = int(len(data) * 0.67)
testsize = len(data) - tsize
training, test = data[0:tsize, :], data[tsize:len(data), :]

# Train/Test size change
look_back = 20
tX, tY = create_data_base(training, look_back)
testX, testY = create_data_base(test, look_back)
tX = np.reshape(tX, (tX.shape[0], 1, tX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM Neural Network Design
input = Input(shape=(1, look_back))
x = LSTM(10)(input)
# x = GRU(10)(input)
x = Dense(1)(x)
model = Model(inputs=input, outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(tX, tY, epochs=650, batch_size=1, verbose=2)

plt.figure(1)
plt.plot(history.history['loss'])
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Training'], loc='upper left')


# Predictions
training_predictions = model.predict(tX)
test_predictions = model.predict(testX)

# Prediction inversion to calculate error
training_predictions = scaling.inverse_transform(training_predictions)
tY = scaling.inverse_transform([tY])
test_predictions = scaling.inverse_transform(test_predictions)
testY = scaling.inverse_transform([testY])

# RMSE Calculation
training_punctuation = math.sqrt(mean_squared_error(tY[0,:], training_predictions[:, 0]))
test_punctuation = math.sqrt(mean_squared_error(testY[0], test_predictions[:,0]))
print('Training Punctuation: %.2f RMSE & Test Punctuation: %.2f RMSE' % (training_punctuation, test_punctuation))


# Training Prediction Offset
plot_training_prediction = np.empty_like(data)
plot_training_prediction[:, :] = np.nan
plot_training_prediction[look_back:len(training_predictions) + look_back, :] = training_predictions

# Test Prediction Offset
plot_test_prediction = np.empty_like(data)
plot_test_prediction[:, :] = np.nan
plot_test_prediction[len(training_predictions) + (look_back * 2) + 1:len(data) - 1, :] = test_predictions

# Show Predictions And Data
plt.figure(2)
plt.plot(scaling.inverse_transform(data))
plt.plot(plot_training_prediction)
plt.plot(plot_test_prediction)
plt.show()



