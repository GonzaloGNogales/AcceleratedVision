import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import TensorBoard

# Variables Initialization
batch_size = 100
num_classes = 10
epochs = 10

# Scaling and Data Conversion
rows, cols = 28, 28
(xt, yt), (xtest, ytest) = mnist.load_data()
xt = xt.reshape(xt.shape[0], rows, cols, 1)
xtest = xtest.reshape(xtest.shape[0], rows, cols, 1)

xt = xt.astype('float32')
xtest = xtest.astype('float32')

xt = xt/255
xtest = xtest/255

yt = keras.utils.to_categorical(yt, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)

# Neural Network Initialization
Inputs = Input(shape=(rows, cols, 1))

# Features Extraction Stage
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(Inputs)
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Classification Stage
x = Flatten()(x)
x = Dense(68, activation='relu')(x)
# x = Dropout(0.25)(x) commented for avoiding Under-Fitting
x = Dense(20, activation='relu')(x)
# x = Dropout(0.25)(x) commented for avoiding Under-Fitting
x = Dense(num_classes, activation='softmax')(x)

# Model Initialization
model = Model(inputs=Inputs, outputs=x)

# Save Model
saveModel = keras.callbacks.ModelCheckpoint('automodel.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
saveWeights = keras.callbacks.ModelCheckpoint('autoweights.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)

# Model Compile
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])

# Model Training
model.fit(xt, yt, batch_size=batch_size, epochs=epochs, callbacks=[saveModel, saveWeights], verbose=1, validation_data=(xtest, ytest))

# Results
score = model.evaluate(xtest, ytest, verbose=1)
print('The resulting score is', score)

# For loading saved models or weights
# Model
from keras.models import load_model
model2 = load_model('automodel.h5')

# Weights
model.load_weights('autoweights.h5')

