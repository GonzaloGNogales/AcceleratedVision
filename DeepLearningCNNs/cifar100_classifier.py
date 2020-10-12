import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from keras.applications import VGG16
import matplotlib.pylab as plt

# Variables Initialization
batch_size = 100
num_classes = 100
epochs = 5

# Load Data Base
(xt, yt), (xtest, ytest) = cifar100.load_data()
_, rows, cols, channels = xt.shape

# Data Type Conversion to Float32
xt = xt.astype('float32')
xtest = xtest.astype('float32')

# Normalize Between 0 and 1
xt = xt/255
xtest = xtest/255

# Codify Outputs to Categorical Values (One Hot Codification)
yt = keras.utils.to_categorical(yt, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)

# Neural Network Initialization
Inputs = Input(shape=(rows, cols, channels))

# Features Extraction Stage
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(Inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Classification Stage
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Model Initialization
model = Model(inputs=Inputs, outputs=x)
model.summary()

# Save Model
# saveModel = keras.callbacks.ModelCheckpoint('automodel.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# saveWeights = keras.callbacks.ModelCheckpoint('autoweights.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)

# Custom Adam Optimizer
Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)  # SGD(lr=0.001, decay=le-6, momentum=0.9, nesterov=True)

# Model Compile
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam, metrics=['categorical_accuracy'])

# Model Training
history = model.fit(xt, yt, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))

# Results
score = model.evaluate(xtest, ytest, verbose=1)
print('The resulting score is', score)

# Result Visualization
# Model Precision
plt.figure(1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epochs')
plt.legend(['Training', 'Test'], loc='upper left')

# Model Loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Test'], loc='upper left')
plt.show()

