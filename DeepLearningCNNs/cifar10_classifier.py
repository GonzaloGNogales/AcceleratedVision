import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from keras.applications import VGG16

# Variables Initialization
batch_size = 100
num_classes = 10
epochs = 30

# Scaling and Data Conversion
(xt, yt), (xtest, ytest) = cifar10.load_data()
_, rows, cols, channels = xt.shape

xt = xt.astype('float32')
xtest = xtest.astype('float32')

xt = xt/255
xtest = xtest/255

yt = keras.utils.to_categorical(yt, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)

# Too Slow Neural Network Architecture for Cifar10
# Inputs = Input(shape=(rows, cols, channels))

# Features Extraction Stage
# x = Conv2D(64, kernel_size=(3, 3), activation='relu')(Inputs)
# x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# Classification Stage
# x = Flatten()(x)
# x = Dense(10, activation='relu')(x)
# x = Dense(num_classes, activation='softmax')(x)

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
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Classification Stage
x = Flatten()(x)
x = Dense(10, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

# Model Initialization
model = Model(inputs=Inputs, outputs=x)

# Custom optimizer
stochastic_gradient_descent = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)  # SGD(lr=0.001, decay=le-6, momentum=0.9, nesterov=True)

# Model Compile
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=stochastic_gradient_descent, metrics=['categorical_accuracy'])

# Model Training
model.fit(xt, yt, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))

# Results
score = model.evaluate(xtest, ytest, verbose=1)
print('The resulting score is', score)
