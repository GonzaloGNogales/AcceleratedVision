import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import vgg16, resnet50, inception_resnet_v2, inception_v3, xception
from keras_preprocessing import image
import matplotlib.pylab as plt

batch_size = 100
num_classes = 10
epochs = 50

TRANSFER_LEARNING = 0
FINE_TUNING = 1
if TRANSFER_LEARNING:
    image_input = Input(shape=(224, 224, 3))
    model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers[:]:
        layer.trainable = False
    x = model.output
    x = Flatten()(x)
    x = Dense(1000)(x)
    x = Dense(100)(x)
    model = Model(inputs=model.input, outputs=x)
    model.summary()

if FINE_TUNING:
    image_input = Input(shape=(224, 224, 3))
    model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = model.output
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(100, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)
    model.summary()
