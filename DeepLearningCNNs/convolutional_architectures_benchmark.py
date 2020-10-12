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

models = ["vgg16", "InceptionV3", "Resnet50", "InceptionV4(Inception-ResnetV2)", "Xception"]
answers = ["container_ship", "motor_scooter", "leopard", "mushroom", "Madagascar_cat", "dalmatian", "marmot", "tiger"]

for nim in range(1, 10):
    # Load images from file system here
    im = "drive/My Drive/imagenet/imagen_%d.png" % nim
    # Images loaded...

    for Model in range(0, 5):
        img = image.load_img(im, target_size=(224, 224, 3))
        image = image.img_to_array(img)
        image = np.expand_dims(image, axis=0)

        if Model == 0:
            from tensorflow.keras.applications.vgg16 import preprocess_input

            image = preprocess_input(image)
            model = vgg16.VGG16(weights='imagenet', include_top=True)
            # model.summary()

        if Model == 1:
            from tensorflow.keras.applications.inception_v3 import preprocess_input

            image = preprocess_input(image)
            model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
            # model.summary()

        if Model == 2:
            from tensorflow.keras.applications.resnet50 import preprocess_input

            image = preprocess_input(image)
            model = resnet50.ResNet50(weights='imagenet', include_top=True)
            # model.summary()

        if Model == 3:
            from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

            image = preprocess_input(image)
            model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=True)
            # model.summary()

        if Model == 4:
            from tensorflow.keras.applications.xception import preprocess_input

            image = preprocess_input(image)
            model = xception.Xception(weights='imagenet', include_top=True)
            # model.summary()

        prediction = model.predict(image)

        from tensorflow.keras.applications.imagenet_utils import decode_predictions

        decoded_prediction = decode_predictions(prediction)
        # decoded_prediction = np.array(decoded_prediction)
        print("The prediction of the model %s is %s with a probability of %f" % (
            models[Model], decoded_prediction[0][0][1], decoded_prediction[0][0][2] * 100))

    plt.figure(1)
    plt.imshow(image.img_to_array(img) / 255)
    plt.show()
