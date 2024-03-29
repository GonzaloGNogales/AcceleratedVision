import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, ConvLSTM2D, Conv3D, \
    BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
import matplotlib.pylab as plt


# Generaremos peliculas sinteticas con cuadrados moviendose dentro de las mismas, que tendran tamaños de 1x1 y 2x2 pixeles
# Estos cuadrados se moveran linealmente en el tiempo, la ventana final de pelicula obtenida sera de 40x40

def genera_peliculas(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    peliculas_ruidosas = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    peliculas_desplazadas = np.zeros((n_samples, n_frames, row, col, 1),
                                     dtype=np.float)

    for i in range(n_samples):
        # Creamos entre 3 y 7 cuadrados en movimiento
        n = np.random.randint(3, 8)

        for j in range(n):
            # posicion inicial
            xcomienzo = np.random.randint(20, 60)
            ycomienzo = np.random.randint(20, 60)
            # Direccion del movimiento
            direccionx = np.random.randint(0, 3) - 1
            direcciony = np.random.randint(0, 3) - 1

            # Tamaño de los cuadrados
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_desplazamiento = xcomienzo + direccionx * t
                y_desplazamiento = ycomienzo + direcciony * t
                peliculas_ruidosas[i, t, x_desplazamiento - w: x_desplazamiento + w,
                y_desplazamiento - w: y_desplazamiento + w, 0] += 1

                # añadimos ruido para que la red tenga que ser mas robusta ante valores
                # que no valgan exactamente la unidad
                if np.random.randint(0, 2):
                    ruido_f = (-1) ** np.random.randint(0, 2)
                    peliculas_ruidosas[i, t,
                    x_desplazamiento - w - 1: x_desplazamiento + w + 1,
                    y_desplazamiento - w - 1: y_desplazamiento + w + 1,
                    0] += ruido_f * 0.1

                # desplazamos una unidad los datos reales
                x_deplazamiento = x_desplazamiento + direccionx * (t + 1)
                y_desplazamiento = ycomienzo + direcciony * (t + 1)
                peliculas_desplazadas[i, t, x_desplazamiento - w: x_desplazamiento + w,
                y_desplazamiento - w: y_desplazamiento + w, 0] += 1

    # Obtenemos ventanas de 40x40 pixeles
    peliculas_ruidosas = peliculas_ruidosas[::, ::, 20:60, 20:60, ::]
    peliculas_desplazadas = peliculas_desplazadas[::, ::, 20:60, 20:60, ::]
    peliculas_ruidosas[peliculas_ruidosas >= 1] = 1
    peliculas_desplazadas[peliculas_desplazadas >= 1] = 1
    return peliculas_ruidosas, peliculas_desplazadas


# Creamos un modelo que tome como entrada las peliculas sinteticas creadas con tamaño (numero de frames,ancho,alto)
# Como salida obtenemos una pelicula del mismo tamaño

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')

# Entrenamiento de la red propuesta
peliculas_ruidosas, peliculas_desplazadas = genera_peliculas(n_samples=1200)
history = seq.fit(peliculas_ruidosas[:1000], peliculas_desplazadas[:1000], batch_size=10,
                  epochs=100, validation_split=0.05)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()
# Testeamos el sistema sobre una pelicula en la cual entregamos 7 posiciones tras las cuales deberá predecir nuevas posiciones

cual = 1004
track = peliculas_ruidosas[cual][:7, ::, ::, ::]

for j in range(16):
    nueva_posicion = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    nueva = nueva_posicion[::, -1, ::, ::, ::]
    track = np.concatenate((track, nueva), axis=0)

# Por último comparamos las predicciones con los datos reales
track2 = peliculas_ruidosas[cual][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predicciones!', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Trayectoria inicial', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Datos Reales', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = peliculas_desplazadas[cual][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    # plt.savefig('%i_animate.png' % (i + 1))

