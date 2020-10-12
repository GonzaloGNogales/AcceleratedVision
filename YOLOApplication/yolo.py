from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import numpy as np
import os
import cv2
from utilidades import decodificacion_salida
from tensorflow.keras.layers import concatenate

ruta_yolo_backend = 'full_yolo_backend.h5'  # nombre del extractor de características a utilizar


class ExtractorCaracteristicasBase(object):  # Clase Base para los posibles extractores de características

    def __init__(self, tamano_entrada):
        raise NotImplementedError("error message")

    def normalizacion(self, imagen):
        raise NotImplementedError("error message")

    def obtener_tamano_salida(self):
        return self.extractor_caracteristicas.get_output_shape_at(-1)[1:3]

    def extraer(self, imagen_entrada):
        return self.extractor_caracteristicas(imagen_entrada)


class FullYoloFeature(ExtractorCaracteristicasBase):  # Extractor de características original de Yolo

    def __init__(self, tamano_entrada):  # Inicializacion de la red
        imagen_entrada = Input(shape=(tamano_entrada, tamano_entrada, 3))

        def espacio_a_profundidad_x2(x):
            return tf.nn.space_to_depth(x, block_size=2)  # Capa especial de Yolo de espacio a profundidad

        # Capa 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(imagen_entrada)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Capa 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Capa 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Capa 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Capa 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        salto_conexion = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Capa 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Capa 21
        salto_conexion = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
            salto_conexion)
        salto_conexion = BatchNormalization(name='norm_21')(salto_conexion)
        salto_conexion = LeakyReLU(alpha=0.1)(salto_conexion)
        salto_conexion = Lambda(espacio_a_profundidad_x2)(salto_conexion)

        x = concatenate([salto_conexion, x])

        # Capa 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.extractor_caracteristicas = Model(imagen_entrada, x)
        self.extractor_caracteristicas.load_weights(ruta_yolo_backend)

    def normalizacion(self, imagen):  # Normalizacion aplicada a las imagenes de entrada
        return imagen / 255.


class YOLO(object):  # Inicializacion de la clase Yolo, que contiene el procesado previo y posterior de Yolo
    def __init__(self, backend, tamano_entrada, etiquetas, max_cajas_por_imagen,
                 tamanos_base):  # Funcion de inicializacion de Yolo

        # Inicializacion de variables
        self.tamano_entrada = tamano_entrada
        self.etiquetas = list(etiquetas)
        self.numero_clases = len(self.etiquetas)
        self.numero_cajas = len(tamanos_base) // 2
        self.class_wt = np.ones(self.numero_clases, dtype='float32')
        self.tamanos_base = tamanos_base
        self.max_cajas_por_imagen = max_cajas_por_imagen

        # Creacion del Modelo

        # 1-Creacion de extractor de caracteristicas
        imagen_entrada = Input(shape=(self.tamano_entrada, self.tamano_entrada, 3))
        self.cajas_buenas = Input(shape=(1, 1, 1, max_cajas_por_imagen, 4))

        self.extractor_caracteristicas = FullYoloFeature(self.tamano_entrada)

        print(self.extractor_caracteristicas.obtener_tamano_salida())
        self.rejilla_alto, self.rejilla_ancho = self.extractor_caracteristicas.obtener_tamano_salida()
        caracteristicas = self.extractor_caracteristicas.extraer(imagen_entrada)

        # 2-Creacion de capas de salida adaptadas al numero de cajas,clases
        salida = Conv2D(self.numero_cajas * (4 + 1 + self.numero_clases), (1, 1), strides=(1, 1), padding='same',
                        name='DetectionLayer', kernel_initializer='lecun_normal')(caracteristicas)
        salida = Reshape((self.rejilla_alto, self.rejilla_ancho, self.numero_cajas, 4 + 1 + self.numero_clases))(salida)
        salida = Lambda(lambda args: args[0])([salida, self.cajas_buenas])

        self.model = Model([imagen_entrada, self.cajas_buenas], salida)

        # Inicializacion de pesos de las capas de salida
        capa = self.model.layers[-4]
        pesos = capa.get_weights()

        nuevo_kernel = np.random.normal(size=pesos[0].shape) / (self.rejilla_alto * self.rejilla_ancho)
        nuevo_bias = np.random.normal(size=pesos[1].shape) / (self.rejilla_alto * self.rejilla_ancho)

        capa.set_weights([nuevo_kernel, nuevo_bias])

        self.model.summary()

    def cargar_pesos(self, weight_path):  # Funcion de cargado de pesos preentrenados
        self.model.load_weights(weight_path)

    def predecir(self, imagen, umbral_objeto, umbral_supresionomaxima):  # Funcion de prediccion
        alto_imagen, ancho_imagen, _ = imagen.shape
        imagen = cv2.resize(imagen, (self.tamano_entrada,
                                     self.tamano_entrada))  # Cambio de tamaño de la imagen de entrada al tamaño de deteccion de Yolo
        imagen = self.extractor_caracteristicas.normalizacion(imagen)  # Normalizacion de la imagen de entrada

        imagen_entrada = imagen[:, :, ::-1]
        imagen_entrada = np.expand_dims(imagen_entrada, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.max_cajas_por_imagen, 4))

        salidared = self.model.predict([imagen_entrada, dummy_array])[
            0]  # prediccion del modelo extractor de caracteristicas
        cajas = decodificacion_salida(salidared, self.tamanos_base, self.numero_clases, umbral_objeto,
                                      umbral_supresionomaxima)  # Decodificacion de la salida Yolo en formato caja

        return cajas, salidared