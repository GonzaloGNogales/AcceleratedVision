import numpy as np
import tensorflow as tf
import cv2


class CajaDetectada:  # Clase Caja---Este es el formato de las salidas de Yolo
    def __init__(self, xmin, ymin, xmax, ymax, c=None, clases=None):  # Inicializacion de Caja
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.clases = clases

        self.etiqueta = -1
        self.puntuacion = -1

    def obtener_etiqueta(self):  # Obtencion de la etiqueta con mayor probabilidad
        if self.etiqueta == -1:
            self.etiqueta = np.argmax(self.clases)

        return self.etiqueta

    def obtener_puntuacion(self):  # Obtencion de la puntuación o probabilidad asignada a la etiqueta elegida
        if self.puntuacion == -1:
            self.puntuacion = self.clases[self.obtener_etiqueta()]

        return self.puntuacion


def iou_cajas(caja1, caja2):  # Funcion de calculo de la Intersección sobre la Union de las Cajas o IOU
    interseccion_ancho = intervalo_superposicion([caja1.xmin, caja1.xmax], [caja2.xmin, caja2.xmax])
    interseccion_alto = intervalo_superposicion([caja1.ymin, caja1.ymax], [caja2.ymin, caja2.ymax])

    interseccion = interseccion_ancho * interseccion_alto

    ancho1, alto1 = caja1.xmax - caja1.xmin, caja1.ymax - caja1.ymin
    ancho2, alto2 = caja2.xmax - caja2.xmin, caja2.ymax - caja2.ymin

    union = ancho1 * alto1 + ancho2 * alto2 - interseccion

    return float(interseccion) / union


def dibujar_deteccion(imagen, cajas, etiquetas):  # Funcion de dibujado de Cajas de salida de Yolo
    imagen_alto, imagen_ancho, _ = imagen.shape
    for caja in cajas:
        xmin = int(caja.xmin * imagen_ancho)  # Obtencion de la xminima normalizada con respectom al ancho de imagen
        ymin = int(caja.ymin * imagen_alto)  # Obtencion de la yminima normalizada con respectom al alto de imagen
        xmax = int(caja.xmax * imagen_ancho)  # Obtencion de la xmaxima normalizada con respectom al ancho de imagen
        ymax = int(caja.ymax * imagen_alto)  # Obtencion de la yminima normalizada con respectom al alto de imagen
        cv2.rectangle(imagen, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Dibujado de rectangulo de deteccion
        cv2.putText(imagen, etiquetas[caja.obtener_etiqueta()] + ' ' + str(caja.obtener_puntuacion()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * imagen_alto, (0, 255, 0),
                    2)  # Escritura de etiqueta y confianza de deteccion

    return imagen


def decodificacion_salida(salidared, tamano_base, numero_clases, umbral_objeto=0.6,
                          umbral_supresionomaxima=0.6):  # Decodificacion de salidas de Yolo a formato Caja
    rejilla_alto, rejilla_ancho, numero_cajas = salidared.shape[:3]  # Ancho y alto de rejilla con respecto a imagen

    cajas = []  # definicion de lista de cajas

    # decodificacion de la salida
    salidared[..., 4] = _sigmoidal(
        salidared[..., 4])  # Aplicacion de activacion sigmoidal al subconjunto de confianzas de la salida de yolo
    salidared[..., 5:] = salidared[..., 4][..., np.newaxis] * _softmax(salidared[..., 5:])
    salidared[..., 5:] *= salidared[...,
                          5:] > umbral_objeto  # Aplicacion de Softmax al conjunto de clases y umbralización
    # segun el umbral de objeto predefinido por nosotros

    # NOTA: Hasta el cuarto elemento se situan X, Y, ancho y alto de caja, a partir del cuarto elemento se situan la confianza de caja y las probabilidades por clase

    for fila in range(
            rejilla_alto):  # Bucle de recorrido de rejilla y cajas para obtener las cajas resultantes con el umbral de objetos aplicado
        for columna in range(rejilla_ancho):
            for b in range(numero_cajas):

                clases = salidared[fila, columna, b, 5:]  # Probabilidades por clase

                if np.sum(clases) > 0:  # Condicion de que haya clases

                    x, y, w, h = salidared[fila, columna, b, :4]  # Obtencion de X,Y,Ancho y Alto de cajas

                    x = (columna + _sigmoidal(
                        x)) / rejilla_ancho  # Obtencion de coordenada X de la esquina superior de la caja a partir de las salidas normalizadas
                    y = (fila + _sigmoidal(
                        y)) / rejilla_alto  # Obtencion de coordenada Y de la esquina superior de la caja a partir de las salidas normalizadas
                    w = tamano_base[2 * b + 0] * np.exp(
                        w) / rejilla_ancho  # Obtención del ancho desnormalizado de la caja propuesta a partir de los posibles anchos y la exponencial del ancho normalizado
                    h = tamano_base[2 * b + 1] * np.exp(
                        h) / rejilla_alto  # Obtención del alto desnormalizado de la caja propuesta a partir de los posibles altos y la exponencial del alto normalizado
                    confianza = salidared[
                        fila, columna, b, 4]  # Obtención de la confianza de las diferentes cajas propuestas

                    caja = CajaDetectada(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confianza,
                                         clases)  # Obtencion de la caja propuesta dentro de la clase caja

                    cajas.append(caja)

    # Bucle de recorrido de cajas para obtener las cajas resultantes con el umbral de supresion no maximo de cajas
    for c in range(numero_clases):
        indices_ordenados = list(reversed(np.argsort([caja.clases[c] for caja in cajas])))

        for i in range(len(indices_ordenados)):
            index_i = indices_ordenados[i]

            if cajas[index_i].clases[c] == 0:
                continue
            else:
                for j in range(i + 1, len(indices_ordenados)):
                    index_j = indices_ordenados[j]

                    if iou_cajas(cajas[index_i], cajas[
                        index_j]) >= umbral_supresionomaxima:  # Aplicacion de umbral de IOU para encontrar las mejores cajas candidatas posibles
                        cajas[index_j].clases[c] = 0

    cajas = [caja for caja in cajas if
             caja.obtener_puntuacion() > umbral_objeto]  # Eliminar aquellas cajas cuyo umbral no supere el umbral del objeto

    return cajas


def intervalo_superposicion(intervalo_a, intervalo_b):  # Funcion del calculo del intervalo de superposicion de cajas
    x1, x2 = intervalo_a
    x3, x4 = intervalo_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoidal(x):  # Funcion de aplicacion de activacion sigmoidal
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):  # Funcion de aplicacion de activacion softmax
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
