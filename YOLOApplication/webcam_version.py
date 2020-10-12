import cv2
import numpy as np
from utilidades import dibujar_deteccion
from yolo import YOLO
import json
import time

# Address load
configuration_address = 'config.json'
pretrained_weights_address = 'yolo_anchors_5.h5'

with open(configuration_address) as configuration_buffer:  # Config file loading
    configuration = json.load(configuration_buffer)

# YOLO Model initialization from config file
yolo = YOLO(backend=configuration['model']['backend'],
            tamano_entrada=configuration['model']['input_size'],
            etiquetas=configuration['model']['tags'],
            max_cajas_por_imagen=configuration['model']['max_boxes_per_image'],
            tamanos_base=configuration['model']['base_sizes'])

# Load pretrained weights into YOLO
yolo.cargar_pesos(pretrained_weights_address)

# Video capture initialization
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    frame = frame[:, int(frame.shape[1] / 2) - int(frame.shape[0] / 2):int(frame.shape[1] / 2) + int(frame.shape[0] / 2), :]
    test = frame.copy()

    #   start = time.time()  # Start timer for FPS measurement

    # YOLO usage HERE
    [boxes, features] = yolo.predecir(test, 0.25, 0.4)

    #   end = time.time()  # End timer for FPS measurement
    #   dif = (end - start)
    #   fps = 1 / dif  # Conversion to frames per second

    # Draw the boxes in the frame
    image_aux = dibujar_deteccion(test, boxes, configuration['model']['tags'])

    # Display the resulting frame
    #   print('Frames por segundo = ' + str(fps))
    cv2.imshow('test', test)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
