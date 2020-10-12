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

yolo.cargar_pesos(pretrained_weights_address)

# Video capture initialization
video_capture = cv2.VideoCapture('prueba1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec configuration for the video
video_fps = video_capture.get(cv2.CAP_PROP_FPS)
ret, frame = video_capture.read()  # Reading first frame
output_video = cv2.VideoWriter('salida.mp4', fourcc, video_fps, (frame.shape[0], frame.shape[0]))

num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print('The number of frames is: ' + str(num_frames))
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
frame_actual = 0

while frame_actual < num_frames - 10:

    ret, frame = video_capture.read()

    frame = frame[:, int(frame.shape[1] / 2) - int(frame.shape[0] / 2):int(frame.shape[1] / 2) + int(frame.shape[0] / 2), :]
    test = frame.copy()

    start = time.time()

    [boxes, features] = yolo.predecir(test, 0.25, 0.4)

    end = time.time()
    dif = (end - start)
    fps = 1 / dif  # Conversion to frames per second

    image_aux = dibujar_deteccion(test, boxes, configuration['model']['tags'])

    output_video.write(test)
    cv2.imshow('test', test)
    cv2.waitKey(1)
    if frame_actual % 120 == 0:
        print('Frame actual = ' + str(frame_actual))
        print('Frames por segundo = ' + str(fps))

    frame_actual = frame_actual + 1

output_video.release()
