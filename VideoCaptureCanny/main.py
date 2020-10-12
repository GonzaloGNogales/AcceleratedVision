import cv2
from random import random
from colorsys import hsv_to_rgb
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # Canny edge detection, RGB transformation and original image overlapping
    edges_img = cv2.Canny(frame, 50, 200)
    rgb_edges_img = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB)
    rgb_edges_img *= np.array((1, 0, 0), np.uint8)

    # Scale edges
    scale_percent = 25  # percent of original size
    width = int(rgb_edges_img.shape[1] + (rgb_edges_img.shape[1] * scale_percent / 100))
    height = int(rgb_edges_img.shape[0] + (rgb_edges_img.shape[0] * scale_percent / 100))
    dim = (width, height)

    # Resize image
    resized_rgb_edges_img = cv2.resize(rgb_edges_img, dim, interpolation=cv2.INTER_AREA)

    # Shape extraction
    h, w, _ = frame.shape
    hh, ww, _ = resized_rgb_edges_img.shape

    # Compute x_off and y_off for placement of upper left corner of resized image
    y_off = round((hh - h) / 2)
    x_off = round((ww - w) / 2)

    # Cropping resized edges image
    cropped_resized_rgb_edges_img = resized_rgb_edges_img[y_off:y_off + h, x_off:x_off + w]

    # Use numpy indexing to place the resized image in the center of background image
    result = resized_rgb_edges_img.copy()
    result[y_off:y_off + h, x_off:x_off + w] = frame

    # Feature application to the original image
    # overlapped_img = np.bitwise_or(result, resized_rgb_edges_img)  # Small image overlapping approach
    # overlapped_img = np.bitwise_or(frame, cropped_resized_rgb_edges_img)  # Cropping image approach
    overlapped_img = np.bitwise_or(frame, rgb_edges_img)  # Adjusted edges approach

    # MSER OPERATIONS
    # Convert the color to grayscale
    #    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize Maximally Stable Extremal Regions (MSER) and output matrix
    #    output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #    mser = cv2.MSER_create(_delta=5, _max_variation=0.5, _max_area=20000, _min_area=60)

    # Detect polygons (regions) from the image
    #    polygons = mser.detectRegions(img)

    # Color output
    #    for polygon in polygons[0]:
    #        colorRGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
    #        colorRGB = tuple(int(color * 255) for color in colorRGB)
    #        output = cv2.fillPoly(output, [polygon], colorRGB)

    # Display the resulting frame
    cv2.imshow('frame', overlapped_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
