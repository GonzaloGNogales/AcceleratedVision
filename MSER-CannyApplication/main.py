import cv2
from random import random
from colorsys import hsv_to_rgb
import numpy as np

# Read the input photo
input_img = cv2.imread('yo.png')

# Resize image to 600 x 694
resized_img = cv2.resize(input_img, (600, 694), interpolation=cv2.INTER_AREA)

# Gaussian Blur Applied to see the result
# blurred_img = cv2.GaussianBlur(resized_img, (7, 7), cv2.BORDER_DEFAULT)

# Canny edge detection, RGB transformation and original image overlapping
edges_img = cv2.Canny(resized_img, 50, 200)
rgb_edges_img = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB)
rgb_edges_img *= np.array((1, 0, 0), np.uint8)

# Feature application to the original image
overlapped_img = np.bitwise_or(resized_img, rgb_edges_img)

# Black background creation
# black_background = np.zeros((edges_img.shape[0], edges_img.shape[1], 3), dtype=np.uint8)


# MSER OPERATIONS
# Convert the color to grayscale
img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Initialize Maximally Stable Extremal Regions (MSER) and output matrix
output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
mser = cv2.MSER_create(_delta=5, _max_variation=0.5, _max_area=20000, _min_area=60)

# Detect polygons (regions) from the image
polygons = mser.detectRegions(img)

# Color output
for polygon in polygons[0]:
    colorRGB = hsv_to_rgb(random(), 1, 1)  # Generate a random color
    colorRGB = tuple(int(color*255) for color in colorRGB)
    output = cv2.fillPoly(output, [polygon], colorRGB)

# Show output
cv2.imshow('Colored Image', resized_img)
cv2.imshow('Canny Image To RGB', rgb_edges_img)
cv2.imshow('Gray Image', img)
cv2.imshow('Result', output)
cv2.imshow('Overlapped Resulting image', overlapped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
