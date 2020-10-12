import numpy as np
import cv2
import matplotlib as plt

img1 = cv2.imread('yo2.png', 0)  # Query image
img2 = cv2.imread('gafas3.png', 0)  # Train image

detector = cv2.xfeatures2d.SIFT_create()
# detector = cv2.ORB_create()

# Find keypoints and descriptors
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

# Brute-Force Matcher with default parameters
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# Apply ratio test
good = [[m] for m,n in matches if m.distance < 0.8 * n.distance]
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow('result', img3)
cv2.waitKey(0)
