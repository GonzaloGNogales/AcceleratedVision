import numpy as np
import cv2

# For storing 5 descriptors that correspond to a binary code we do:
descriptors = np.array([[[1, 1, 1]], [[8, 8, 8]], [[8, 8, 9]], [[9, 9, 9]], [[1, 3, 1]]], dtype=np.uint8)

# For accomplishing that we create a FlannBasedMatcher using Hamming Distance
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
search_params = dict(checks=-1)  # Maximum leafs to visit when searching for neighbours.
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Then we store the descriptors. This could be done while the descriptors are being calculated
for d in descriptors:
    flann.add([d])

# Now we could search for the k most similar descriptors to [[8,8,8]]
results = flann.knnMatch(np.array([[8,8,8]], dtype=np.uint8), k=3)

# Show the result
for r in results:
    for m in r:
        print("Res - dist:", m.distance, " img: ", m.imgIdx, " queryIdx: ", m.queryIdx, " trainIdx:", m.trainIdx)