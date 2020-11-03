import time
import cv2
import numpy as np 

start_time = time.time()

image_vec = cv2.imread('image16.jpeg')

edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
blurred_float = image_vec.astype(np.float32) / 255.0
edges = edgeDetector.detectEdges(blurred_float) * 255.0

print(time.time()-start_time)

cv2.namedWindow("window",cv2.WINDOW_NORMAL)
cv2.imshow("window", edges)
cv2.waitKey(0)