#at max 700ms
import time
import cv2
import numpy as np 



edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")

start_time = time.time()
image_vec = cv2.imread('image5.jpeg')


blurred_float = image_vec.astype(np.float32) / 255.0
edges = edgeDetector.detectEdges(blurred_float)*255.0

gray1 = edges + edges + edges
gray1 = gray1.astype(np.uint8)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray1,kernel,iterations = 1)
kernel = np.ones((7,7),np.uint8)
dilated = cv2.dilate(erosion, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]


(new_x, new_y, new_w, new_h) = cv2.boundingRect(cnt)
start_point = (new_x, new_y)
end_point = (new_x+new_w, new_y+new_h)

cv2.rectangle(image_vec, start_point, end_point, (0,0,255), 2)
hull = cv2.convexHull(cnt)
cv2.drawContours(image_vec,[hull], -1, (255,0,0), 3)

print(time.time()-start_time)

cv2.namedWindow("window",cv2.WINDOW_NORMAL)
cv2.imshow("window", image_vec)
cv2.waitKey(0)