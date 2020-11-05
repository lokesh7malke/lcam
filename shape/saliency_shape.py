import cv2
import time
import numpy as np

start = time.time()

original = cv2.imread('check4.jpeg')

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(original)
saliencyMap = saliencyMap*255
check = saliencyMap.astype(np.uint8)

thresh1 = cv2.adaptiveThreshold(check, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5) 

canny = cv2.Canny(thresh1, 100, 255, 1)
kernel = np.ones((3,3),np.uint8)
dilated = cv2.dilate(canny, kernel,iterations = 1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]

(new_x, new_y, new_w, new_h) = cv2.boundingRect(cnt)
start_point = (new_x, new_y)
end_point = (new_x+new_w, new_y+new_h)

cv2.rectangle(original, start_point, end_point, (0,0,255), 2)
#hull = cv2.convexHull(cnt)
cv2.drawContours(original,[cnt], -1, (255,0,0), 3)

print(time.time()-start)

cv2.namedWindow("window",cv2.WINDOW_NORMAL)
cv2.imshow("window", original)
cv2.waitKey(0)
