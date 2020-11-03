import cv2
import time
import numpy as np

start_time = time.time()
original = cv2.imread('003.jpeg')
template = original.copy()

hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
canny = cv2.Canny(hsv, 80, 255, 1)

kernel = np.ones((7,7),np.uint8)
dilated = cv2.dilate(canny, kernel)
check = dilated.copy()

_, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

color = (0,0,255)

max_perimeter =  int(cv2.contourArea(contours[0]))
index = 0
for i in range(1,len(contours)):
    perimeter = int(cv2.contourArea(contours[i]))
    if perimeter > max_perimeter :
    	max_perimeter = perimeter
    	index = i

(new_x, new_y, new_w, new_h) = cv2.boundingRect(contours[index])

start_point = (new_x, new_y)
end_point = (new_x+new_w, new_y+new_h)

cv2.rectangle(original, start_point, end_point, color, 2)
cv2.drawContours(original,[contours[index]], -1, (255,0,0), 3)

cv2.namedWindow("video_feed",cv2.WINDOW_NORMAL)
cv2.imshow("video_feed",original)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey(0)

