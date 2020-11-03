from skimage.measure import compare_ssim
import time
import cv2
import numpy as np

start_time = time.time()

before = cv2.imread("image5.jpeg")
after = cv2.imread("image61.jpeg")

before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(before_gray, after_gray, full=True)
diff = (diff * 255).astype("uint8")

kernel = np.ones((5,5),np.uint8)
diff = cv2.erode(diff,kernel,iterations = 1)

canny = cv2.Canny(diff,100,255,1)

kernel = np.ones((5,5),np.uint8)
dilated = cv2.dilate(canny, kernel)
check = dilated.copy()

_, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_perimeter =  int(cv2.contourArea(contours[0]))
index = 0
for i in range(1,len(contours)):
    perimeter = int(cv2.contourArea(contours[i]))
    if perimeter > max_perimeter :
    	max_perimeter = perimeter
    	index = i

color = (0,0,255)

(new_x, new_y, new_w, new_h) = cv2.boundingRect(contours[index])
start_point = (new_x, new_y)
end_point = (new_x+new_w, new_y+new_h)

cv2.rectangle(before, start_point, end_point, color, 2)
cv2.drawContours(before,[contours[index]], -1, (255,0,0), 3)

print(time.time()-start_time)

cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.imshow('mask',before)
cv2.waitKey(0)
