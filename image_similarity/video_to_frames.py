import cv2
import os

cap = cv2.VideoCapture('http://192.168.20.155:8080/video')

def detect_camera_object(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    gray = cv2.medianBlur(gray, 11) 
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    canny = cv2.Canny(gray, 20, 255, 1)

    _, contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
    	epsilon = 0.1*cv2.arcLength(c,True)
    	contours_poly[i] = cv2.approxPolyDP(c,epsilon,True)
    	boundRect[i] = cv2.boundingRect(contours_poly[i])

    color = (0,0,255)

    max_perimeter =  int(cv2.arcLength(contours_poly[0],True))
    index = 0
    for i in range(1,len(contours)):
    	perimeter = int(cv2.arcLength(contours_poly[i],True))
    	if perimeter > max_perimeter :
    		max_perimeter = perimeter
    		index = i

    new_x = int(0.8*(boundRect[index][0]))
    new_y = int(0.8*(boundRect[index][1]))
    new_w = int(1.2*(boundRect[index][0]+boundRect[index][2]))
    new_h = int(1.2*(boundRect[index][1]+boundRect[index][3]))

    img = img[new_y:new_y+new_h, new_x:new_x+new_w]

    return img

count = 0
while True:
    ret, frame = cap.read()
    if count > 3:
    	break
    if ret:
    	cv2.imwrite('image_similarity_opencv/pcb'+str(count)+'.jpeg', detect_camera_object(frame))
    	print("saved -",count)
    #cv2.imshow("video_feed", frame)
    #k = cv2.waitKey(1)
    #if k%256 == 32:
        # SPACE pressed
    #    break

    count += 1

cap.release()