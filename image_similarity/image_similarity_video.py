import cv2
import numpy as np

original = cv2.imread("pcb0.jpeg",cv2.IMREAD_GRAYSCALE)
#image_to_compare = cv2.imread("check3.png")

def find_similarity(original, image_to_compare):

	original = cv2.resize(original, (880, 580), interpolation = cv2.INTER_NEAREST)
	image_to_compare = cv2.resize(image_to_compare, (880, 580), interpolation = cv2.INTER_NEAREST)

	# 2) Check for similarities between the 2 images
	sift = cv2.xfeatures2d.SIFT_create()
	kp_1, desc_1 = sift.detectAndCompute(original, None)
	kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

	index_params = dict(algorithm=0, trees=5)
	search_params = dict()
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(desc_1, desc_2, k=2)

	good_points = []
	for m, n in matches:
		if m.distance < 0.6*n.distance:
			good_points.append(m)

	# Define how similar they are
	number_keypoints = 0
	if len(kp_1) <= len(kp_2):
		number_keypoints = len(kp_1)
	else:
		number_keypoints = len(kp_2)

	match_points = len(good_points)
	match_percent = len(good_points)/(number_keypoints) * 100 

	return match_points, match_percent

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

    #drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
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

    #print(int(cv2.arcLength(contours_poly[index],True)))
    #cv2.rectangle(img, (new_x, new_y), (new_w, new_h), color, 2)
    img = img[new_y:new_y+new_h, new_x:new_x+new_w]

    return img

cap = cv2.VideoCapture('http://192.168.20.155:8080/video')
cv2.namedWindow("video_feed",cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    

    image_to_compare = detect_camera_object(frame)
    #image_to_compare = frame
    cv2.imshow("video_feed", image_to_compare)
    k = cv2.waitKey(1)
    image_to_compare = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2GRAY)
    if k%256 == 32:
        # SPACE pressed
        break

    points, percent  = find_similarity(original, image_to_compare)
    print("Match_points :", points)
    print("Match_percentage :", round(percent,2),"%")

#cv2.waitKey(0)
cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()
