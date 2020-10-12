import cv2
import numpy as np
import random as rng
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

#def capture_object():
def detect_camera_object(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    gray = cv2.medianBlur(gray, 11) 
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    canny = cv2.Canny(gray, 20, 255, 1)

    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    new_x = int(0.7*(boundRect[index][0]))
    new_y = int(0.7*(boundRect[index][1]))
    new_w = int(1.2*(boundRect[index][0]+boundRect[index][2]))
    new_h = int(1.2*(boundRect[index][1]+boundRect[index][3]))

    #print(int(cv2.arcLength(contours_poly[index],True)))
    #cv2.rectangle(img, (new_x, new_y), (new_w, new_h), color, 2)
    img = img[new_y:new_y+new_h, new_x:new_x+new_w]

    return img

def predict_output(image,model):
    #model = load_model('final_orientation_model3.h5')
    #img = load_img(name,target_size=(450,300))
    #image1 = load_img(image,target_size=(580,880))
    image = cv2.resize(image, (880, 580), interpolation = cv2.INTER_NEAREST)
    cv2.imshow("video_feed", image)
    k = cv2.waitKey(1)
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    pred = model.predict(input_arr)

    return pred

cap = cv2.VideoCapture('http://192.168.20.155:8080/video')
cv2.namedWindow("video_feed",cv2.WINDOW_NORMAL)
    #print("Capture the object.....")
model = load_model('orientation_pcb_model2.h5')
while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    
    
    detected_object = detect_camera_object(frame)
    out = predict_output(detected_object,model)
    print("The label for your image is :", out)

cap.release()
#detected_object = detect_camera_object()
#out = predict_output(detected_object)
#cv2.imshow('Contours', canny)
#print("The label for your image is :", out)
#cv2.namedWindow('Detected_object',cv2.WINDOW_NORMAL)
#cv2.imshow('Detected_object', detected_object)
#cv2.waitKey(0)
cv2.destroyAllWindows()




