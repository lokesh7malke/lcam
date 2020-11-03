import cv2

im1 = cv2.imread("img1.jpeg",0)
im2 = cv2.imread("img2.jpeg",0)


d1 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I2,0)

print(d1)
