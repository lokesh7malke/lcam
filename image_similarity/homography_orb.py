import cv2
import math
import numpy as np

original = cv2.imread("image8.jpeg",0)
image_to_compare = cv2.imread("image8_15.jpeg",0)

original = cv2.resize(original, (880, 580), interpolation = cv2.INTER_NEAREST)
image_to_compare = cv2.resize(image_to_compare, (880, 580), interpolation = cv2.INTER_NEAREST)


orb = cv2.ORB_create()
kp_1, desc_1 = orb.detectAndCompute(original, None)
kp_2, desc_2 = orb.detectAndCompute(image_to_compare, None)

#matcher = cv2.BFMatcher()   
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc_1,desc_2)

src_pts = np.float32([ kp_1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
# Define how similar they are
number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
print(M.shape)

print("GOOD Matches:", len(matches))

theta = - math.atan2(M[0,1], M[0,0]) * 180 / math.pi
print(theta)

final_img = cv2.drawMatches(original, kp_1,  
image_to_compare, kp_2, matches,None) 
   
#final_img = cv2.resize(final_img, (1000,650)) 
  
# Show the final image 
cv2.namedWindow("Matches",cv2.WINDOW_NORMAL)
cv2.imshow("Matches", final_img) 
cv2.waitKey(0) 


