import cv2
import math
import numpy as np

original = cv2.imread("image8.jpeg")
image_to_compare = cv2.imread("image8_15.jpeg")

original = cv2.resize(original, (880, 580), interpolation = cv2.INTER_NEAREST)
image_to_compare = cv2.resize(image_to_compare, (880, 580), interpolation = cv2.INTER_NEAREST)

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

src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_points ]).reshape(-1,1,2)
dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_points ]).reshape(-1,1,2)
# Define how similar they are
number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
print(M.shape)

print("GOOD Matches:", len(good_points))

theta = - math.atan2(M[0,1], M[0,0]) * 180 / math.pi
print(theta)

