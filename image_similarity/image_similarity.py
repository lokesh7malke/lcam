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

number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)


print("GOOD Matches:", len(good_points))
print("How good it's the match: ", len(good_points) / number_keypoints * 100)

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

cv2.namedWindow("result",cv2.WINDOW_NORMAL)
cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))

cv2.waitKey(0)
cv2.destroyAllWindows()