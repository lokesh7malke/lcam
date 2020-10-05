import numpy as np    
#import pydicom    
import cv2  

image1 = cv2.imread("mobile_0.jpeg")
image2 = cv2.imread("mobile_19.jpeg")

image1 = cv2.resize(image1, (1100, 1100))
image2 = cv2.resize(image2, (1100, 1100))

image1_arr = np.asarray(image1)
image2_arr = np.asarray(image2)

print(image1_arr.shape)
print(image2_arr.shape)

mge_obj = cv2.reg_MapperGradEuclid()
retval = mge_obj.calculate(image1_arr, image2_arr)

#print(retval.shape)

map_proj = cv2.reg.MapTypeCaster_toProjec(retval)
print(map_proj.normalize())


#b = cv2.reg_MapperGradEuclid.calculate(image1_arr,image2_arr)
#retval.normalize()
#print(len(retval))

