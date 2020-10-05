# lcam

1. orientation1.py :

This was the first way tried for detecting rotation angle of an object by drawing a rectangle around the object. The values of the angle are only between 0 and 90 degrees. So it gives same angle for original position and 180 degree flipped position.

2. orientation2.py :

Opencv technique "Image Registration" was used int this code. It outputs a MapAffine object by comparing two images. It was not possible to extract angle from that object. Maybe that object is used in middle of some process.
