# lcam

1. _orientation1.py_ :

This was the first way tried for detecting rotation angle of an object by drawing a rectangle around the object. The values of the angle are only between 0 and 90 degrees. So it gives same angle for original position and 180 degree flipped position.

2. _orientation2.py_ :

Opencv technique "Image Registration" was used int this code. It outputs a MapAffine object by comparing two images. It was not possible to extract angle from that object. Maybe that object is used in middle of some process.

3. _generate__augmented__images.ipynb_ :

This file is used image augmentation which is very much useful for small training data. It uses a combination of various possible changes in the images to increase the number of images.

4.  _final__orientation__.ipynb_ :

Code used for Training the model on ESP images and got accuracy upto 100%
