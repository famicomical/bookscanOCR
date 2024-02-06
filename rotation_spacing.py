"""
Automatically detect rotation and line spacing of an image of text using
Radon transform

If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)

It doesn't work with black borders

input is a grayscale image in array-like form
"""
from skimage.transform import radon
import numpy as np
try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic, parabolic_polyfit

    def argmax(x):
	    return parabolic_polyfit(x, np.argmax(x),3)[0]
except ImportError:
    from numpy import argmax


def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.abs(a) ** 2))


def get_angle(img, anglerange):

	# return the angle required to make page straight
	
	img = img - np.mean(img)  # Demean; make the brightness extend above and below zero

	#speed up by assuming no deviations over +/- anglerange degrees
	rot_increment=0.25	#rotate by rot_increment degrees per iteration of radon transform
	offset=90-anglerange

	angles=np.arange(90-anglerange,90+anglerange,rot_increment) 
	sinogram = radon(img,theta=angles, circle=False)
	r = np.array([rms_flat(line) for line in sinogram.transpose()])

	#convert index of r to actual degrees
	rotation = (argmax(r)*rot_increment+offset)
	return (90-rotation)

