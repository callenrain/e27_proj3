#!/usr/bin/env python

import cv2
import numpy
import sys
import os

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)

# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)

# Set up parameters for stereo matching (see OpenCV docs for details)
min_disparity = 0
max_disparity = 16
window_size = 11
param_P1 = 0
param_P2 = 20000


# Create a stereo matcher object
matcher = cv2.StereoSGBM(min_disparity, 
                         max_disparity, 
                         window_size, 
                         param_P1, 
                         param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparities.
disparity = matcher.compute(cam_image, proj_image) / 16.0

M_inv = numpy.array([[ 0.00166667,  0.        , -0.53333333],
        [ 0.        ,  0.00166667, -0.4       ],
        [ 0.        ,  0.        ,  1.        ]])

h, w = disparity.shape

x = numpy.linspace(0, w-1, w)
y = numpy.linspace(0, h-1, h)
xv, yv = numpy.meshgrid(x, y)
ones = numpy.ones((1, h*w))
grid = numpy.dstack((xv, yv)).reshape(h*w, 2).transpose()
wide = numpy.vstack((grid, ones))
mapped = numpy.dot(M_inv, wide)
mapped = mapped[0:2]



baseline = .05
focal_length = 600
z_max = 8
disp_max = (baseline*focal_length)/z_max
a, b = numpy.where(disparity < disp_max)
zero_locs = a*w + b

wide_disparity = disparity.reshape((h*w))
wide_disparity = numpy.delete(wide_disparity, zero_locs)
mapped = numpy.vstack((numpy.delete(mapped[0], zero_locs), numpy.delete(mapped[1], zero_locs)))

z = (baseline*focal_length)/(numpy.array([wide_disparity]))
xyz = numpy.vstack((z*mapped[0], z*mapped[1], z))

numpy.savez(cam_file.split('.')[0]+"_xyz", xyz.transpose())


# Pop up the disparity image.
#cv2.imshow('Disparity', disparity/disparity.max())
#while cv2.waitKey(5) < 0: pass
