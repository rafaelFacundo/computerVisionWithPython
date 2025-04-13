import numpy as np
import cv2
from matplotlib import pyplot as plt

#Capturing frames from a depth camera

# depth cameras usually have others sensors
# capturing more informations about the 
# distance between the object and the camera

# back in the chapter 2 we discussed the
# concept that a computer can have multi
# ple channels
# suppose a given device is a depth camera
# each channel might correspond to a
# different lens and sensor
# also each channel might correspond to
# different kinds of data
# such as normal color image versus a depth
# map
# OpenCV via its optional support for
# openNI 2, allows us to requet any of the
# following channels from a depth camera
# for example
#   CAP_OPENNI_DEPTH_MAP
#       a depth map, a grayscale image in
#       which each pixel value is the estimated
#       distance from the came
#   CAP_OPENNI_POINT_CLOUD_MAP
#       a color image in which each color
#       corresponds to a x, y, or z
#       spatial dimension
#   CAP_OPENNI_DISPARITY_MAP
#       this is a disparity map, that is, a
#       grayscale image in which each pixel
#       value is the stereo disparity of a surface
#   CAP_OPENNI_VALID_DEPTH_MASK
#       this a valid depth mask that shows
#       whether the depth information at a 
#       given pixel is believed to be valid
#   CAP_OPENNI_BGR_IMAGE

# Converting 10-bit images to 8-bit

# as we noted, some of the channels of a 
# depth camera use a range larger than
# 8 bits for their data. So, a large range
# is useful for computations, but inconvenient 
# to display 
# imshow function re-scales and truncates the
# given input data in order to convert
# the image for display
# code made in managers.py

# Creating a mask from a disparity map

# let's assume that a user's face, or some
# other object of interest, occupies most
# of the depth camera's field of view
# however the image also contains some other
# content that is not of interest 
# by analyzing the disparity map we can tell
# that some piels within the reactangle
# are outliers - too near or too far to really be a
# part of a face or another object of interst
# we can make a maske to exclude this outliers
# code in depth.py in the cameo project

# Depth estimation with a normal camera

# We don't have a depth camera so we can
# try to make a depth estimation by using
# normal cameras, we can do it using 
# triangulation from different camera 
# perspectives
# if we use two cameras simultaneously 
# this approach is called stereo vision
# if we use one camera, but we move it over 
# time to obtain different perspectives, this
# approach is called structure from motion
# 
# For the stereo motion we use the epipolar
# geometry 
# it works by trace imaginary lines from the 
# camera to each object in the image, then
# does the same on the second image, and calculates
# the distance to an object based on the intersection
# of the lines corresponding to the same object
# let's see how openCV does it

# defining the parameters of the stereo algorithm

minDisparity = 16
numDisparities = 192 - minDisparity
blockSize = 5
uniquenessRatio = 1
speckleWindowSize = 3
speckleRange = 3
disp12MaxDiff = 200
P1 = 600
P2 = 2400

stereo = cv2.StereoSGBM_create(
    minDisparity = minDisparity,
    numDisparities = numDisparities,
    blockSize = blockSize,
    uniquenessRatio = uniquenessRatio,
    speckleRange = speckleRange,
    speckleWindowSize = speckleWindowSize,
    disp12MaxDiff = disp12MaxDiff,
    P1 = P1,
    P2 = P2
)

# TO DO: finish this implementation 

# Foreground detection with the GrabCut algorithm

""" original = cv2.imread('angel.png')
img = original.copy()
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (100, 1, 421, 378)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])
plt.show() """

#Image segmentation with the Watershed algorithm

img = cv2.imread('lebronJames2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Remove noise.
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
    iterations = 2)

# Find the sure background region.
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Find the sure foreground region.
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(
    dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# Find the unknown region.
unknown = cv2.subtract(sure_bg, sure_fg)

# Label the foreground objects.
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1.
markers += 1
# Label the unknown region as 0.
markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers==-1] = [255,0,0]

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
