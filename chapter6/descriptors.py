import cv2
import matplotlib.pyplot as plt
import numpy as np

# basically in this chapter we will try to
# identify features in an image and with this
# perform a lot of actions like for example
# search for similiar parts in two images
# 
# Understanding types of feature detection and matching 
# 
# a number of algorithms can be used to detect
# and describe features 
# the most common used in openCV are
# Harris
#   useful for detecting corners
# Sift 
#   useful for detecting blobs
# SURF
#   useful for detecting blobs
# Fast
#   useful for detecting corners
# Brief
#   useful for detecting blobs
# Orb 
#   this algorithms stands for 
#   Oriented FAST and Rotated BRIEF
#   useful for detecting a combination of
#   corners and blobs
# Matching features can be performed with the following methods
# Brute-force matching
# FLANN-based matching
# Spatial verification can then be performed with homography.

# Defining features

# what is it ? 
# Broadly speakin, a feature is an area of 
# interest in the image that is unique or easily
# recognizable

# Detecting Harris corners

""" img = cv2.imread('chess.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2,23, 0.04)
img[dst > 0.01 * dst.max()] = [0,0,255]
cv2.imshow('corners', img)
cv2.waitKey()
cv2.destroyAllWindows() """

# Detecting DoG features and extracting SIFT descriptors

""" img = cv2.imread('madisonSquare.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
cv2.drawKeypoints(img, keypoints, img, (51, 163, 236), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift_keypoints', img)
cv2.waitKey()
cv2.destroyAllWindows() """

# Anatomy of a keypoint

# each keypoint is an instance of the cv2.KeyPoint
# class, which has the following properties
#   pt(point)
#   size -> diameter
#   angle -> orientation
#   reponse -> the strength 
#   octave -> the layer where the feature was found
#   class_id -> can be used to assign a custom identifier to a keypoint
#   
#  Detecting Fast Hessian features and extracting SURF descriptors 

""" img = cv2.imread('madisonSquare.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(8000)
keypoints, descriptors = surf.detectAndCompute(gray, None)
cv2.drawKeypoints(img, keypoints, img, (51, 163, 236), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift_keypoints', img)
cv2.waitKey()
cv2.destroyAllWindows() """

# Using ORB with FAST features and BRIEF descriptors

# yonger that SIFT and SURF, and alson faster 
# mixes the techniques used in the FAST keypoint
# detector and the BRIEF keypoint descriptor
# so lets take a look at this two
# FAST
#   The features from accelerated segment test
#   works by analyzing circular neighborhoods
#   of 16 pixels. it marks each pixel in a neighborhood
#   as brighter or darker than a particular threshold
#   which is defined relative to the center of the circle
#   a neighborhood is deemed to be a corner if it contains a number
#   of contigous pixels marked as brighter or darker
# BRIEF
#   binary robust independent elementary features
#   on the other hand, is not a feature detection algorithm
#   but a descriptor 

# Brute-force matching
# A brute-force matcher is a descriptor matcher that compares two sets of keypoint
# descriptors and generates a result that is a list of matches.
# 
# Matching a logo in two images

""" img0 = cv2.imread('nasaLogo.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('kennedySpaceCenter.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des0, des1)

matches = sorted(matches, key=lambda x:x.distance)

img_matches = cv2.drawMatches(
    img0, kp0, img1, kp1, matches[:25], img1,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show()
 """

# Filtering matches using K-Nearest Neighbors and the ratio test


""" img0 = cv2.imread('nasaLogo.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('kennedySpaceCenter.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2)

pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)

matches = [x[0] for x in pairs_of_matches
    if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

img_matches = cv2.drawMatches(
img0, kp0, img1, kp1, matches[:25], img1,
flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show() """

# Matching with FLANN

# Fast Library for Approximate Nearest Neighbors
# 

""" img0 = cv2.imread('paint.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('paintSearch.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des0, des1, k=2)

mask_matches = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        mask_matches[i]=[1, 0]

img_matches = cv2.drawMatchesKnn(
    img0, kp0, img1, kp1, matches, None,
    matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
    matchesMask=mask_matches, flags=0)

plt.imshow(img_matches)
plt.show()
 """

# Performing homography with FLANN-based matches

