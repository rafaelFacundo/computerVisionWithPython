import cv2


# Building Custom Object Detectors
# 
# one of the most common challenges in computer
# vision 
# 
# Understanding HOG descriptors
# HOG is a feature descriptor, so it belongs to 
# the same family as algorithm that we saw in the 
# brevious chapter surf, sift and orb
# the algorithm is really clever 
# an image is divided into cells and a set of
# gradients is calculated for each cell
# each gradient describes the change in pixel
# intensities in a given direction
# together, these gradientes form a histogram 
# representation of the cell
# 
# Visualizing HOG
# 
# Using HOG to describe regions of an image
# 
# Detecting people with HOG descriptors
# 
""" 
def is_inside(i, o):
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('LAwomen.jpg')
found_rects, found_weights = hog.detectMultiScale(
    img, winStride=(4, 4), scale=1.02)

found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])

for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv2.putText(img, text, (x, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
cv2.imshow('Women in LA Detected', img)
cv2.imwrite('./women_in_LA.jpg', img)
cv2.waitKey(0) """

# Creating and training an object detector

# Understanding BoW

# Is the technique by which we assign a weight
# or count to each word in a series of documents
# we then represent these documents with vectors
# of the counts 

# Applying BoW to computer vision

# using what we saw in the previous section 
# we can follow this steps to build a classifier
# 1 Take a sample dataset of images.
# 2 For each image in the dataset, extract descriptors (with SIFT, SURF, ORB, or a
# similar algorithm).
# 3 Add each descriptor vector to the BoW trainer
# 4 Cluster the descriptors into k clusters whose centers (centroids) are our visual
# words. This last point probably sounds a bit obscure, but we will explore it
# further in the next section.

# missing implementions