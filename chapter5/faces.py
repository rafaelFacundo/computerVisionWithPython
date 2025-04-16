import cv2

# conceptualizing haar cascades

# Specifically, we
# look at Haar cascade classifiers, which analyze the contrast between adjacent image regions
# to determine whether or not a given image or sub image matches a known type.

# features are the abstractions that we 
# extract from the image data
# though any pixel might influence multiple
# features
# a set of features is represented as a 
# vector
# and the similarity can be calculated by
# measuring the distance between the images'
# corresponding feature vectors
# each haar-like features describes the pattern
# of contrast among adjacent image regions
# for example, edges, vertices, and thin lines
# each generate a kind of feature. Some features
# are distinctive in the sense that they typically occur
# in a certain class of object (such as face)
# but not in other objects
# these distinctive features can be organized
# into a hierarchy, called a cascade, in which
# the highest layers contain features
# of greatest distinctiveness, enabling a classifier to 
# quickly reject subjects that lack these features

# For any given subject the features may vary
# depending on the scale of the image
# and the size of the neighborhood winthin which
# contrast is being evaluated
# the latter is called the window size.
# to make a haar cascade classifier scale-invariant
# the window size is kept constant but images
# are rescaled a number of times
# so the size of an object may match the window size

# together the original image and the rescale images
# are called the image pyramid
# and each level in pyramid is a smaller
# rescale image

# Getting Haar cascade data

# Using OpenCV to perform face detection

# Performing face detection on a still image

# the first and most basic way to perform 
# face detection is to load an image and 
# detect faces in it

""" face_cascade = cv2.CascadeClassifier('./project/cascades/haarcascade_frontalface_default.xml')
img = cv2.imread('kobeShaq.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.08, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
#cv2.namedWindow('Face dectetor')
cv2.imshow("Face nba ", img)
cv2.imwrite('kobeShaqDetected.jpg', img)
cv2.waitKey(0) """

# Performing face detection on a video

face_cascade = cv2.CascadeClassifier(
'./project/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
'./project/cascades/haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 1.03, 5, minSize=(40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey),
                    (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)