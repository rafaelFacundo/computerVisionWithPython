import cv2
import os
import numpy

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

""" face_cascade = cv2.CascadeClassifier(
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
        cv2.imshow('Face Detection', frame) """


# Performing face recognition

# detecting faces is a fantastic feature
# of openCV and one that constitutes the basis
# for a more advanced operation -> face recognition
# that is, the ability of a program, given an image
# or a video feed containing a person's face
# to identify that person
# one of the ways to achieve this is to train
# the program by feeding it a set of classified
# pictures and to perform recognition based 
# on features of those pictures

# Generating the data for face recognition

# let's write a script that will generates this
# images for us

""" output_folder = './project/data/imgs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_cascade = cv2.CascadeClassifier(
'./project/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
'./project/cascades/haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
count = 0

while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5, minSize=(120,120))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0,0), 2)
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            cv2.imwrite(face_filename, face_img)
            count += 1
        cv2.imshow('Capturing Faces...', frame)
 """

# Recognizing faces

# openCV implements three different algorithms
# for recognizing faces
#   Eigenfaces, Fisherfaces, and Local Binary Pattern Histograms (LBPHs)
# Eigenfaces and Fisherfaces are derived from a more 
# general proporse algorithm called principal component analysis
# (PCA) 

# Loading the training data for face recognition

def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        #print(dirname, subdirnames, filenames)
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename),
                cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # The file cannot be loaded as an image.
                    # Skip it.
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)
    return names, training_images, training_labels


path_to_training_images = '/home/rafael/Documents/computerVisionWithPython/chapter5/project/data/imgs'
training_image_size = (200, 200)
names, training_images, training_labels = read_images(
path_to_training_images, training_image_size)



# Performing face recognition with Eigenfaces

print("TRAINING")
model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, training_labels)
print("TRAINING FINISHED")

face_cascade = cv2.CascadeClassifier(
'./project/cascades/haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[x:x+w, y:y+h]
            if roi_gray.size == 0:
                # The ROI is empty. Maybe the face is at the image edge.
                # Skip it.
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)
            text = '%s, confidence=%.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', frame)