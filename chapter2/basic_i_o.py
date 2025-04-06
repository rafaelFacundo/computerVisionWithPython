import numpy
import cv2
import os 

#creating a 3x3 square black image
img = numpy.zeros((3,3), dtype=numpy.uint8);

print(img)
# by printing this image we could see that each
# pixel is represented by a 8-bit number(a byte)
# which means that each pixel are in the 0-255
# range, where 0 is black and 255 is white
# and the values between this two are shades of 
# gray, in other words, this is a grayscale image

# now let's convert this image to a BGR format
# using this function
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
print('\n')
# let's see how the image changed
print(img)

# now we can see that each pixel is a three-element
# array, with each integer representing one of the
# three color B, G, and R, respectively 

# we can inspect the structure of a image
# rows, columns, and the number of channels 
print('\n')
img = numpy.zeros((5, 3), dtype=numpy.uint8)
print(img.shape)
# this print (5,3) which indicates that we have a
# grayscale image with 5 rows and three columns
# if we then convert this image to a BGR format and
# print the shape again we could see (5,3,3) which
# means that now we have three channels for each 
# pixel

# we can also convert a image format to another 
# format, for example we can convert a png image to
# jpeg image 
image = cv2.imread('courage.png');
cv2.imwrite('courageJPEG.jpg', image)

# by default the imread fuction returns a image 
# in the BGR format even if the file uses a
# grayscale format, BGR represents the same color
# model as RGB but whit the byte order reversed

# optionally, we may specify the mode of imread

# we can load, for example, a png file as a 
# grayscale image (losing color information)
# and then save the image

grayImage = cv2.imread('courage.png', cv2.IMREAD_GRAYSCALE);
cv2.imwrite('courageGrayScale.png', grayImage);

#Converting between an image and raw bytes

# an OpenCV image is a 2d or 3d array of the
# numpy.array type. An 8-bit grayscale image is a
# 2d array containing byte values.
# a 24-bit BGR image is a 3d array, which also contains
# byte values

# and we can convert byte arrays to a numpy.array
# type

# Make an array of 120,000 random bytes
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

#convert the array to make a 400x300 grayscale image
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('randomGray.png', grayImage)

# convert the array to make a 400x100 color image
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('randomColor.png', bgrImage)

#Accessing image data with numpy.array
# numpy array comes with a bunch of manipulations
# that makes the image manipulations easier
# let's say we want to change the color of the pixel
# at coordinate (0,0) we can do

img = cv2.imread('courage.png')
img[0, 0] = [255, 255, 255]
cv2.imwrite('whiteDotCourage.png', img)

# lets say we want to change the blue value of the
# pixel at coordinate (150,120)

img = cv2.imread('whiteDotCourage.png')
img.itemset((150, 120, 0), 255) # Sets the value of a pixel's blue channel
print(img.item(150, 120, 0))
cv2.imwrite('whiteDotCourage.png', img)

# lets say we to manipulate a whole channel 
# the green channel
img = cv2.imread('whiteDotCourage.png')
img[:,:,1] = 0
cv2.imwrite('whiteDotCourage.png', img)

# there are several interesting things we can do
# by accessing raw pixels with numpy's array
# one of them is defining regions of interests(ROI)
# once the region is define we can peform a number
# of operations
# for example we can bind this region to a variable
# define a second region and assign the value of the
# first region to the second 
# hence, copying a portion of the image over to 
# another position in the image

# the shapes have to be the same

img = cv2.imread('losAngeles.jpg')
my_roi = img[0:200, 0:200]
img[300:500, 300:500] = my_roi
cv2.imwrite('losAngelesPartChange.jpg', img)

#Reading/writing a video file
""" 
videoCapture = cv2.VideoCapture('santaMonica.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter(
'MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'),
fps, size)

success, frame = videoCapture.read()
while success: 
    videoWriter.write(frame)
    success, frame = videoCapture.read() """

#Capturing camera frames

# A stream of camera frames is represented by a 
# VideoCapture object too
# however, for a camera, we construct a videoCapture
# by passing the camera's device index instead of a
# video's filename


""" cameraCapture = cv2.VideoCapture(0)
fps = 30 # An assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
'MyOutputVideoCamera.avi', cv2.VideoWriter_fourcc('I','4','2','0'),
fps, size)
success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1 # 10 seconds of frames
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1 """

# TO DO: search about the following warning that
# appears when run this camera code
#[ WARN:0@1.025] global ./modules/videoio/src/cap_gstreamer.cpp 
# (1405) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1

# Displaying an image in a window
""" img = cv2.imread('losAngeles.jpg')
cv2.imshow('Los angeles', img)
cv2.waitKey()
cv2.destroyAllWindows() """

#Displaying camera frames in a window
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)
print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()