import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
]);

kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 1, 2, 1, -1],
    [-1, 2, 4, 2, -1],
    [-1, 1, 2, 1, -1],
    [-1, -1, -1, -1, -1]
]);

#loading the image in gray scale
losAngelesImage = cv2.imread('losAngeles.jpg',0);

# the covolution could be done with numpy 
# but the numpy method only accepts one dimensional convolutions
# we still could use it to make multidimensional covolutions 
# but it is more simple to use the convolve method from scipy
k3 = ndimage.convolve(losAngelesImage, kernel_3x3);
k5 = ndimage.convolve(losAngelesImage, kernel_5x5);
# these k3 and k5 variables are the result of applying
# two hpf to the image

# here we have another way of apply a hpf to a image
# first we apply a lpf to the image 
# and then I subtract the result from the original image
# so, basically, what is gonna rest are the high frequencies
blurred = cv2.GaussianBlur(losAngelesImage, (17,17),0)
g_hpf = losAngelesImage - blurred;
# The image generated from this approach had the best result
# it was because it uses a low pass filter instead of a high pass filter
# so it attenuates the noises on the image

""" cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("blurred", blurred)
cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows() """


# int this kernel the pixel of interest gets a weight of 9
# while the adjacent pixels gets a weight of -1
# so the result will be nine times the current pixel minus the
# adjacent pixels, so it will increase the contrast between the pixels
# and the image become more sharp

kernel = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])

grayLosAngeles = cv2.imread('losAngeles.jpg', 0)
colorfullLosAngeles = cv2.imread('losAngeles.jpg')

# this function filter2D allow us to convolve a image 
# with a kernel of a arbitrary lenght 
# the -1 parameters indicates the depth of the result image 
# (it can be 8 bits per channel)
# with -1 we indicate that the result image will have the same depth
# as the source image

""" cv2.filter2D(grayLosAngeles, -1, kernel, grayLosAngeles)
cv2.filter2D(colorfullLosAngeles, -1, kernel, colorfullLosAngeles)


cv2.imshow("sharp gray", grayLosAngeles)
cv2.imshow("sharp colorfull", colorfullLosAngeles)

cv2.waitKey()
cv2.destroyAllWindows() """

# Edge detection with Canny
# openCV offers a handy function called 
# Canny

""" img = cv2.imread('losAngeles.jpg',0)
cv2.imwrite('cannyLosAngeles.jpg', cv2.Canny(img, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows() """

# the canny edge detection is a complex algorithm
# but it's very interesting. it is a five-step 
# process
# 1. denoise the image with a gaussian filter
# 2. Calculate the gradients
# 3. Apply non-maximum suppresion (NMS) on the edge
#   Basically it mean that the algorithm select the
#   best edges from a set of overlapping edges
# 4. apply a double treshold to all detected edges
#   to eliminate any false positive
# 5. Analyze all the edge and their connections 
#   to each other to select the real ones and 
#   discart the weak ones

# Contour detection

# a vital task in computer vision is contour
# detection 
# let's see an example

""" img = np.zeros((200,200), dtype=np.uint8);
img[50:100, 50:100] = 255;

ret, thresh = cv2.threshold(img, 127, 255, 0);
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows() """

# we create an empty black image that is 
# 200 x 200 pixels in size
# then we place a white square in the center
# by utilizing array's ability to assign 
# values on a slice
# then we threshold the image and call the 
# findContours 
# this function has three parameters 
#   the input imag
#   the hierarchy type
#   and the contour approximation method
# the second parameter specifies the type of 
#   hierarchy tree returned by the function 
#   the arguments we passed specifies that 
#   we want the entire hierarchy of external
#   and internal contours
#       this relationships may matter if we
#       we want to find smaller objects
#       inside larger objects
# if we only want to retrieve the most external
# contours, we can use RETR_EXTERNAL as 
# parameter
# the function findContours returns two elements
# the contours and their hierarchy
# so we used the contours to draw the gree 
# outlines

# Bounding box, minimum area rectangle, and
#   minimum enclosing circle

# find the contours of a white square in a
# black image is simple, but when we have 
# irregular shapes is when we can use the 
# fully potential of the function find
# contours
# the follwing example reads an image from a
# file, converts it into grayscale, applies
# a threshold to the grayscale, and finds the
# contours in the thresholded image

img = cv2.pyrDown(cv2.imread("leBronJames.jpg"), cv2.IMREAD_UNCHANGED)

ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127,
    255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# now for each contours that were found in the
# image we can draw: 
#   the bounding box
#   the minimum enclosing rectangle
#   the minimum enclosing circle

""" for c in contours:
    #find the bounding box coordinates
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)

    #find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum
    # area rectangle
    box = cv2.boxPoints(rect)
    #normaliza the coordinates to integers
    box = np.int_(box)
    #draw contous
    cv2.drawContours(img, [box], 0, (0,0,255),3)
    
    # calculate the center and radius of 
    # minimum enclosing circle
    (x,y), radius = cv2.minEnclosingCircle(c)
    #cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img, center, radius, (0,255,0),2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows() """


# Detecting lines, circles, and other shapes

# foundation is the hough transform

img = cv2.imread('volleyMatch.png')
""" gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)
minLineLength = 20
maxLineGap = 5
lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20,
    minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0),2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows() """


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.medianBlur(gray_img, 5)
circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,120,
param1=100,param2=30,minRadius=5,maxRadius=40)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
cv2.imwrite("img_circles.jpg", img)
cv2.imshow("HoughCirlces", img)
cv2.waitKey()
cv2.destroyAllWindows()
