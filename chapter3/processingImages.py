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

img = cv2.imread('losAngeles.jpg',0)
cv2.imwrite('cannyLosAngeles.jpg', cv2.Canny(img, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()






