import cv2
import numpy as np

def gammaCorrection(image, gammaValue = 1.0):
    if image is not None:
        inverseGamma = 1/gammaValue
        table = np.array([
            ((i/255.0) ** inverseGamma) * 255 
            for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)
    raise Exception("Gamma correction need an image as parameter")


def resize(image, xfactor=-1, yfactor=-1, proportion=(-1,-1)):
    if image is None:
        raise Exception("resize needs an image as parameter")
    if proportion[0] == -1 and proportion[1] == -1 and xfactor == -1 and yfactor == -1:
        raise Exception("You need to set xFactor and yFactor or proportion")
    frameResized = np.zeros(10);
    if (xfactor == -1 or yfactor == -1) and proportion[0] != -1 and proportion[1] != -1 :
        frameResized = cv2.resize(
            image, 
            proportion, 
            interpolation=cv2.INTER_LINEAR
        )
        return frameResized
    elif (proportion[0] == -1 or proportion[1] == -1) and xfactor != -1 and yfactor != -1 :
        frameResized = cv2.resize(
            image, 
            None, 
            fx=xfactor, 
            fy=yfactor,
            interpolation=cv2.INTER_LINEAR
        )
        return frameResized

def convertImageToGrayScale(image):
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    raise Exception("Convert image to gray scale needs an image as parameter")

def applyGaussianBlur(image, kernelSize=(3,3), standartDeviation=1):
    if image is not None:
        return cv2.GaussianBlur(image, kernelSize, standartDeviation);
    raise Exception("apply gaussian blur needs an image as parameter")


def applyMedianBlur(image, kernelSize=3):
    if image is not None:
        return cv2.medianBlur(image, kernelSize)
    raise Exception("Apply median blur needs an image as parameter")

def applyCannyEdgeDetection(image, thresholdOne=50,thresholdTwo=12):
    if image is not None:
        imageBlurred = applyGaussianBlur(image)
        return cv2.Canny(imageBlurred, thresholdOne, thresholdTwo)
    raise Exception("Apply canny edge detection needs an image as parameter")

def cannyWithBlurAndGamma(image):
    if image is not None:
        imageGrayBlurred = cv2.GaussianBlur(image, (5,5), 1);
        imageGrayBlurredWithGameCorrection = gammaCorrection(imageGrayBlurred, 0.4);
        imageGrayBlurredWithGameCorrectionAndMedianBlur = cv2.medianBlur(
            imageGrayBlurredWithGameCorrection,
            1
        )
        cannyEdges = cv2.Canny(
            imageGrayBlurredWithGameCorrectionAndMedianBlur,
            30,
            12
        )
        return cannyEdges
    raise Exception("Apply canny With Blur And Gamma needs an image as parameter")
