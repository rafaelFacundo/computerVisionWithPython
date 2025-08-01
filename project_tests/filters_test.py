import cv2
import numpy as np
import math


def gammaCorrection(image, gammaValue = 1.0):
    inverseGamma = 1/gammaValue
    table = np.array([
        ((i/255.0) ** inverseGamma) * 255 
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

""" def logTransformation(image, constant=1): """




frame = cv2.imread('frame4.png');

scaleFactor_X_axis = 0.7;
scaleFactor_y_axis = 0.7;

frameResized = cv2.resize(
    frame, 
    (420,420), 
    interpolation=cv2.INTER_LINEAR
)

frameResizedGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

frameResizedGrayBlurred = cv2.GaussianBlur(frameResizedGray, (5,5), 1);

frameResizedGrayBlurredWithGameCorrection = gammaCorrection(
    frameResizedGrayBlurred,
    0.5
);

frameResizedGrayBlurredWithGameCorrectionAndMedianBlur = cv2.medianBlur(
    frameResizedGrayBlurredWithGameCorrection,
    5
)

#_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


cannyEdges = cv2.Canny(
    frameResizedGrayBlurredWithGameCorrectionAndMedianBlur,
    50,
    100
)

kernel = np.ones((3,1),np.uint8)

cv2.imwrite("teste.jpg", cannyEdges)

erosion = cv2.dilate(cannyEdges, kernel, iterations=1)

cv2.imshow('frame', binary);
cv2.waitKey();
cv2.destroyAllWindows();