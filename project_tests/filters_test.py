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
    None, 
    fx=scaleFactor_X_axis, 
    fy=scaleFactor_y_axis,
    interpolation=cv2.INTER_LINEAR
)

frameResizedGray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY);

frameResizedGrayBlurred = cv2.GaussianBlur(frameResizedGray, (3,3), 1);

frameResizedGrayBlurredWithGameCorrection = gammaCorrection(frameResizedGrayBlurred, 0.4);

frameResizedGrayBlurredWithGameCorrectionAndMedianBlur = cv2.medianBlur(
    frameResizedGrayBlurredWithGameCorrection,
    3
)

cannyEdges = cv2.Canny(
    frameResizedGrayBlurredWithGameCorrectionAndMedianBlur,
    50,
    12
)


minLineLength = 1
maxLineGap = 90

lines = cv2.HoughLinesP(
    cannyEdges,
    3, 
    np.pi/180.0,
    90,
    minLineLength, 
    maxLineGap
)

for x1, y1, x2,y2 in lines[0]:
    print(x1,y1,x2,y2)
    cv2.line(
        frameResized,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        4
    )


cv2.imshow('frame', frameResized);
cv2.waitKey();
cv2.destroyAllWindows();