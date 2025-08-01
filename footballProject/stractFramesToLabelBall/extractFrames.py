import cv2
import tensorflow as tf

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


filterModel =  tf.keras.models.load_model(
    "/home/rafael/Documents/computerVisionWithPython/footballProject/stopedProject/filterModel.keras"
)

videosPath = [
    "/home/rafael/Documents/computerVisionWithPython/footballProject/output.mp4",
]

framesOutputPath = "/home/rafael/Documents/computerVisionWithPython/footballProject/framesToLabelBall"
frameIndex2 = 30080


frameIndex = 1678
for video in videosPath:
    videoCapture = cv2.VideoCapture(video)
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frameIndex2)
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayImageResized = cv2.resize(
            gray, 
            (320, 320),
            interpolation=cv2.INTER_LINEAR
        )
        cannyBlurImage = cannyWithBlurAndGamma(grayImageResized)

        cannyBlurImage = np.expand_dims(cannyBlurImage, axis=(0, -1))

        pred = filterModel.predict(cannyBlurImage)

        if pred > 0.5:
            cv2.imwrite(f"{framesOutputPath}/{frameIndex}.jpg", frame)

        frameIndex += 1