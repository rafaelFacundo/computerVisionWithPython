import tensorflow as tf
import cv2
import numpy as np
import filtersFunctions

filterModel = tf.keras.models.load_model("/home/rafael/Documents/computerVisionWithPython/footballProject/stopedProject/filterModel.keras");

pathToVideo = "/home/rafael/Documents/computerVisionWithPython/footballProject/videosToExtracFrames/output.mp4";

matchVideo = cv2.VideoCapture(pathToVideo);
while matchVideo.isOpened():
    ret, frame = matchVideo.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayImageResized = cv2.resize(
        gray, 
        (320, 320),
        interpolation=cv2.INTER_LINEAR
    )
    cannyBlurImage = filtersFunctions.cannyWithBlurAndGamma(grayImageResized)

    cannyBlurImage = np.expand_dims(cannyBlurImage, axis=(0, -1))

    pred = filterModel.predict(cannyBlurImage)

    RESULT = "YES" if pred > 0.5 else "NO" 

    print(f"RESULT => {pred}")

    frame = cv2.putText(
        frame,
        f"{RESULT}",
        (40,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (0,0,255),
        5
    )

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

matchVideo.release()
cv2.destroyAllWindows()
