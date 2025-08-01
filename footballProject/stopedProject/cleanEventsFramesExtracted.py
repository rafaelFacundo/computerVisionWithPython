import tensorflow as tf
import cv2
import numpy as np
import filtersFunctions
import os
import shutil

filterModel = tf.keras.models.load_model("filterModel.keras");

folderPath = "./jLeagueMatches/3925410/eventsFrames"

for folder in os.listdir(folderPath):
    print(folder)
    for eventFolder in os.listdir(f"{folderPath}/{folder}"):
        RESULT = 1;
        print(f"-- testing for {eventFolder}")
        for frame in os.listdir(f"{folderPath}/{folder}/{eventFolder}"):
            imageFrame = cv2.imread(
                f"{folderPath}/{folder}/{eventFolder}/{frame}",
                cv2.COLOR_BGR2GRAY
            )
            grayImageResized = cv2.resize(
                imageFrame, 
                (320, 320),
                interpolation=cv2.INTER_LINEAR
            )
            cannyBlurImage = filtersFunctions.cannyWithBlurAndGamma(grayImageResized)

            cannyBlurImage = np.expand_dims(cannyBlurImage, axis=(0, -1))

            pred = filterModel.predict(cannyBlurImage)

            if pred > 0.5:
                print('-- ALL GOOd')
                RESULT = 1  
            else:
                print(f"-- Found a bad frame in event {eventFolder} in frame {frame}")
                RESULT = 0 
                break
        if RESULT == 0:
            shutil.rmtree(f"{folderPath}/{folder}/{eventFolder}")
            print(f"--- {eventFolder} deleted")