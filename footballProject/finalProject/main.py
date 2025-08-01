from ultralytics import YOLO
import cv2
import tensorflow as tf
import numpy as np
import filtersFunctions
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
import os


#used models
filterModel = tf.keras.models.load_model("/home/rafael/Documents/computerVisionWithPython/footballProject/finalProject/filterModel.keras");
model = YOLO("/home/rafael/Documents/computerVisionWithPython/footballProject/finalProject/yoloModels/best.pt")
pitchModel = YOLO("/home/rafael/Documents/computerVisionWithPython/footballProject/finalProject/footbalPitchModel/best.pt")

#video to test
matchVideo = cv2.VideoCapture("/home/rafael/Documents/computerVisionWithPython/footballProject/finalProject/videos/output.mp4")

#frame index to start the video
#using this make tests
frameIndex2 = 70080
matchVideo.set(cv2.CAP_PROP_POS_FRAMES, frameIndex2)

#file with the ball positions
ballPositionFile = open("ballPositions.txt", "a")

#number of the test
testNumber = 17

os.makedirs(f"./matchTests/test{testNumber}", exist_ok=True)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1,1,2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1,2).astype(np.float32)

PitchConfig = SoccerPitchConfiguration()

ballCenter = []

indexss = 0

def increaseContrast(image):
    alpha = 2
    beta = -100
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image



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

    if pred > 0.5:
        yoloResults = model.predict(frame);

        for r in yoloResults:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf[0])
                if r.names[cls_id] == "ball"  and conf >= 0.5:
                    ballCenter = []
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                    ballCenter.append((x1+x2)/2)
                    ballCenter.append((y1 + y2)/2)

                    frame_with_increased_contrast = increaseContrast(frame)

                    results = pitchModel(frame_with_increased_contrast)[0]

                    #print(len(results.keypoints))
                    

                    key_points = sv.KeyPoints.from_ultralytics(results)

                    keypointstodraw = []

                    if len(key_points.xy) > 0:

                        for index, key_point in enumerate(key_points.xy[0]):
                            if (key_points.confidence[0][index] > 0.6):
                                keypointstodraw.append(key_point)
                        
                        for point in keypointstodraw:
                            print(int(point[0]), int(point[1]))
                            #cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0,0,255), 1 )
                        
                        cv2.imwrite(f"./matchTests/test{testNumber}/teste{indexss}.jpg", frame)
                        indexss += 1

                        filter_ = key_points.confidence[0] > 0.5
                        frame_reference_points = key_points.xy[0][filter_]
                        frame_reference_keypoints = sv.KeyPoints(xy=frame_reference_points[np.newaxis,...])
                        pitch_reference_points = np.array(PitchConfig.vertices)[filter_]

                        

                        viewtransformer = ViewTransformer(
                            source=frame_reference_points,
                            target=pitch_reference_points
                        )

                        ball_position_on_pitch = viewtransformer.transform_points(
                            np.array(ballCenter)
                        )

                        string = ""

                        for position in ball_position_on_pitch:
                            print(position)
                            string += f"{position }"

                        ballPositionFile.write(f"{string}\n")
                

ballPositionFile.close() 

matchVideo.release()
cv2.destroyAllWindows()







