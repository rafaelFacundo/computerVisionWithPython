from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

import matplotlib
matplotlib.use('Agg')  # Backend sem GUI

import matplotlib.pyplot as plt


model = YOLO("/home/rafael/Documents/computerVisionWithPython/footballProject/footbalPitchModel/best.pt")

ballModel = YOLO("/home/rafael/Documents/computerVisionWithPython/footballProject/yoloModels/best.pt")


testImage = cv2.imread("./1870.jpg")

ballYoloResults = ballModel.predict(testImage);

ballCenter = []
for r in ballYoloResults:
            for box in r.boxes:
                cls_id = int(box.cls)
                if r.names[cls_id] == "ball":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    ballCenter.append((x1+x2)/2)
                    ballCenter.append((y1 + y2)/2)
                    #cv2.rectangle(testImage, (x1,y1), (x2,y2), (0,255,0), 3)

yoloResults = model.predict(testImage)

# to do: try to rescale the frame to 640x640 squares

#testImage2 = cv2.resize(testImage, (640,640), interpolation=cv2.INTER_LINEAR)


""" results = model(testImage)[0]
# print(results.keypoints.conf)

print("RESULTS")

print(len(results.keypoints))


key_points = sv.KeyPoints.from_ultralytics(results)

#print(key_points)

keypointstodraw = []




for index, key_point in enumerate(key_points.xy[0]):
    if (key_points.confidence[0][index] > 0.5):
        keypointstodraw.append(key_point)

PitchConfig = SoccerPitchConfiguration()
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1,1,2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1,2).astype(np.float32)

print(PitchConfig.vertices)

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

pitch = draw_pitch(config=PitchConfig)
pitch = draw_points_on_pitch(
    config=PitchConfig,
    xy=ball_position_on_pitch,
    face_color=sv.Color.WHITE,
    edge_color=sv.Color.BLACK,
    radius=10,
    pitch=pitch
)
#sv.plot_image(pitch)

plt.imshow(pitch)
plt.axis('off')
plt.savefig("pitch_output.png", dpi=300, bbox_inches='tight') """

#print(frame_reference_keypoints)

""" for point in keypointstodraw:
    cv2.circle(testImage, (int(point[0]), int(point[1])), 10, (0,0,255), 1 )

    

cv2.imwrite("frame3.jpg", testImage) """

matchVideo = cv2.VideoCapture("/home/rafael/Documents/computerVisionWithPython/footballProject/videosToExtracFrames/teste.mp4")




""" 
ffmpeg -i /home/rafael/Documents/computerVisionWithPython/footballProject/videoplayback.mp4 -c:v libx264 -crf 23 -preset fast /home/rafael/Documents/computerVisionWithPython/footballProject/output.mp4

 """

