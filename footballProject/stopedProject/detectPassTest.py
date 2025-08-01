import cv2
import numpy as np
from tensorflow.keras.models import load_model
import filtersFunctions


# Carrega modelos
filtro_model = load_model("./filterModel.keras")
passe_model = load_model("./modelo_pass_lstm.keras")

passFile = open("pass.txt", "a")

# Abrir vídeo
video_path = "./videoTest/teste.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_window = []
frame_indices = []
frame_index = 0

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (320, 320))
    return resized

def calcular_optical_flow(frames):
    flows = []
    prev_frame = frames[0]
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=50, qualityLevel=0.3, minDistance=7)
    if prev_pts is None:
        return None
    for t in range(1, len(frames)):
        next_frame = frames[t]
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None)
        if next_pts is None:
            return None
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]
        flow = (good_new - good_old).flatten()
        expected = 50 * 2
        if flow.shape[0] < expected:
            flow = np.pad(flow, (0, expected - flow.shape[0]))
        else:
            flow = flow[:expected]
        flows.append(flow)
        prev_frame = next_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
    return np.array(flows)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed = preprocess_frame(frame)
    cannyFrame = filtersFunctions.cannyWithBlurAndGamma(preprocessed)
    cannyBlurImage = np.expand_dims(cannyFrame, axis=(0, -1))
    
    prediction = filtro_model.predict(cannyBlurImage)


    if prediction > 0.5:

        frame_window.append(preprocessed)
        frame_indices.append(frame_index)

        if len(frame_window) > 128:
            frame_window.pop(0)
            frame_indices.pop(0)

        if len(frame_window) == 128:
            flow = calcular_optical_flow(frame_window)
            if flow is not None and flow.shape == (127, 100):
                input_seq = flow.reshape(1, 127, 100)
                passe_pred = passe_model.predict(input_seq)
                
                if passe_pred > 0.5:
                    tempo = frame_indices[-1] / fps
                    minuto = int(tempo // 60)
                    segundo = int(tempo % 60)
                    print("LKAJDLAKSJDLAKSJDlas")
                    passFile.write(f"Passe detectado em {minuto:02d}:{segundo:02d} (frame {frame_indices[-1]})\n")

                    for frame in frame_window:
                        cv2.imshow("frame", frame)
                        cv2.waitKey()
                    cv2.destroyAllWindows()
                    
                    for _ in range(128):
                        cap.read()
                        frame_index += 1
                    
                    frame_window = []
                    frame_indices = []

    frame_index += 1

cap.release()
