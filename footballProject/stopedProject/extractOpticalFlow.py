import os
import cv2
import numpy as np

# Configurações
BASE_DIR = "./jLeagueMatches/3925410/eventsFrames"
OUTPUT_DIR = "./jLeagueMatches/3925410/opticalFlowFeatures"
MAX_POINTS = 50
NUM_FRAMES = 128
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

os.makedirs(OUTPUT_DIR, exist_ok=True)

def processar_evento(event_path):
    frames = sorted(
        [f for f in os.listdir(event_path) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    frames = [cv2.resize(cv2.imread(os.path.join(event_path, f), cv2.IMREAD_GRAYSCALE), (320,320), interpolation=cv2.INTER_LINEAR) for f in frames]

    features = []
    prev_frame = frames[0]
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=MAX_POINTS, qualityLevel=0.3, minDistance=7)

    if prev_pts is None or len(prev_pts) < 5:
        return None

    for t in range(1, len(frames)):
        next_frame = frames[t]
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **LK_PARAMS)
        
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]
        #flow = good_new - good_old
        FIXED_POINT_COUNT = 50  
        flow = good_new - good_old  
        flow_flat = flow.flatten()

        expected_size = FIXED_POINT_COUNT * 2
        current_size = flow_flat.shape[0]

        if current_size < expected_size:
            padded_flow = np.pad(flow_flat, (0, expected_size - current_size))
        else:
            padded_flow = flow_flat[:expected_size]
        #features.append(flow.flatten())
        features.append(padded_flow)

        
        prev_frame = next_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
        

    return np.array(features)

# Loop pelos eventos
labels = []
paths = []

for classe in ["pass", "no_pass"]:
    entrada = os.path.join(BASE_DIR, classe)
    saida = os.path.join(OUTPUT_DIR, classe)
    os.makedirs(saida, exist_ok=True)

    for evento in os.listdir(entrada):
        caminho = os.path.join(entrada, evento)
        if not os.path.isdir(caminho):
            continue

        print(f"Processando: {classe}/{evento}")
        passOpticalFlowList = os.listdir(f"./jLeagueMatches/3925410/opticalFlowFeatures/pass")
        noPassOpticalFlowList = os.listdir(f"./jLeagueMatches/3925410/opticalFlowFeatures/no_pass")
        
        if f"{evento}.npy" not in passOpticalFlowList and f"{evento}.npy" not in noPassOpticalFlowList:
            vetor = processar_evento(caminho)
            if vetor is not None:
                np.save(os.path.join(saida, f"{evento}.npy"), vetor)
                paths.append(os.path.join(saida, f"{evento}.npy"))
                labels.append(1 if classe == "pass" else 0)
        else:
            print("--- already saved")

# Salva lista de caminhos e labels
np.save(os.path.join(OUTPUT_DIR, "data_paths.npy"), np.array(paths))
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), np.array(labels))

print("✅ Finalizado! Vetores salvos como .npy")