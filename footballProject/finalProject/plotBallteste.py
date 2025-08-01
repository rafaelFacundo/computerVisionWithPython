import numpy as np
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PitchConfig = SoccerPitchConfiguration()

# 1. Lista para armazenar todas as posições válidas da bola
ball_positions = []

# 2. Lê as posições da bola do arquivo
with open("ballPositions.txt", "r") as ballFile:
    for line in ballFile:
        try:
            coords = line.strip().replace("[", "").replace("]", "").split()
            if len(coords) >= 2:
                x, y = map(float, coords)
                ball_positions.append([x, y])
        except:
            continue  # ignora linhas inválidas como "-1"

# 3. Desenha o campo
pitch = draw_pitch(config=PitchConfig)

# 4. Desenha todos os pontos em uma única imagem
pitch = draw_points_on_pitch(
    config=PitchConfig,
    xy=np.array(ball_positions),
    face_color=sv.Color.WHITE,
    edge_color=sv.Color.BLACK,
    radius=10,
    pitch=pitch
)

# 5. Salva a imagem final
plt.imshow(pitch)
plt.axis('off')
plt.savefig("final_pitch_ball_positions.png", dpi=300, bbox_inches='tight')
plt.close()


