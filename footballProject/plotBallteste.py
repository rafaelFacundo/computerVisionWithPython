""" import numpy as np
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PitchConfig = SoccerPitchConfiguration()

ballFile = open("ballPositions.txt", "r")

i = 0
for line in ballFile.readlines():
    ball_position_on_pitch = []
    for x in line.rstrip('\n').split(","):
        ball_position_on_pitch.append(round(float(x)))
    pitch = draw_pitch(config=PitchConfig)
    pitch = draw_points_on_pitch(
        config=PitchConfig,
        xy=np.array([ball_position_on_pitch]),
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=pitch
    )
    #sv.plot_image(pitch)

    plt.imshow(pitch)
    plt.axis('off')
    plt.savefig(f"pitch_output-{i}.png", dpi=300, bbox_inches='tight')
    i += 1 """




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







""" 
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# 1. Carrega os dados
positions = []
with open("ballPositions.txt", "r") as file:
    for line in file:
        if "-1" in line:
            continue
        coords = line.strip().replace("[", "").replace("]", "").split()
        if len(coords) >= 2:
            x, y = map(float, coords)
            positions.append((x, y))

positions = np.array(positions)

# 2. Separa x e y
x = positions[:, 0]
y = positions[:, 1]

# 3. Normaliza para StatsBomb pitch
# (Você pode ajustar os divisores se o campo tiver outro tamanho)
x_normalized = x / 7000 * 120
y_normalized = y / 2000 * 80

# 4. Cria o campo
pitch = Pitch(pitch_type='statsbomb', pitch_color='#2c2c2c', line_color='white')
fig, ax = pitch.draw(figsize=(12, 8))

# 5. Plota o mapa de calor
bin_statistic = pitch.bin_statistic(x_normalized, y_normalized, statistic='count', bins=(50, 50))
pitch.heatmap(bin_statistic, ax=ax, cmap='jet', edgecolors='none')

# 6. Título e exibição
plt.title("Mapa de Calor da Bola", color='white', fontsize=18)
plt.show() """


