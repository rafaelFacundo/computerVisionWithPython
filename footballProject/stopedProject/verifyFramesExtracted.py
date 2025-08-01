""" import os

folderPath = "./jLeagueMatches/3925226/eventsFrames"
passFolder = f"{folderPath}/pass"
noPassFolder = f"{folderPath}/no_pass"

passEvents = os.listdir(passFolder)
noPassEvents = os.listdir(noPassFolder)

numberOfNoPass = len(noPassEvents);
numberOfPass = len(passEvents);

print(f"Number of no pass = {numberOfNoPass}")
print(f"Number of pass    = {numberOfPass}")

totalFrames = 0;
minFrames = 0;
maxFrames = 0;
meanFrames = 0;

listToRead = noPassEvents
folterToRead = noPassFolder

for index, event in enumerate(listToRead):
    totalFrames += len(os.listdir(f"{folterToRead}/{event}"))
    if index == 0:
        minFrames = len(os.listdir(f"{folterToRead}/{event}"))
        maxFrames = len(os.listdir(f"{folterToRead}/{event}"))
    else:
        if len(os.listdir(f"{folterToRead}/{event}")) < minFrames:
            minFrames = len(os.listdir(f"{folterToRead}/{event}"))
        elif len(os.listdir(f"{folterToRead}/{event}")) > maxFrames:
            maxFrames = len(os.listdir(f"{folterToRead}/{event}"))
else:
    meanFrames = totalFrames/numberOfPass

print(f"Total frames {totalFrames}")
print(f"Min frames {minFrames}")
print(f"Max frames {maxFrames}")
print(f"Mean frames {meanFrames}")
 """

import os
import cv2
import shutil

MAX_FRAMES = 128
HEAD = 30
TAIL = 30

def ajustar_frames_em_pasta(event_dir):
    frame_files = sorted(
        [f for f in os.listdir(event_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    total = len(frame_files)
    if total == 0:
        return

    # Carrega as imagens em memória
    frames = [cv2.imread(os.path.join(event_dir, f)) for f in frame_files]

    if total == MAX_FRAMES:
        return  # já está ok

    elif total < MAX_FRAMES:
        # Completa o meio com repetição
        meio = frames[HEAD:-TAIL] if total > HEAD + TAIL else frames[HEAD:]
        restante = MAX_FRAMES - HEAD - TAIL
        if not meio:
            meio = frames  # fallback para eventos curtos
        passo = len(meio) / restante
        meio_expandido = [meio[int(i * passo)] for i in range(restante)]

        nova_seq = frames[:HEAD] + meio_expandido + frames[-TAIL:]

    else:
        # Reduz o meio
        meio = frames[HEAD:total - TAIL]
        restante = MAX_FRAMES - HEAD - TAIL
        passo = len(meio) / restante
        meio_reduzido = [meio[int(i * passo)] for i in range(restante)]

        nova_seq = frames[:HEAD] + meio_reduzido + frames[-TAIL:]

    # Limpa pasta e salva nova sequência
    for f in os.listdir(event_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(event_dir, f))

    for idx, frame in enumerate(nova_seq):
        cv2.imwrite(os.path.join(event_dir, f"{idx}.jpg"), frame)

    print(f"✅ {event_dir} ajustado para 128 frames.")

def processar_todas_as_pastas(base_path):
    for classe in ["pass", "no_pass"]:
        classe_path = os.path.join(base_path, classe)
        if not os.path.isdir(classe_path):
            continue
        for evento in os.listdir(classe_path):
            evento_path = os.path.join(classe_path, evento)
            if os.path.isdir(evento_path):
                ajustar_frames_em_pasta(evento_path)

# Caminho base
base_dir = "./jLeagueMatches/3925226/eventFrames2"
processar_todas_as_pastas(base_dir)

