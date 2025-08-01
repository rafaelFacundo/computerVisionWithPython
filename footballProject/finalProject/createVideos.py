import cv2
import os
import re

def ordenar_por_indice(nome_arquivo):
    # Extrai número do frame de 'teste<num>.jpg'
    match = re.match(r'teste(\d+)\.jpg', nome_arquivo)
    return int(match.group(1)) if match else float('inf')

def criar_video_da_pasta(caminho_pasta, saida_video, fps=10):
    arquivos = sorted([
        f for f in os.listdir(caminho_pasta)
        if f.lower().endswith('.jpg') and f.startswith('teste')
    ], key=ordenar_por_indice)

    if not arquivos:
        print(f"[!] Nenhum frame encontrado em {caminho_pasta}")
        return

    primeiro_frame = cv2.imread(os.path.join(caminho_pasta, arquivos[0]))
    altura, largura, _ = primeiro_frame.shape

    writer = cv2.VideoWriter(
        saida_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (largura, altura)
    )

    for nome_arquivo in arquivos:
        caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo)
        frame = cv2.imread(caminho_arquivo)
        if frame is not None:
            writer.write(frame)

    writer.release()
    print(f"✅ Vídeo criado: {saida_video}")

def processar_todas_as_pastas(caminho_raiz):
    for nome_pasta in sorted(os.listdir(caminho_raiz)):
        caminho_completo = os.path.join(caminho_raiz, nome_pasta)
        if os.path.isdir(caminho_completo):
            saida = os.path.join(caminho_raiz, f"{nome_pasta}.mp4")
            criar_video_da_pasta(caminho_completo, saida)

caminho_raiz = "/home/rafael/Documents/computerVisionWithPython/footballProject/finalProject/matchTests"
processar_todas_as_pastas(caminho_raiz)
