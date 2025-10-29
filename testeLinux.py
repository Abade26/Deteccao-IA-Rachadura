import cv2
from ultralytics import YOLO
import numpy as np

# Caminho do modelo
model_path = 'best.pt'  # coloque o modelo no mesmo diretório do script
model = YOLO(model_path)

# Abre a webcam (0 = webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a inferência com segmentação
    results = model(frame, task='segment', verbose=False)
    result = results[0]

    # Se houver máscaras e boxes, desenha sobre o frame
    if result.masks and result.boxes:
        for i, m in enumerate(result.masks.data):
            mask_array = m.cpu().numpy()  # shape original
            # Redimensiona máscara para o tamanho do frame
            mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
            mask_color = np.zeros_like(frame, dtype=np.uint8)
            mask_color[:, :, 0] = (mask_resized * 255).astype(np.uint8)  # azul

            # Combina máscara com o frame
            frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

            # Confiança da box
            conf = result.boxes.conf[i].item()
            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cv2.putText(frame,
                        f"{conf*100:.1f}%",
                        (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

    cv2.imshow("Detecção de Rachaduras - Pressione 'q' para sair", frame)

    # Sai com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
