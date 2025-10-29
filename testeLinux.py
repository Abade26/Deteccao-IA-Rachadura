import cv2
from ultralytics import YOLO
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
import time
import os

model_path = 'best.pt'
model = YOLO(model_path)

EMAIL_REMETENTE = "rodrigoabade26@gmail.com"
SENHA = "aqrg elck abec ycfk" 
EMAIL_DESTINATARIO = ""


DELAY_ALERTA = 300  
ultimo_alerta = 0

def enviar_email(imagem_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_REMETENTE
        msg['To'] = EMAIL_DESTINATARIO
        msg['Subject'] = "⚠️ Alerta: Rachadura detectada"

        body = "Uma rachadura com mais de 85% de confiança foi detectada."
        msg.attach(MIMEText(body, 'plain'))

        with open(imagem_path, "rb") as f:
            mime = MIMEBase('image', 'jpeg')
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            mime.add_header('Content-Disposition', f'attachment; filename="%s"' % os.path.basename(imagem_path))
            msg.attach(mime)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        print("📩 Email enviado com sucesso!")
    except Exception as e:
        print("Erro ao enviar email:", e)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, task='segment', verbose=False)
    result = results[0]

    if result.masks and result.boxes:
        for i, m in enumerate(result.masks.data):
            conf = result.boxes.conf[i].item()

            if conf >= 0.85:
                tempo_atual = time.time()
                if tempo_atual - ultimo_alerta > DELAY_ALERTA:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    img_name = f"alerta_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)

                    enviar_email(img_name)
                    ultimo_alerta = tempo_atual

            mask_array = m.cpu().numpy()
            mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
            mask_color = np.zeros_like(frame, dtype=np.uint8)
            mask_color[:, :, 0] = (mask_resized * 255).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
