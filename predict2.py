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

# Caminho do modelo
model_path = 'best.pt'
model = YOLO(model_path)

# Configurações do e-mail
EMAIL_REMETENTE = "rodrigoabade26@gmail.com"
SENHA = "aqrg elck abec ycfk"  # use senha de app, não a senha normal
EMAIL_DESTINATARIO = "profcelsobarreto@hotmail.com"

# Delay entre alertas (em segundos)
DELAY_ALERTA = 300  # 5 minutos
ultimo_alerta = 0

# Função para enviar e-mail com anexo
def enviar_email(imagem_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_REMETENTE
        msg['To'] = EMAIL_DESTINATARIO
        msg['Subject'] = "⚠️ Alerta: Rachadura detectada"

        # Corpo do e-mail
        body = "Uma rachadura com mais de 80% de confiança foi detectada."
        msg.attach(MIMEText(body, 'plain'))

        # Anexo
        with open(imagem_path, "rb") as f:
            mime = MIMEBase('image', 'jpeg')
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            mime.add_header('Content-Disposition', f'attachment; filename="%s"' % os.path.basename(imagem_path))
            msg.attach(mime)

        # Conexão com servidor SMTP (exemplo: Gmail)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        print("📩 Email enviado com sucesso!")

    except Exception as e:
        print("Erro ao enviar email:", e)

# Abre a webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a inferência
    results = model(frame, task='segment', verbose=False)
    result = results[0]

    if result.masks and result.boxes:
        for i, m in enumerate(result.masks.data):
            conf = result.boxes.conf[i].item()

            # 🚨 Verifica se confiança > 80%
            if conf >= 0.8:
                tempo_atual = time.time()
                if tempo_atual - ultimo_alerta > DELAY_ALERTA:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    img_name = f"alerta_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)

                    enviar_email(img_name)
                    ultimo_alerta = tempo_atual

            # Máscara azul
            mask_array = m.cpu().numpy()
            mask_color = np.zeros_like(frame, dtype=np.uint8)
            mask_color[:, :, 0] = (mask_array * 255).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

            # Box + confiança
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
