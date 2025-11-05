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

# ----------------------
# Configurações do YOLO
# ----------------------
model_path = 'runs/segment/train7/weights/best.pt'
model = YOLO(model_path)

# ----------------------
# Configurações do e-mail
# ----------------------
EMAIL_REMETENTE = "alerta@walleye.com.br"
SENHA = "n#7CEAFdc@"  # pode usar variável de ambiente para segurança
EMAIL_DESTINATARIO = "rodrigoabade26@gmail.com"


#EMAIL_REMETENTE = "rodrigoabade26@gmail.com"
#SENHA = "aqrg elck abec ycfk" 
#EMAIL_DESTINATARIO = ""

SMTP_HOST = "smtp.hostinger.com"
SMTP_PORT = 587  # TLS

# ----------------------
# Configurações de alerta
# ----------------------
DELAY_ALERTA = 300  # 5 minutos
ultimo_alerta = 0

# ----------------------
# Função para enviar e-mail com anexo
# ----------------------
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
            mime.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(imagem_path)}"')
            msg.attach(mime)

        # Conexão com Hostinger SMTP
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        print("📩 Email enviado com sucesso!")

    except Exception as e:
        print("Erro ao enviar email:", e)

# ----------------------
# Configurações da câmera
# ----------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

FRAME_INTERVAL = 3  # Processa 1 a cada 3 frames para melhorar FPS
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_INTERVAL != 0:
        continue  # Pula frames para otimizar FPS

    # Inferência YOLO
    results = model(frame, task='segment', verbose=False)
    result = results[0]

    if result.masks and result.boxes:
        for i, m in enumerate(result.masks.data):
            conf = result.boxes.conf[i].item()

            # 🚨 Confiança > 80%
            if conf >= 0.8:
                tempo_atual = time.time()
                if tempo_atual - ultimo_alerta > DELAY_ALERTA:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    img_name = f"alerta_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)

                    enviar_email(img_name)
                    ultimo_alerta = tempo_atual
                    break  # evita múltiplos envios simultâneos

            # Máscara vermelha sobre o frame
            mask_array = m.cpu().numpy()
            mask_color = np.zeros_like(frame, dtype=np.uint8)
            mask_color[:, :, 2] = (mask_array * 255).astype(np.uint8)  # vermelho
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
