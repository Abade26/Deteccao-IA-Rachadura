from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText

# ===== Configurações =====
MODEL_PATH = "best.pt"
DELAY_ALERTA = 300
ultimo_alerta = 0

EMAIL_REMETENTE = "rodrigoabade26@gmail.com"
SENHA = "aqrg elck abec ycfk"  # senha de app do Gmail
EMAIL_DESTINATARIO = "rodrigoabade26@gmail.com"

# ===== Função de envio de e-mail =====
def enviar_email(imagem_path):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_REMETENTE
        msg["To"] = EMAIL_DESTINATARIO
        msg["Subject"] = "⚠️ Alerta: Rachadura detectada"
        msg.attach(MIMEText("Uma rachadura com mais de 85% de confiança foi detectada.", "plain"))

        with open(imagem_path, "rb") as f:
            mime = MIMEBase("image", "jpeg")
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            mime.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(imagem_path)}"')
            msg.attach(mime)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        print("📩 Email enviado com sucesso!")
    except Exception as e:
        print("❌ Erro ao enviar email:", e)

# ===== Carregar modelo =====
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo '{MODEL_PATH}' não encontrado!")
    exit(1)
model = YOLO(MODEL_PATH)

# ===== Inicializar câmera =====
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)
print("📸 Câmera iniciada com sucesso!")

# ===== Loop principal =====
cv2.namedWindow("Live", cv2.WINDOW_NORMAL)  # Janela única
while True:
    try:
        frame = picam2.capture_array()
        if frame is None:
            continue

        # YOLO
        results = model(frame, task="segment", verbose=False)
        result = results[0]

        # Aplicar máscaras e caixas
        if result.masks and result.boxes:
            for i, m in enumerate(result.masks.data):
                conf = result.boxes.conf[i].item()
                if conf >= 0.8:
                    agora = time.time()
                    if agora - ultimo_alerta > DELAY_ALERTA:
                        # Salvar imagem
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        img_name = f"alerta_{timestamp}.jpg"
                        cv2.imwrite(img_name, frame)
                        enviar_email(img_name)
                        ultimo_alerta = agora

                mask_array = m.cpu().numpy()
                mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
                mask_color = np.zeros_like(frame, dtype=np.uint8)
                mask_color[:, :, 0] = (mask_resized * 255).astype(np.uint8)
                frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"{conf*100:.1f}%", (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

        # Mostrar vídeo ao vivo
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("⚠️ Erro:", e)
        time.sleep(1)
        continue

# ===== Encerramento =====
picam2.stop()
cv2.destroyAllWindows()
print("✅ Encerrado.")
