import cv2
import json

class QRReader:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.detector = cv2.QRCodeDetector()

    def read_once(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Erro ao acessar a câmera")
            return None

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            data, points, _ = self.detector.detectAndDecode(frame)
            if data:
                cap.release()
                cv2.destroyAllWindows()
                return data

            cv2.imshow("Aponte o QR Code...", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return None

    @staticmethod
    def parse_payload(raw):
        try:
            obj = json.loads(raw)
            if not all(k in obj for k in ("id", "codigo_pareador", "api_url")):
                return None
            return {
                "id": obj["id"],
                "codigo_pareador": obj["codigo_pareador"],
                "api_url": obj["api_url"].rstrip("/")
            }
        except:
            print("QR não é JSON válido")
            return None
