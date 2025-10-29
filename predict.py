import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
                                                                                                                                                                                                                         
model_path = 'runs/segment/train7/weights/best.pt'

model = YOLO(model_path)

img_path = 'C:/Users/Abade/Documents/IA-Treinada/361_teste.jpg'

results = model(img_path, task='segment')

result = results[0]

result.show()

result.save()
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if result.masks and result.boxes:
    for i, m in enumerate(result.masks.data):
        mask_array = m.cpu().numpy()
        conf = result.boxes.conf[i].item()  

        plt.figure()
        plt.imshow(img_rgb)
        plt.imshow(mask_array, cmap='Blues', alpha=0.4)  
        plt.title(f"Rachadura {i+1} – Acuracia: {conf*100:.1f}%")
        plt.axis('off')
        plt.show()
else:
    print("Nenhuma detecção encontrada.")
