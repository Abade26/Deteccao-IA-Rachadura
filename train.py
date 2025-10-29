from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data="data.yaml",
        epochs=60,
        batch=24,
        imgsz=640,
        device=0
    )

if __name__ == "__main__":   # <- ESSENCIAL NO WINDOWS
    main()
