from ultralytics import YOLO

def main():
    model = YOLO("runs/segment/train7/weights/last.pt")
    model.train(resume=True)

if __name__ == "__main__":
    main()
