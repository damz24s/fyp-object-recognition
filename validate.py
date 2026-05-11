from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/runs3/train/fyp_v3/weights/best.pt")
    model.val(data="C:/Users/jack/Documents/FYP/customDataset-v1/data.yaml")