from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")

    model.train(
        data="C:/Users/jack/Documents/FYP/customDataset-v1/data.yaml",
        epochs=200,
        imgsz=640,
        patience=75,
        batch=16,
        workers=2,
        device=0,
        pretrained=False,
        hsv_h=0.015,      
        hsv_s=0.7,         
        hsv_v=0.4,        
        degrees=10,       
        translate=0.1,    
        flipud=0.1,       
        mosaic=1.0,       
        project="runs3/train",
        name="fyp_v3"
    )