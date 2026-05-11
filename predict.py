from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/runs3/train/fyp_v3/weights/best.pt")
    
    model.predict(
        source="C:/Users/jack/Documents/FYP/real_test10.jpg",
        save=True,
        conf=0.1,

    )

        #project="runs3/real_images",
        #name="fyp_v3.3_test"