# Object Recognition in Assistive Home Robotics for Elderly Care

**Final Year Project — CMP6200 | Birmingham City University | 2025/26**  
**Student:** Samuel Adegbusi (22225976)  
**Supervisor:** Mohamed Ihmeida

---

## Overview

This project develops a real-time object detection system using YOLOv8n, trained from scratch on a custom dataset of everyday household objects relevant to elderly care. The system is designed to support assistive home robots in helping elderly individuals locate misplaced items, addressing challenges caused by age-related cognitive decline.

The trained model is to be exported to ONNX format for deployment on a Raspberry Pi.

---

## Dataset

- **Source:** [ElderlyObjects on Roboflow](https://universe.roboflow.com/object-detection1-wvtyn/elderlyobjects/dataset/1)
- **Total images:** 4,916 (fully annotated)
- **Classes:** 7 — Glasses, TV Remote, Wheelchair, Keys, Wrist Watch, Mug, Mobile Phone
- **Split:** 73% train / 18% validation / 10% test
- **Annotation tool:** Roboflow

> The raw image dataset is hosted on Roboflow. Use the link above to access and export it.

---

## Model

- **Architecture:** YOLOv8n (`yolov8n.yaml`)
- **Pretrained weights:** None — trained from scratch
- **Image size:** 640×640
- **Epochs:** 200
- **Patience:** 50
- **Hardware used:** NVIDIA RTX 4060
- **Best weights:** `run3-best.pt` (included in this repository)

---

## Results

Three training runs were conducted, with iterative improvements between each:

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Epochs completed | 74 (early stopped) | 200 | 200 |
| Early stop patience | 25 | 50 | 50 |
| Best mAP50 | 0.927 | **0.952** | 0.946 |
| Best Precision | 0.923 | **0.949** | 0.941 |
| Best Recall | 0.894 | **0.907** | 0.901 |

**Run 2 is the best performing model**, achieving mAP50 of 0.952 at epoch 155.

Run 1 was stopped prematurely due to `patience=25` — metrics were still improving at the point of termination. Runs 2 and 3 used `patience=50` and `epochs=200` to allow full convergence.

---

## Repository Structure

```
fyp-object-recognition/
├── train.py                  # Training script
├── best.pt                   # Best model weights (Run 2)
├── data.yaml                 # Dataset configuration
├── run1results.csv           # Training metrics — Run 1
├── run2results.csv           # Training metrics — Run 2
├── run3results.csv           # Training metrics — Run 3
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install ultralytics
```

### Training

```bash
python train.py
```

Or directly:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # from scratch
model.train(
    data='data.yaml',
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
    mosaic=1.0      
)
```

### Inference with best weights

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(source='your_image.jpg', conf=0.5)
results[0].show()
```

### Export to ONNX (for Raspberry Pi deployment)

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx')
```

---

## References

Rabin, L.A. et al. (2015). Subjective cognitive decline in older adults: An overview of self-report measures used across 19 international research studies. *Journal of Alzheimer's Disease*, 48(Suppl 1), S63–S86.

Jocher, G. et al. (2023). *Ultralytics YOLOv8*. Available at: https://github.com/ultralytics/ultralytics

---

*Birmingham City University — CMP6200 Individual Honours Project*
