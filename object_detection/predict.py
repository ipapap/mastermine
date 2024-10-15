import torch
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from PIL import Image

# path = '.'
# yaml = os.path.join(path, 'data.yaml')

# model = YOLO('object_detection/weights/best.pt')

# image_path = 'samples/queries/DJI_20240703154815_0007_W.JPG'

# results = model.predict(Image.open(image_path), conf=0.01, iou=0.02, augment=True, agnostic_nms=True, imgsz=400)

# detected_info = []

# for result in results[0].boxes:
#     box = result.xyxy[0].cpu().numpy() 
#     class_id = int(result.cls[0].cpu().numpy())  # Get the class ID
#     class_name = model.names[class_id]  

#     x_center = (box[0] + box[2]) / 2
#     y_center = (box[1] + box[3]) / 2

#     detected_info.append({
#         'class': class_name,
#         'center': (x_center, y_center)
#     })

# results[0].plot()  

# plt.imshow(results[0].plot())
# plt.axis('off')  
# plt.show()

# print("Detected objects information:")
# for info in detected_info:
#     print(f"Class: {info['class']}, Center: {info['center']}")

def detect_objects(image_path, model_path):
    model = YOLO(model_path)

    results = model.predict(Image.open(image_path), conf=0.01, iou=0.02, augment=True, agnostic_nms=True, imgsz=400)

    detected_info = []

    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        class_id = int(result.cls[0].cpu().numpy())
        class_name = model.names[class_id]

        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2

        detected_info.append({
            'class': class_name,
            'center': (x_center, y_center)
        })

    # plt.imshow(results[0].plot())
    # plt.axis('off')
    # plt.show()

    return detected_info