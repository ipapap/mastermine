from ultralytics import YOLO
from PIL import Image

model = YOLO('object_detection/weights/best.pt')
def detect_objects(image,model=model):

    results = model.predict(image, conf=0.3, iou=0.02, augment=True, agnostic_nms=True, imgsz=400)
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

    return detected_info