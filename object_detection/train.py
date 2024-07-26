import torch
from ultralytics import YOLO
import os
path='../MasterMine.v7i.yolov8/'
yaml = os.path.join(path,'data.yaml')
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8l.pt') 
# Train the model
# results = model.train(data='data/images/train_yolo.yaml', epochs=100, imgsz=640,batch=32)
results = model.train(data=yaml, epochs=200,batch=64,imgsz=640,scale=0.4,degrees=360) # 
#train with augmentation

# results = model.train(data='xView_one_class/xView.yaml', epochs=100, imgsz=1180,batch=1,scale=0.1)
# results = model.train(data='xView/xView.yaml', epochs=500, imgsz=640,batch=2,lr0=0.02)
print(results)

# #https://docs.ultralytics.com/modes/predict/#inference-arguments
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# p=model.predict(Image.open('/home/gns/Pictures/mine_madudi.png'),conf=0.01,iou=0.02,augment=True,agnostic_nms=True,imgsz=400)
# plt.imshow(cv2.cvtColor(p[0].plot(),cv2.COLOR_BGR2RGB))

model = YOLO('object_detection/weights/best.pt')
results = model.val(data=yaml, epochs=200,batch=64,imgsz=640,scale=0.4,degrees=360) # 