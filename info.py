# 查看模型信息

from ultralytics import YOLO
model = YOLO('/home/user/TwoStream_Yolov8-main/yaml/Crossattenyolov8s.yaml')
model.info()