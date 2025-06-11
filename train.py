#训练
from ultralytics import YOLO
import ultralytics.nn.tasks
model = YOLO('/home/user/TwoStream_Yolov8-main/yaml/Crossattenyolov8s.yaml')
# model.load('yolov8s.pt')
results = model.train(data='/home/user/TwoStream_Yolov8-main/data/pcb.yaml',
                imgsz=512,
                epochs=300,
                batch=16,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=True,
                project='runs/train',
                name='yolov8sepoch300batch16',
                single_cls=False,
                cache=False,
                )
