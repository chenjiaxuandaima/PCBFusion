# 测试
from ultralytics import YOLO 
model = YOLO('/home/user/TwoStream_Yolov8-main/runs/train/yolov8sepoch300batch16nopreCrossattenlastmiloss4/weights/best.pt') 
# metrics = model.val(data='/home/user/TwoStream_Yolov8-main/data/pcb.yaml',split='test',imgsz=512,batch=16)
metrics = model.val(data='/home/user/TwoStream_Yolov8-main/data/pcb.yaml',workers=0,imgsz=512,batch=16,
                    name='yolov8sepoch300batch16nopreCrossattenlastmiloss4'
                    )

