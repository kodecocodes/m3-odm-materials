from ultralytics import YOLO
import os

model = YOLO("yolov8x-oiv7")
model.export(format="coreml", nms=True, int8=True)
os.rename("yolov8x-oiv7.mlpackage", "yolov8x-oiv7-int.mlpackage")
model.export(format="coreml", nms=True)

model = YOLO("yolov8n-oiv7")
model.export(format="coreml", nms=True)

model = YOLO("yolov8m-oiv7")
model.export(format="coreml", nms=True)
