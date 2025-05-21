
from ultralytics import YOLO
import os

cwd = os.path.dirname(os.path.abspath(__file__))

yolo_weights = os.path.join(cwd, r"yolo\train\weights\best.pt")
yolo = YOLO(yolo_weights, verbose=False)

