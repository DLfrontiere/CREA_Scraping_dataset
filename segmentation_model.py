# segmentation_model.py
import os
from ultralytics import YOLO

class SegmentationModel:
    def __init__(self, model_name='yolov8n-seg.pt'):
        self.model = YOLO(model_name)  # Load a pre-trained segmentation model

    def segment_image(self, image_path, save_dir=None):
        results = self.model.predict(source=image_path, save=True, save_txt=True, save_masks=True, save_dir=save_dir)
        return results


