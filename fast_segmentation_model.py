# segmentation_model.py
import os
from ultralytics import FastSAM
class FastSegmentationModel:
    def __init__(self, model_name="FastSAM-s.pt"):
        self.model = FastSAM(model_name)  # or FastSAM-x.pt

    def segment_image(self, image_path, text):
        results = self.model.predict(source=image_path, texts=text)
        return results
