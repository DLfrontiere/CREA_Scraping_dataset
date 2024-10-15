# segmentation_model.py

from mobile_sam import sam_model_registry,SamAutomaticMaskGenerator
import torch
import os
from ultralytics import YOLO
from ultralytics import SAM
#from ultralytics import FastSAM
from PIL import Image
from numpy import asarray

class YoloSegmentationModel:
    def __init__(self, model_name='yolov8n-seg.pt'):
        self.model = YOLO(model_name)  # Load a pre-trained segmentation model

    def segment_image(self, image_path, text):
        results = self.model.predict(source=image_path, texts=text)
        return results

class SAMSegmentationModel:
    def __init__(self, model_name='mobile_sam.pt'):
        #self.model = SAM(model_name)  # Load a pre-trained segmentation model
        self.model_type = "vit_t"
        self.sam_checkpoint = f"./{model_name}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mobile_sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.mobile_sam.to(device=self.device)
        self.mobile_sam.eval()
        #self.predictor = SamPredictor(self.mobile_sam)


    def seg_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        data = asarray(image)
        mask_generator = SamAutomaticMaskGenerator(self.mobile_sam)
        masks = mask_generator.generate(data)
        return masks
        #results = self.model.predict(source=image_path, texts=text)
        #return results
        
class FastSegmentationModel:
    def __init__(self, model_name="FastSAM-s.pt"):
        self.model = FastSAM(model_name)  # or FastSAM-x.pt

    def seg_image(self, image_path, text):
        results = self.model.predict(source=image_path, texts=text)
        return results
        
from autodistill_fastsam import FastSAM
from autodistill.detection import CaptionOntology
        
class GroundedFastSAM:

    def __init__(self):
        self.ontology_dict = {}

    def add_element(self,element):

        self.ontology_dict[f"{element}"]=f"{element}"
        self.ontology = CaptionOntology(self.ontology_dict)
        self.model = FastSAM(self.ontology)
        
    def seg_image(self,image):
        return self.model.predict(image)

"""
segmentation_model = GroundedFastSAM()
segmentation_model.add_element("apple")
image_path = "/home/domenico/Desktop/Projects/CREA/CreaFoodScraping/dataset/images/MELE,_FRESCHE,_SENZA_BUCCIA/000002.jpg"
detections = segmentation_model.seg_image(image_path)
print(detections)
"""
