# main.py
import os
import cv2
import numpy as np
from food_database import FoodDatabase
from image_crawler import ImageCrawler
from segmentation_model import SegmentationModel
from fast_segmentation_model import FastSegmentationModel
import glob
from sklearn.model_selection import train_test_split
import shutil
import yaml

def main():
    # Initialize the database
    database_path = os.path.join(os.getcwd(), 'nutritional_data.db')
    database = FoodDatabase(database_path=database_path)
    food_names = database.fetch_food_names()
    
    # For demonstration, limit the number of food items
    food_names = food_names[:5]  # Fetch first 5 food items
    
    # Initialize the image crawler
    image_crawler = ImageCrawler(storage_root='food_images')
    
    # Initialize the segmentation model
    segmentation_model = FastSegmentationModel()  # Replace with your actual segmentation model
    
    # Prepare directories for dataset
    dataset_dir = 'dataset'
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    for dir_path in [images_dir, labels_dir]:
        for split in ['train', 'val']:
            split_dir = os.path.join(dir_path, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)

    # Iterate over food names
    for food_name in food_names:
        print(f"Processing '{food_name}'...")
        # Crawl images for this food
        image_dir = image_crawler.crawl_images(keyword=food_name, max_num=20)
        
        # Get list of images
        image_paths = glob.glob(os.path.join(image_dir, '*'))
        
        # Segment each image
        segmented_dir = os.path.join('segmented_images', food_name.replace(' ', '_'))
        for image_path in image_paths:
            # Segment the image
            results = segmentation_model.segment_image(image_path, text=food_name)
            # results is a list; for each value we have: bb, mask, score, class_int, _, _
            # Generate the mask images

            # Load the original image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            image_height, image_width = image.shape[:2]

            # Prepare paths for saving
            image_filename = os.path.basename(image_path)
            image_name, _ = os.path.splitext(image_filename)

            # Create directories if they don't exist
            image_save_dir = os.path.join(segmented_dir, 'images')
            label_save_dir = os.path.join(segmented_dir, 'labels')
            os.makedirs(image_save_dir, exist_ok=True)
            os.makedirs(label_save_dir, exist_ok=True)

            # Copy the image to the segmented_dir/images
            dest_image_path = os.path.join(image_save_dir, image_filename)
            shutil.copy(image_path, dest_image_path)

            # Prepare the label file
            label_path = os.path.join(label_save_dir, f"{image_name}.txt")

            with open(label_path, 'w') as label_file:
                for result in results:
                    print(result)
                    bb, mask, score, class_int, _, _ = result
                    # Check if mask is valid
                    if mask is None or np.sum(mask) == 0:
                        continue

                    # Extract the contours from the mask
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours:
                        # Skip small contours
                        if len(contour) < 6:
                            continue

                        # Flatten contour array and normalize points
                        normalized_points = []
                        for point in contour.squeeze():
                            x, y = point
                            x_norm = x / image_width
                            y_norm = y / image_height
                            normalized_points.extend([x_norm, y_norm])

                        # Write to label file in YOLO format
                        class_id = food_names.index(food_name)  # Get class index
                        label_line = f"{class_id} " + ' '.join(map(str, normalized_points))
                        label_file.write(label_line + '\n')

        # Collect segmented images and labels
        segmented_images = glob.glob(os.path.join(segmented_dir, 'images', '*.jpg'))
        label_files = glob.glob(os.path.join(segmented_dir, 'labels', '*.txt'))
        
        # Split data into training and validation sets
        images_labels = list(zip(segmented_images, label_files))
        train_data, val_data = train_test_split(
            images_labels, test_size=0.2, random_state=42
        )
        
        # Copy images and labels to dataset directory
        for img_path, lbl_path in train_data:
            img_filename = os.path.basename(img_path)
            lbl_filename = os.path.basename(lbl_path)
            shutil.copy(img_path, os.path.join(images_dir, 'train', img_filename))
            shutil.copy(lbl_path, os.path.join(labels_dir, 'train', lbl_filename))
        for img_path, lbl_path in val_data:
            img_filename = os.path.basename(img_path)
            lbl_filename = os.path.basename(lbl_path)
            shutil.copy(img_path, os.path.join(images_dir, 'val', img_filename))
            shutil.copy(lbl_path, os.path.join(labels_dir, 'val', lbl_filename))

    # Create data.yaml file
    data_yaml = {
        'path': dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': [food.replace(' ', '_') for food in food_names]
    }
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
    
    # Fine-tune the segmentation model
    model = SegmentationModel(model_name='yolov8n-seg.pt')
    model.model.train(data=os.path.join(dataset_dir, 'data.yaml'), epochs=10, imgsz=640)

if __name__ == "__main__":
    main()

