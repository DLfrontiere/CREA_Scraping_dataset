# main.py

import os
import glob
import shutil
import yaml
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch

# Import custom modules (ensure these are available in your project)
from food_database import FoodDatabase
from image_crawler import ImageCrawler
from segmentation_model import (
    YoloSegmentationModel,
    FastSegmentationModel,
    SAMSegmentationModel
)

def download_images_for_class(food_name, image_crawler, dataset_dir, required_images_per_class):
    """
    Checks existing images for a food class and downloads additional images if necessary.

    Args:
        food_name (str): Name of the food class.
        image_crawler (ImageCrawler): Initialized image crawler.
        dataset_dir (str): Root directory for storing images.
        required_images_per_class (int): Number of images required per class.
    
    Returns:
        str: Food name if processed successfully, None otherwise.
    """
    try:
        print(f"Checking images for '{food_name}'...")
        food_dir = os.path.join(dataset_dir, 'images', food_name.replace(' ', '_'))
        existing_images = glob.glob(os.path.join(food_dir, '*'))
        num_existing = len(existing_images)
        print(f"Found {num_existing} images for '{food_name}'.")

        if num_existing < required_images_per_class:
            images_needed = required_images_per_class - num_existing
            print(f"Need to download {images_needed} more images for '{food_name}'.")
            downloaded_dir = image_crawler.crawl_images(
                keyword=food_name,
                max_num=images_needed
            )
            # Optionally, handle post-processing of downloaded images here
        else:
            print(f"'{food_name}' already has sufficient images.")
        
        return food_name
    except Exception as e:
        print(f"Error processing '{food_name}': {e}")
        return None

def create_dataset_multithreaded(database, image_crawler, dataset_dir, required_images_per_class=20, limit_classes=5, max_workers=5):
    """
    Creates the dataset by ensuring each food class has the required number of images.
    Utilizes multithreading to download images concurrently.

    Args:
        database (FoodDatabase): Initialized food database.
        image_crawler (ImageCrawler): Initialized image crawler.
        dataset_dir (str): Root directory for storing images.
        required_images_per_class (int): Number of images required per class.
        limit_classes (int): Number of classes to process (for demonstration).
        max_workers (int): Maximum number of threads to use.
    
    Returns:
        list: List of successfully processed food names.
    """
    food_names = database.fetch_food_names()[:limit_classes]  # Limiting classes for demonstration
    successful_foods = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks for each food class
        future_to_food = {
            executor.submit(
                download_images_for_class,
                food_name,
                image_crawler,
                dataset_dir,
                required_images_per_class
            ): food_name for food_name in food_names
        }

        for future in as_completed(future_to_food):
            food = future_to_food[future]
            result = future.result()
            if result:
                successful_foods.append(result)
            else:
                print(f"Failed to process '{food}'.")

    return successful_foods

def create_dataset(database, image_crawler, dataset_dir, required_images_per_class=20, limit_classes=5):
    """
    Creates the dataset by ensuring each food class has the required number of images.
    If not, it crawls and downloads additional images.

    Args:
        database (FoodDatabase): Initialized food database.
        image_crawler (ImageCrawler): Initialized image crawler.
        dataset_dir (str): Root directory for storing images.
        required_images_per_class (int): Number of images required per class.
        limit_classes (int): Number of classes to process (for demonstration).
    """
    # Original single-threaded implementation
    # Kept here for reference; you can remove it if not needed
    food_names = database.fetch_food_names()[:limit_classes]  # Limiting classes for demonstration

    for food_name in food_names:
        print(f"Checking images for '{food_name}'...")
        food_dir = os.path.join(dataset_dir, 'images', food_name.replace(' ', '_'))
        existing_images = glob.glob(os.path.join(food_dir, '*'))
        num_existing = len(existing_images)
        print(f"Found {num_existing} images for '{food_name}'.")

        if num_existing < required_images_per_class:
            images_needed = required_images_per_class - num_existing
            print(f"Need to download {images_needed} more images for '{food_name}'.")
            downloaded_dir = image_crawler.crawl_images(
                keyword=food_name,
                max_num=images_needed
            )
            # Optionally, handle post-processing of downloaded images here
        else:
            print(f"'{food_name}' already has sufficient images.")

    return food_names

def label_images(segmentation_model, dataset_dir, segmented_dir, food_names):
    """
    Labels images using the segmentation model and organizes them into segmented directories.

    Args:
        segmentation_model: Initialized segmentation model.
        dataset_dir (str): Root directory where images are stored.
        segmented_dir (str): Directory to store segmented images and labels.
        food_names (list): List of food class names.
    """
    for food_name in food_names:
        print(f"Labeling images for '{food_name}'...")
        images_path = os.path.join(dataset_dir, 'images', food_name.replace(' ', '_'))
        image_paths = glob.glob(os.path.join(images_path, '*'))

        for image_path in image_paths:
            # Segment the image
            detections = segmentation_model.seg_image(image_path)
            if detections is None:
                print(f"No detections for {image_path}. Skipping.")
                continue

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
            image_save_dir = os.path.join(segmented_dir, 'images', food_name.replace(' ', '_'))
            label_save_dir = os.path.join(segmented_dir, 'labels', food_name.replace(' ', '_'))
            os.makedirs(image_save_dir, exist_ok=True)
            os.makedirs(label_save_dir, exist_ok=True)

            # Copy the image to the segmented_dir/images
            dest_image_path = os.path.join(image_save_dir, image_filename)
            shutil.copy(image_path, dest_image_path)

            # Prepare the label file
            label_path = os.path.join(label_save_dir, f"{image_name}.txt")

            with open(label_path, 'w') as label_file:
                for detection in detections:
                    boxes = detection.boxes
                    masks = detection.masks
                    if masks is None:
                        continue
                    num_detections = len(boxes)
                    for i in range(num_detections):
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        mask = masks.data[i].cpu().numpy()
                        score = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())

                        if mask is None or np.sum(mask) == 0:
                            continue

                        # Extract contours from the mask
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        for contour in contours:
                            if len(contour) < 6:
                                continue

                            normalized_points = []
                            for point in contour.squeeze():
                                x, y = point
                                x_norm = x / image_width
                                y_norm = y / image_height
                                normalized_points.extend([x_norm, y_norm])

                            # YOLO format: class_index followed by normalized polygon points
                            class_index = food_names.index(food_name)
                            label_line = f"{class_index} " + ' '.join(map(str, normalized_points))
                            label_file.write(label_line + '\n')

def split_and_organize_data(segmented_dir, dataset_dir, test_size=0.2):
    """
    Splits the data into training and validation sets and organizes them accordingly.

    Args:
        segmented_dir (str): Directory containing segmented images and labels.
        dataset_dir (str): Root directory for the final dataset.
        test_size (float): Proportion of the dataset to include in the validation split.
    """
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    for split in ['train', 'val']:
        for dir_path in [images_dir, labels_dir]:
            split_dir = os.path.join(dir_path, split)
            os.makedirs(split_dir, exist_ok=True)

    # Iterate over each food class
    for food_name in os.listdir(os.path.join(segmented_dir, 'images')):
        image_folder = os.path.join(segmented_dir, 'images', food_name)
        label_folder = os.path.join(segmented_dir, 'labels', food_name)

        segmented_images = glob.glob(os.path.join(image_folder, '*'))
        label_files = glob.glob(os.path.join(label_folder, '*.txt'))

        # Ensure that each image has a corresponding label
        images_labels = []
        for img_path in segmented_images:
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(label_folder, f"{image_name}.txt")
            if os.path.exists(lbl_path):
                images_labels.append((img_path, lbl_path))
            else:
                print(f"Label file missing for image: {img_path}. Skipping.")

        if not images_labels:
            print(f"No valid image-label pairs found for '{food_name}'. Skipping split.")
            continue

        train_data, val_data = train_test_split(images_labels, test_size=test_size, random_state=42)

        for img_path, lbl_path in train_data:
            shutil.copy(img_path, os.path.join(images_dir, 'train', os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(labels_dir, 'train', os.path.basename(lbl_path)))

        for img_path, lbl_path in val_data:
            shutil.copy(img_path, os.path.join(images_dir, 'val', os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(labels_dir, 'val', os.path.basename(lbl_path)))

def create_data_yaml(dataset_dir, food_names):
    """
    Creates a data.yaml file for model training.

    Args:
        dataset_dir (str): Root directory of the dataset.
        food_names (list): List of class names.
    """
    data_yaml = {
        'path': dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': [food.replace(' ', '_') for food in food_names]
    }
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
    print("data.yaml file created.")

def train_segmentation_model(dataset_dir, data_yaml_path, model_name='yolov8n-seg.pt', epochs=10, imgsz=640):
    """
    Trains the segmentation model.

    Args:
        dataset_dir (str): Root directory of the dataset.
        data_yaml_path (str): Path to the data.yaml file.
        model_name (str): Pre-trained model to fine-tune.
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
    """
    # Initialize the model
    model = YoloSegmentationModel(model_name=model_name)
    print(f"Training model '{model_name}'...")
    
    # Train the model
    model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz)
    
    # Save the trained model
    trained_model_path = os.path.join(dataset_dir, 'trained_model.pt')
    model.save(trained_model_path)
    print(f"Trained model saved at '{trained_model_path}'.")

def initialize_segmentation_model(model_type='GroundedFastSAM', **kwargs):
    """
    Initializes the segmentation model based on the specified type.

    Args:
        model_type (str): Type of segmentation model to initialize.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        segmentation_model: An instance of the specified segmentation model.
    """
    if model_type == 'GroundedFastSAM':
        segmentation_model = GroundedFastSAM(**kwargs)
    elif model_type == 'YoloSegmentationModel':
        segmentation_model = YoloSegmentationModel(**kwargs)
    elif model_type == 'FastSegmentationModel':
        segmentation_model = FastSegmentationModel(**kwargs)
    elif model_type == 'SAMSegmentationModel':
        segmentation_model = SAMSegmentationModel(**kwargs)
    else:
        raise ValueError(f"Unsupported segmentation model type: {model_type}")
    
    print(f"Initialized segmentation model: {model_type}")
    return segmentation_model

def process_single_image(segmentation_model, image_path, segmented_dir, food_name, food_names):
    """
    Processes a single image: segments it and writes the label.
    
    Args:
        segmentation_model: Initialized segmentation model.
        image_path (str): Path to the image.
        segmented_dir (str): Directory to store segmented images and labels.
        food_name (str): Name of the food class.
        food_names (list): List of all food class names.
    """
    try:
        # Segment the image
        detections = segmentation_model.seg_image(image_path)
        if detections is None:
            print(f"No detections for {image_path}. Skipping.")
            return

        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        image_height, image_width = image.shape[:2]

        # Prepare paths for saving
        image_filename = os.path.basename(image_path)
        image_name, _ = os.path.splitext(image_filename)

        # Create directories if they don't exist
        image_save_dir = os.path.join(segmented_dir, 'images', food_name.replace(' ', '_'))
        label_save_dir = os.path.join(segmented_dir, 'labels', food_name.replace(' ', '_'))
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        # Copy the image to the segmented_dir/images
        dest_image_path = os.path.join(image_save_dir, image_filename)
        shutil.copy(image_path, dest_image_path)

        # Prepare the label file
        label_path = os.path.join(label_save_dir, f"{image_name}.txt")

        with open(label_path, 'w') as label_file:
            for detection in detections:
                boxes = detection.boxes
                masks = detection.masks
                if masks is None:
                    continue
                num_detections = len(boxes)
                for i in range(num_detections):
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    mask = masks.data[i].cpu().numpy()
                    score = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())

                    if mask is None or np.sum(mask) == 0:
                        continue

                    # Extract contours from the mask
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours:
                        if len(contour) < 6:
                            continue

                        normalized_points = []
                        for point in contour.squeeze():
                            x, y = point
                            x_norm = x / image_width
                            y_norm = y / image_height
                            normalized_points.extend([x_norm, y_norm])

                        # YOLO format: class_index followed by normalized polygon points
                        class_index = food_names.index(food_name)
                        label_line = f"{class_index} " + ' '.join(map(str, normalized_points))
                        label_file.write(label_line + '\n')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def label_images_parallel(segmentation_model, dataset_dir, segmented_dir, food_names, max_workers=8):
    """
    Labels images using the segmentation model and organizes them into segmented directories.
    Utilizes multiprocessing to handle multiple images concurrently.

    Args:
        segmentation_model: Initialized segmentation model.
        dataset_dir (str): Root directory where images are stored.
        segmented_dir (str): Directory to store segmented images and labels.
        food_names (list): List of food class names.
        max_workers (int): Number of parallel processes.
    """
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for food_name in food_names:
            print(f"Labeling images for '{food_name}'...")
            images_path = os.path.join(dataset_dir, 'images', food_name.replace(' ', '_'))
            image_paths = glob.glob(os.path.join(images_path, '*'))

            for image_path in image_paths:
                # Submit a separate task for each image
                future = executor.submit(
                    process_single_image,
                    segmentation_model,
                    image_path,
                    segmented_dir,
                    food_name,
                    food_names
                )
                tasks.append(future)

        # Optionally, monitor progress
        for future in as_completed(tasks):
            pass  # You can implement progress tracking here

    print("Labeling completed.")

# Update your main.py accordingly
def main():
    # Configuration
    dataset_dir = 'dataset'
    segmented_dir = 'segmented_images'
    database_path = os.path.join(os.getcwd(), 'nutritional_data.db')
    required_images_per_class = 20

    # Initialize components
    database = FoodDatabase(database_path=database_path)
    image_crawler = ImageCrawler(storage_root=os.path.join(dataset_dir, 'images'))
    segmentation_model = initialize_segmentation_model(model_type='YoloSegmentationModel')

    len_food = len(database.fetch_food_names())
    limit_classes = min(1000, len_food)  # Adjust as needed

    # Step 1: Create the Dataset with Multithreading
    print("\n--- Step 1: Creating the Dataset (Multithreaded) ---")
    food_names = create_dataset_multithreaded(
        database=database,
        image_crawler=image_crawler,
        dataset_dir=dataset_dir,
        required_images_per_class=required_images_per_class,
        limit_classes=limit_classes,
        max_workers=10  # Adjust based on your system's capabilities
    )

    # Proceed only if some food classes were successfully processed
    if not food_names:
        print("No food classes were successfully processed. Exiting.")
        return

    # Step 2: Label the Dataset Images (Parallelized)
    print("\n--- Step 2: Labeling the Dataset Images (Parallelized) ---")
    label_images_parallel(
        segmentation_model=segmentation_model,
        dataset_dir=dataset_dir,
        segmented_dir=segmented_dir,
        food_names=food_names,
        max_workers=8  # Adjust based on your CPU cores
    )

    # Step 2.3: Split and Organize Data
    print("\n--- Step 2.3: Splitting and Organizing Data ---")
    split_and_organize_data(
        segmented_dir=segmented_dir,
        dataset_dir=dataset_dir,
        test_size=0.2
    )

    # Step 3.1: Create data.yaml
    print("\n--- Step 3.1: Creating data.yaml ---")
    create_data_yaml(
        dataset_dir=dataset_dir,
        food_names=food_names
    )
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')

    # Step 3.2: Train the Segmentation Model
    print("\n--- Step 3.2: Training the Segmentation Model ---")
    train_segmentation_model(
        dataset_dir=dataset_dir,
        data_yaml_path=data_yaml_path,
        model_name='yolov8n-seg.pt',  # Replace with your preferred model
        epochs=10,  # Adjust as needed
        imgsz=640  # Adjust as needed
    )

    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
