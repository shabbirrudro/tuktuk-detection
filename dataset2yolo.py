import json
import os
from PIL import Image
import shutil
from tqdm import tqdm
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
import tarfile
import io

def create_yolo_folders():
    """Create YOLO format directory structure"""
    dirs = ['dataset/images/train', 'dataset/images/val', 'dataset/images/test',
            'dataset/labels/train', 'dataset/labels/val']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def get_bbox_from_polygon(polygon_points, img_width, img_height):
    """Convert polygon points to YOLO format bounding box"""
    polygon = Polygon(polygon_points)
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    
    # Convert to YOLO format (x_center, y_center, width, height)
    x_center = (bounds[0] + bounds[2]) / 2 / img_width
    y_center = (bounds[1] + bounds[3]) / 2 / img_height
    width = (bounds[2] - bounds[0]) / img_width
    height = (bounds[3] - bounds[1]) / img_height
    
    return [x_center, y_center, width, height]

def process_archive(archive_path):
    """Process IDD dataset directly from tar.gz archive"""
    # Class mapping
    class_mapping = {
        'autorickshaw': 0,
        'car': 1,
        'motorcycle': 2,
        'rider': 3,
        'truck': 4,
        'bus': 5,
        'vehicle fallback': 6,
        # Add other classes as needed
    }
    
    print("Creating directory structure...")
    create_yolo_folders()
    
    print(f"Processing archive: {archive_path}")
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Get all files in archive
        all_files = tar.getmembers()
        
        # Process images first to get dimensions
        image_info = {}
        for member in tqdm(all_files, desc="Processing images"):
            if not member.name.endswith('_leftImg8bit.png'):
                continue
                
            # Extract split type (train/val/test) and filename
            parts = member.name.split('/')
            if len(parts) < 4 or 'leftImg8bit' not in parts:
                continue
                
            split = parts[2]  # train/val/test
            if split == 'test':
                # Extract test images without labels
                f = tar.extractfile(member)
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data))
                
                # Save image
                img_filename = f"{parts[3]}_{parts[4]}"
                img_path = os.path.join('dataset/images/test', img_filename)
                img.save(img_path)
                continue
                
            # Store image dimensions and save image
            f = tar.extractfile(member)
            img_data = f.read()
            img = Image.open(io.BytesIO(img_data))
            img_width, img_height = img.size
            
            # Save image
            img_filename = f"{parts[3]}_{parts[4]}"
            img_path = os.path.join(f'dataset/images/{split}', img_filename)
            img.save(img_path)
            
            # Store image info for label conversion
            json_name = member.name.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
            image_info[json_name] = {
                'width': img_width,
                'height': img_height,
                'split': split,
                'img_filename': img_filename
            }
        
        # Process labels
        for member in tqdm(all_files, desc="Processing labels"):
            if not member.name.endswith('_polygons.json'):
                continue
                
            if member.name not in image_info:
                continue
                
            info = image_info[member.name]
            split = info['split']
            img_width = info['width']
            img_height = info['height']
            
            # Read JSON data
            f = tar.extractfile(member)
            label_data = json.load(f)
            
            # Create YOLO format label file
            label_filename = info['img_filename'].replace('.png', '.txt')
            label_path = os.path.join(f'dataset/labels/{split}', label_filename)
            
            # Convert annotations
            with open(label_path, 'w') as f:
                for obj in label_data['objects']:
                    if obj['label'] in class_mapping and not obj['deleted']:
                        class_id = class_mapping[obj['label']]
                        # Convert polygon points to list of coordinates
                        polygon_points = np.array(obj['polygon'])
                        bbox = get_bbox_from_polygon(polygon_points, img_width, img_height)
                        
                        # Write YOLO format line
                        f.write(f"{class_id} {' '.join([str(x) for x in bbox])}\n")
    
    print("Creating dataset.yaml...")
    # Create dataset.yaml
    yaml_content = """
path: ./dataset  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images

# Classes
nc: 7  # number of classes
names: ['autorickshaw', 'car', 'motorcycle', 'rider', 'truck', 'bus', 'vehicle fallback']  # class names
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    archive_path = "dataset/idd-segmentation.tar.gz"
    process_archive(archive_path)