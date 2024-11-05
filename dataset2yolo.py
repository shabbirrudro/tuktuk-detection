"""
This script converts the IDD dataset into YOLO format
"""
import json
import os
from PIL import Image
import shutil
from tqdm import tqdm

def download_and_extract():
    """
    Note: You'll need to manually download the dataset first from:
    https://idd.insaan.iiit.ac.in/ after registration
    """
    # Create necessary directories
    dirs = ['dataset/images/train', 'dataset/images/val', 
            'dataset/labels/train', 'dataset/labels/val']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert IDD bbox to YOLO format"""
    x, y, width, height = bbox
    
    # Convert to YOLO format (normalized coordinates)
    x_center = (x + width/2) / img_width
    y_center = (y + height/2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return [x_center, y_center, norm_width, norm_height]

def convert_annotations(ann_file, output_dir, images_dir):
    """Convert IDD JSON annotations to YOLO format"""
    # IDD class mapping (modify based on your needs)
    class_mapping = {
        'auto_rickshaw': 0,
        'car': 1,
        'bus': 2,
        'truck': 3,
        # Add other classes as needed
    }
    
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Process each image
    for image_info in tqdm(annotations['images']):
        image_id = image_info['id']
        img_width = image_info['width']
        img_height = image_info['height']
        
        # Get annotations for this image
        image_anns = [ann for ann in annotations['annotations'] 
                     if ann['image_id'] == image_id]
        
        # Create YOLO format label file
        label_file = os.path.join(output_dir, 
                                f"{os.path.splitext(image_info['file_name'])[0]}.txt")
        
        with open(label_file, 'w') as f:
            for ann in image_anns:
                category_name = next(cat['name'] for cat in annotations['categories'] 
                                  if cat['id'] == ann['category_id'])
                
                if category_name in class_mapping:
                    class_id = class_mapping[category_name]
                    bbox = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
                    f.write(f"{class_id} {' '.join([str(x) for x in bbox])}\n")

def prepare_dataset():
    """Main function to prepare the dataset"""
    # 1. Create directory structure
    download_and_extract()
    
    # 2. Convert annotations
    convert_annotations(
        'path/to/idd_annotations_train.json',
        'dataset/labels/train',
        'dataset/images/train'
    )
    convert_annotations(
        'path/to/idd_annotations_val.json',
        'dataset/labels/val',
        'dataset/images/val'
    )
    
    # 3. Create dataset.yaml
    yaml_content = """
path: ./dataset  # dataset root dir
train: images/train  # train images
val: images/val  # val images

# Classes
nc: 4  # number of classes
names: ['auto_rickshaw', 'car', 'bus', 'truck']  # class names
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    prepare_dataset()