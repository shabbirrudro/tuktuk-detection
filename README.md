# Tuktuk (Auto-rickshaw) Detection using YOLOv8 and IDD Dataset

A deep learning project to detect auto-rickshaws (tuk-tuks) in Indian street scenes using YOLOv8 and the India Driving Dataset (IDD). This system is optimized for Indian subcontinental traffic conditions and can handle various lighting conditions, occlusions, and complex traffic scenarios.

## Features

- Custom YOLOv8 model trained on IDD dataset
- Optimized for auto-rickshaw detection
- Real-time inference capabilities
- Comprehensive evaluation metrics
- Easy-to-use training and inference pipeline

## Try the pretrained model
```bash
# Clone the repository
git clone https://github.com/shabbirrudro/tuktuk-detection.git
cd tuktuk-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Try the pretrained model
yolo predict model=model/weights.pt source=Path\to\Your\Image.jpg
```

## Train your own model
Download the IDD Segmentation Dataset

1. Register at https://idd.insaan.iiit.ac.in/dataset/details/
2. Download idd-segmentation.tar.gz


```bash
# Clone the repository
git clone https://github.com/shabbirrudro/tuktuk-detection.git
cd tuktuk-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Setup dataset
mkdir -p dataset
# And move the downloaded dataset to this folder

# Convert dataset to YOLO format
python dataset2yolo.py 

# Train model
yolo train data=dataset.yaml model=yolov8n.pt epochs=100

```

## Requirements

```txt
torch==2.2.0+cu121
torchvision==0.17.0+cu121
--extra-index-url https://download.pytorch.org/whl/cu121

pillow==10.2.0
tqdm==4.66.1
torch==2.2.0
ultralytics==8.1.0
numpy>=1.24.0,<2.0.0
opencv-python==4.9.0.80
pandas>=2.0.0
pyyaml>=6.0.1
shapely>=2.0.0
```


Work in Progress