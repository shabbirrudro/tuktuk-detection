# Tuk-tuk Detection using YOLOv8

This project implements an object detection system specifically designed to identify and locate tuk-tuks (auto-rickshaws) in images from the Indian subcontinent using YOLOv8. The system can be used for various applications including traffic monitoring, urban planning, and transportation studies.

## Features

- Custom YOLOv8 model training for tuk-tuk detection
- Real-time inference on images
- Result visualization with bounding boxes and confidence scores
- Comprehensive evaluation metrics
- Easy-to-use Python API

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tuktuk-detector.git
cd tuktuk-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Option 1: Create Your Own Dataset

1. Collect images of tuk-tuks from various sources
2. Annotate the images using tools like LabelImg or CVAT
3. Convert annotations to YOLO format
4. Organize the dataset as follows:
```
tuktuk_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### Option 2: Use Existing Datasets

You can use existing datasets such as:
- Open Images Dataset (filter for auto-rickshaw)
- Indian Vehicles Dataset
- Custom collected datasets

## Usage

### Training

```python
from tuktuk_detector import TukTukDetector

# Initialize detector
detector = TukTukDetector('tuktuk_dataset')

# Train model
results = detector.train_model(epochs=100)
```

### Inference

```python
# Make predictions on a single image
detector.visualize_results('path/to/test_image.jpg')
```

## Model Architecture

The system uses YOLOv8, which offers several advantages:
- Fast inference speed
- High accuracy
- Good balance between performance and computational requirements

## Project Structure

```
tuktuk-detector/
├── requirements.txt
├── train_tuktuk_detector.py
├── tuktuk_dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── README.md
```

## Training Tips

1. **Data Collection**:
   - Collect images in various lighting conditions
   - Include different tuk-tuk models and colors
   - Capture various angles and distances
   - Include crowded scenes

2. **Training Parameters**:
   - Start with the default parameters
   - Adjust batch size based on your GPU memory
   - Use learning rate scheduling for better convergence

3. **Validation**:
   - Regularly monitor training progress
   - Check for overfitting
   - Validate on diverse test cases

## Evaluation Metrics

The system provides several evaluation metrics:
- mAP (mean Average Precision)
- Precision
- Recall
- F1-Score

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 team for the base detection framework
- Ultralytics for their excellent implementation
- Contributors to the training datasets

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/tuktuk-detector

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{tuktuk_detector,
  author = {Your Name},
  title = {Tuk-tuk Detection using YOLOv8},
  year = {2024},
  url = {https://github.com/yourusername/tuktuk-detector}
}
```