# Penguin and Turtle Detection Project

## Project Description
This project implements image classification and object detection for penguins and turtles, comparing multiple deep learning and machine learning approaches.

## Models
- ResNet50: Classification and Object Detection
- DenseNet121: Classification and Object Detection 
- MobileNet: Image Classification
- HOG+SVM: Traditional Machine Learning Approach
- YOLOv5: Object Detection

## Requirements
- Python 3.8+
- PyTorch 1.8+
- OpenCV
- scikit-learn
- See requirements.txt for other dependencies

## File Structure
- `self-built models/`: Custom implemented models
  - `ClassifyModel.py`: Custom classification network
  - `ClassificationTest.py`: Testing script for classification
  - `ClassificationTrain.py`: Training script for classification
  - `DetectionModel.py`: Custom detection network
  - `DetectionTest.py`: Testing script for detection
  - `DetectionTrain.py`: Training script for detection
  - `load_datasets.py`: Dataset loading utilities
  - `readme.txt`: Instructions for model usage
  
- `resnet50.ipynb`: ResNet50 model implementation
- `DenseNet121.ipynb`: DenseNet121 model implementation
- `9517classification mobilenet.ipynb`: MobileNet classifier
- `9517classification hog+svm.ipynb`: HOG+SVM implementation
- `9517 Detect YOLOv5.ipynb`: YOLOv5 object detection

## Usage
1. Clone the repository
```bash
git clone https://github.com/leyin650/turtle-penguin-detection.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare the dataset:
   - Place training images in `archive/train/train/`
   - Place validation images in `archive/valid/valid/`
   - Ensure annotation files are in the correct location

4. Run notebooks:
   - Open desired notebook in Jupyter/VS Code
   - Execute cells sequentially
   - Check results in output cells

## Performance
- Classification Accuracy: ~95%
- Detection IoU: ~0.85
- Detailed metrics available in each notebook

## License
This project is under the MIT License. See LICENSE file for details.