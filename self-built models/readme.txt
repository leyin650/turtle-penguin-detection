# Self-built Models Implementation

## File Structure
- `load_datasets.py`: Dataset loading and preprocessing utilities
- `ClassifyModel.py`: Custom classification network architecture
- `DetectionModel.py`: Custom detection network architecture
- `ClassificationTrain.py`: Training script for classification model
- `DetectionTrain.py`: Training script for detection model
- `ClassificationTest.py`: Evaluation script for classification model
- `DetectionTest.py`: Evaluation script for detection model

## Model Outputs
- `classifyModel.h5`: Best performing classification model
- `detectionModel.h5`: Best performing detection model

## Dataset Structure
```
datasets/
└── Penguins_vs_Turtles/
    ├── train/
    ├── valid/
    ├── train_annotations
    └── valid_annotations
```

## Usage Instructions
1. Prepare Environment:
   - Place all Python files in the same directory
   - Create `datasets` folder following above structure

2. Training:
   ```bash
   python ClassificationTrain.py  # Train classification model
   python DetectionTrain.py      # Train detection model
   ```

3. Testing:
   ```bash
   python ClassificationTest.py  # Evaluate classification model
   python DetectionTest.py      # Evaluate detection model
   ```

## Notes
- Models will automatically save best performing weights during training
- Test scripts will load the saved models for evaluation
- Ensure dataset paths are correctly configured in `load_datasets.py`