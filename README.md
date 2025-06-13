# Skin Cancer Binary Classification Model

## Model Performance:
- Validation Accuracy: 86.33%
- Test Accuracy: 88.03%
- AUC Score: 93.88%

## Architecture:
- ResNet50 with Transfer Learning
- 2-Phase Training Strategy
- Input Size: 180x180x3

## Usage:
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('improved_skin_cancer_model.keras')

# Make prediction
prediction = model.predict(your_image_array)
result = "Malignant" if prediction > 0.5 else "Benign"
```
## Model
- https://drive.google.com/drive/folders/1ymdof2t6sQMmGFi84vdFuvEFfhfhMV8h?usp=sharing

## Classes:
- 0: Benign
- 1: Malignant
