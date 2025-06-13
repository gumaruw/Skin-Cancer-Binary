# Skin Cancer Binary Classification Model

- 📊 Dataset: Malignant vs Benign (3,600 total images)
- 🤖 Model: ResNet50 with 2-Phase Training
- 📈 Validation Accuracy: 0.8633 (86.33%)
- 🎯 Target Achievement: ✅ SUCCESS
- 🧪 Test Accuracy: 0.8955 (89.55%)

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

## Dataset
- https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

## Model & Sample Images
- https://drive.google.com/drive/folders/1ymdof2t6sQMmGFi84vdFuvEFfhfhMV8h?usp=sharing

## Classes:
- 0: Benign
- 1: Malignant
