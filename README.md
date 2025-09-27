# Skin Cancer Classification: Malignant vs Benign

This project focuses on **binary classification** of skin lesions into **Malignant** and **Benign** categories using **deep learning**.

## Overview
- **Objective:** Automatically detect malignant skin lesions from dermatoscopic images.  
- **Dataset:** ~3,600 labeled images of skin lesions (Malignant vs Benign).  
- **Approach:** Transfer learning with **ResNet50** pre-trained on ImageNet.  
- **Performance:** Achieved **88.03% test accuracy** and **93.88% AUC**.  

## Features
- Image preprocessing and augmentation to improve model generalization.  
- Binary classification pipeline using convolutional neural networks.  
- Evaluation metrics include **accuracy**, **AUC**, and **confusion matrix** visualization.  

## Model Performance
- **Validation Accuracy:** 86.33%  
- **Test Accuracy:** 88.03%  
- **AUC Score:** 93.88%  

## Technology Stack
- **Python 3.10+**  
- **TensorFlow / Keras**  
- **OpenCV, NumPy, Pandas**  
- **Matplotlib & Seaborn** for visualization  


## Architecture
- **ResNet50** with Transfer Learning  
- **2-Phase Training Strategy**  
- **Input Size:** 180x180x3  


## Usage:
```bash
git clone https://github.com/gumaruw/Skin-Cancer-Malignant-vs-Benign.git
cd Skin-Cancer-Malignant-vs-Benign
```
```bash
pip install -r requirements.txt
```

## Resources
Dataset: [Skin Cancer Malignant vs Benign Dataset on Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)  

Model & Sample Images: [Google Drive Folder](https://drive.google.com/drive/folders/1ymdof2t6sQMmGFi84vdFuvEFfhfhMV8h?usp=sharing)

## Notes

- Designed for research and educational purposes.
- Model can be further improved by increasing dataset size or fine-tuning hyperparameters.

## Classes:
- 0: Benign
- 1: Malignant
