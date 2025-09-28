# Skin Cancer Binary Classification | Malignant vs Benign

## About
This project is part of **SifAI**, a dual-model AI system for medical image analysis.     
It focuses on **binary classification** of skin lesions into **Malignant** and **Benign** categories using **deep learning**.    
You can see the other model from skin cancer module from here: [GitHub](https://github.com/gumaruw/Skin-Cancer-SifAI). 

## Dataset 
Skin Cancer ISIC: [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

## Project Structure
project/  
├─ data/           — ISIC dataset    
├─ model/          — Saved/produced model files    
├─ test_results/   — Output predictions & metrics    
├─ venv/           — Python virtual environment    
├─ requirements.txt   
├─ .gitignore        
├─ skin_cancer_binary.py        — Training script 

## Overview
- **Objective:** Automatically detect malignant skin lesions from dermatoscopic images.  
- **Dataset:** ~3,600 labeled images of skin lesions (Malignant vs Benign).  
- **Approach:** Transfer learning with **ResNet50** pre-trained on ImageNet.  
- **Performance:** Achieved **88.03% test accuracy** and **93.88% AUC**.   

## Model Performance
- **Validation Accuracy:** 86.33%  
- **Test Accuracy:** 88.03%  
- **AUC Score:** 93.88%  

## Architecture
- **ResNet50** with Transfer Learning  
- **2-Phase Training Strategy**  
- **Input Size:** 180x180x3  

## Video Demonstration
A video showcasing **model integration into the SifAI platform** is here:  [Google Drive Folder](https://drive.google.com/drive/folders/14x163_HpD7DB1LjPwVKphoIj2z6kkeHR?usp=sharing)

## Model & Sample Images
[Google Drive Folder](https://drive.google.com/drive/folders/1ymdof2t6sQMmGFi84vdFuvEFfhfhMV8h?usp=sharing)

## Screenshots From SifAI
<img width="800" height="393" alt="image" src="https://github.com/user-attachments/assets/d2369b8c-6b3b-4c70-98ce-bee6eb8fbf50" />
<img width="800" height="382" alt="image" src="https://github.com/user-attachments/assets/7438a2a5-8605-40ed-9a9a-a57b9c7d4bc0" />
<img width="800" height="386" alt="image" src="https://github.com/user-attachments/assets/0eeac84f-340a-4246-b780-f4236c46b440" />)
<img width="800" height="363" alt="image" src="https://github.com/user-attachments/assets/b4d3c60d-7494-4b9b-bf16-454057ba8e7d" />
<img width="800" height="385" alt="image" src="https://github.com/user-attachments/assets/440aa743-1826-4b32-bd17-dc49c4409942" />

## Notes
- Designed for research and educational purposes.
- Model can be further improved by increasing dataset size or fine-tuning hyperparameters.

## Classes:
- 0: Benign
- 1: Malignant

## Usage:
```bash
git clone https://github.com/gumaruw/Skin-Cancer-Binary.git
```
```bash
cd Skin-Cancer-Binary
```
```bash
pip install -r requirements.txt
```
