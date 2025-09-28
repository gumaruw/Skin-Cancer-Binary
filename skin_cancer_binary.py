import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
import warnings
warnings.filterwarnings('ignore')

print("TF version:", tf.__version__)
print("GPU ready:", tf.config.list_physical_devices('GPU'))

# GPU memory growth 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# DATA LOAD
dataset_path = '/kaggle/input/skin-cancer-malignant-vs-benign'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

print(f"Dataset Path: {dataset_path}")
print(f"Train Path: {train_path}")
print(f"Test Path: {test_path}")

# sınıflar
class_names = []
if os.path.exists(train_path):
    class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"📋 Sınıflar: {class_names}")
    
    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {class_name}: {count} görüntü")


# Image params - daha küçük boyut deniyorum çünkü GPU memory hatası aldım
IMG_HEIGHT = 180  
IMG_WIDTH = 180   
BATCH_SIZE = 16   
VALIDATION_SPLIT = 0.15  

# aggresivve augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=40,         
    width_shift_range=0.3,          
    height_shift_range=0.3,         
    shear_range=0.3,            
    zoom_range=0.3,             
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3], 
    channel_shift_range=0.2,    
    fill_mode='nearest'
)

# Validation için minimal preprocessing
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# Test data preprocessing
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

print("Preparing data generators...")

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode='binary',
    subset='training', shuffle=True, seed=42
)

validation_generator = val_datagen.flow_from_directory(
    train_path, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode='binary',
    subset='validation', shuffle=False, seed=42
)

test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode='binary',
    shuffle=False
) if os.path.exists(test_path) else None

print("Train samples:", train_generator.samples)
print("Val samples:", validation_generator.samples)
if test_generator:
    print("Test samples:", test_generator.samples)

print("Class indices:", train_generator.class_indices)

# Class weights (dengesizlik varsa)
y_train = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)


def create_efficientnet():
    
    # Base model
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # İlk katmanları dondur, son katmanları açık bırak
    base_model.trainable = True
    
    # Fine-tuning için son katmanları train edilebilir yap
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_resnet_model():
    
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = True

    fine_tune_at = len(base_model.layers) - 15
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_mobilenet_model():
    
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = True
    
    # Fine-tuning
    fine_tune_at = len(base_model.layers) - 10
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# EfficientNet yerine ResNet denedim
print("Creating model...")
model = create_resnet_model()  # ResNet50 
# model = create_efficientnet()  # EfficientNet 
# model = create_mobilenet_model()  # MobileNet 

model.summary()

initial_learning_rate = 0.0001 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # val_loss yerine val_accuracy
        patience=8,              # Daha fazla patience
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,             # Daha agresif reduction
        patience=4,             # Daha az patience
        min_lr=1e-8,
        verbose=1,
        mode='max'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_skin_cancer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# training

print("Phase 1: Training classifier layers only...")

for layer in model.layers[:-6]:  # base modeli son 6 katman hariç dondur
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ],
    verbose=1
)

# İkinci aşama: Fine-tuning (çok düşük lr)
print("\nPhase 2: fine-tuning full model")

for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

history_phase2 = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("training completed.")

# Historyleri birleştir
def combine_histories(hist1, hist2):
    combined = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key]
    return combined

combined_history = combine_histories(history_phase1, history_phase2)

# viz
def plot_training_history(history_dict):
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    axes[0,0].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0,0].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0,0].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target (80%)')
    axes[0,0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0, 1])
    
    # Loss
    axes[0,1].plot(history_dict['loss'], label='Training Loss', linewidth=2)
    axes[0,1].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    axes[0,1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history_dict:
        axes[1,0].plot(history_dict['auc'], label='Training AUC', linewidth=2)
        axes[1,0].plot(history_dict['val_auc'], label='Validation AUC', linewidth=2)
        axes[1,0].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target (0.8)')
        axes[1,0].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('AUC')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim([0, 1])
    
    # Learning Rate
    axes[1,1].plot(range(len(history_dict['loss'])), [0.001]*10 + [0.0001]*len(history_dict['loss'][10:]), 
                   label='Learning Rate', linewidth=2, color='orange')
    axes[1,1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_yscale('log')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(combined_history)

# eval

def comprehensive_evaluation(model, generator, title="Evaluation"):
    
    print(f"\n {title}")
    generator.reset()
    predictions = model.predict(generator, verbose=1)
    predicted_probs = predictions.flatten()
    predicted_classes = (predicted_probs > 0.5).astype(int)
    true_classes = generator.classes[:len(predicted_classes)]
    target_names = ['Benign', 'Malignant']
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=target_names, output_dict=True)
    print(classification_report(true_classes, predicted_classes, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{title} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Yüzdelik 
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'{cm_percent[i, j]:.1%}', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()
    
    # ROC 
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(true_classes, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    return accuracy, precision, recall, f1, roc_auc

val_metrics = comprehensive_evaluation(model, validation_generator, "Validation Set")

if test_generator:
    test_metrics = comprehensive_evaluation(model, test_generator, "Test Set")

# save
model.save('skin_cancer_model.keras')
model.save_weights('skin_cancer_weights.weights.h5')
print("Model & weights saved")

# summary
print("\nFINAL RESULTS")
print(f"Validation Acc: {val_metrics[0]:.4f}")
if test_generator:
    print(f"Test Acc: {test_metrics[0]:.4f}")