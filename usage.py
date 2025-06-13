# 🔬 Skin Cancer Binary Classification Model
# Usage Guide for Friends/Colleagues
# Model Performance: 88% Test Accuracy, 93.88% AUC

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ================================
# 📥 MODEL LOADING
# ================================

def load_skin_cancer_model(model_path='improved_skin_cancer_model.keras'):
    """
    Trained skin cancer classification model'ini yükle
    
    Args:
        model_path: Model dosyasının yolu
    
    Returns:
        Loaded TensorFlow model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model başarıyla yüklendi!")
        print(f"📊 Model Accuracy: 88.03%")
        print(f"📊 AUC Score: 93.88%")
        return model
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return None

# ================================
# 🖼️ IMAGE PREPROCESSING
# ================================

def preprocess_image(image_path, target_size=(180, 180)):
    """
    Görüntüyü model için hazırla
    
    Args:
        image_path: Görüntü dosyasının yolu
        target_size: Hedef boyut (180, 180)
    
    Returns:
        Preprocessed image array
    """
    try:
        # Görüntüyü yükle
        image = Image.open(image_path)
        
        # RGB'ye çevir (eğer RGBA ise)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Boyutlandır
        image = image.resize(target_size)
        
        # Array'e çevir ve normalize et
        img_array = np.array(image) / 255.0
        
        # Batch dimension ekle
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, image
    
    except Exception as e:
        print(f"❌ Görüntü işlenirken hata: {e}")
        return None, None

# ================================
# 🔮 PREDICTION FUNCTIONS
# ================================

def predict_skin_cancer(model, image_path, show_image=True):
    """
    Tek görüntü üzerinde skin cancer tahmini yap
    
    Args:
        model: Yüklenmiş TensorFlow model
        image_path: Görüntü dosyasının yolu
        show_image: Görüntüyü göster (True/False)
    
    Returns:
        prediction_result: dict with results
    """
    
    # Görüntüyü hazırla
    img_array, original_image = preprocess_image(image_path)
    
    if img_array is None:
        return None
    
    # Tahmin yap
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Sonucu yorumla
    if prediction > 0.5:
        result = "Malignant (Kötü Huylu)"
        confidence = prediction
        risk_level = "⚠️ YÜKSEK RİSK"
        color = 'red'
    else:
        result = "Benign (İyi Huylu)"
        confidence = 1 - prediction
        risk_level = "✅ DÜŞÜK RİSK"
        color = 'green'
    
    # Sonuçları göster
    print(f"\n🔬 SKIN CANCER ANALİZ SONUCU")
    print(f"=" * 40)
    print(f"📊 Tahmin: {result}")
    print(f"📈 Güven: {confidence:.1%}")
    print(f"🎯 Risk Seviyesi: {risk_level}")
    print(f"📋 Raw Score: {prediction:.4f}")
    
    # Görüntüyü göster
    if show_image and original_image:
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image)
        plt.title(f'{result}\nGüven: {confidence:.1%}', 
                 fontsize=14, color=color, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Sonuçları dictionary olarak döndür
    return {
        'prediction': result,
        'confidence': confidence,
        'raw_score': prediction,
        'risk_level': risk_level,
        'is_malignant': prediction > 0.5
    }

def batch_predict(model, image_folder, show_results=True):
    """
    Bir klasördeki tüm görüntüleri analiz et
    
    Args:
        model: Yüklenmiş model
        image_folder: Görüntülerin bulunduğu klasör
        show_results: Sonuçları göster
    
    Returns:
        results: list of prediction results
    """
    
    # Desteklenen dosya formatları
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Klasördeki görüntüleri bul
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print("❌ Klasörde desteklenen görüntü bulunamadı!")
        return []
    
    print(f"📁 {len(image_files)} görüntü bulundu. Analiz başlıyor...")
    
    results = []
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        
        print(f"\n📸 [{i}/{len(image_files)}] Analiz ediliyor: {filename}")
        
        result = predict_skin_cancer(model, image_path, show_image=show_results)
        
        if result:
            result['filename'] = filename
            results.append(result)
    
    # Özet göster
    if results:
        malignant_count = sum(1 for r in results if r['is_malignant'])
        benign_count = len(results) - malignant_count
        
        print(f"\n📊 TOPLU ANALİZ ÖZETİ")
        print(f"=" * 30)
        print(f"📈 Toplam Analiz: {len(results)}")
        print(f"⚠️ Malignant: {malignant_count}")
        print(f"✅ Benign: {benign_count}")
        print(f"📋 Risk Oranı: {malignant_count/len(results):.1%}")
    
    return results

# ================================
# 🎯 MAIN USAGE EXAMPLE
# ================================

def main():
    """Ana kullanım örneği"""
    
    print("🔬 Skin Cancer Classification Model")
    print("=" * 40)
    
    # Model'i yükle
    model = load_skin_cancer_model()
    
    if model is None:
        print("❌ Model yüklenemedi! Dosya yolunu kontrol edin.")
        return
    
    # Tek görüntü analizi örneği
    print("\n1️⃣ TEK GÖRÜNTÜ ANALİZİ:")
    
    # BURAYA KENDİ GÖRÜNTÜ YOLUNUZU YAZIN
    image_path = "path/to/your/skin_image.jpg"
    
    if os.path.exists(image_path):
        result = predict_skin_cancer(model, image_path)
        
        if result:
            # Ek bilgiler
            if result['is_malignant']:
                print("\n⚠️ UYARI: Malignant tahmin edildi!")
                print("💡 Öneri: Dermatolog kontrolü önerilir.")
            else:
                print("\n✅ İyi haber: Benign tahmin edildi.")
                print("💡 Öneri: Düzenli kontroller unutulmasın.")
    else:
        print(f"❌ Görüntü bulunamadı: {image_path}")
    
    # Klasör analizi örneği
    print("\n2️⃣ KLASÖR ANALİZİ:")
    
    folder_path = "path/to/your/images/folder"
    
    if os.path.exists(folder_path):
        results = batch_predict(model, folder_path, show_results=False)
        
        # Sonuçları CSV'ye kaydet
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv('skin_cancer_analysis_results.csv', index=False)
            print("📄 Sonuçlar 'skin_cancer_analysis_results.csv' dosyasına kaydedildi.")
    else:
        print(f"❌ Klasör bulunamadı: {folder_path}")

# ================================
# 🚨 DİSCLAİMER
# ================================

def show_disclaimer():
    """Önemli uyarılar"""
    print("\n" + "="*60)
    print("🚨 ÖNEMLİ UYARILAR VE DİSCLAİMER")
    print("="*60)
    print("⚠️ Bu model sadece eğitim/araştırma amaçlıdır!")
    print("⚠️ Tıbbi tanı koymak için kullanılmamalıdır!")
    print("⚠️ Herhangi bir şüphe durumunda doktora başvurun!")
    print("⚠️ Model %88 doğrulukta, %12 hata payı vardır!")
    print("✅ Sadece yardımcı araç olarak kullanın!")
    print("="*60)

# ================================
# 🚀 KULLANIM
# ================================

if __name__ == "__main__":
    show_disclaimer()
    
    # Ana fonksiyonu çalıştır
    main()
    
    print("\n✅ Analiz tamamlandı! 🎉")

# ================================
# 📋 HIZLI KULLANIM ÖRNEKLERİ
# ================================

"""
🚀 HIZLI BAŞLANGIÇ:

1. Model'i yükle:
   model = load_skin_cancer_model('improved_skin_cancer_model.keras')

2. Tek görüntü analiz et:
   result = predict_skin_cancer(model, 'image.jpg')

3. Klasör analiz et:
   results = batch_predict(model, 'images_folder/')

4. Sonuçları kullan:
   if result['is_malignant']:
       print("Doktora git!")
   else:
       print("Normal görünüyor.")

📊 Model Bilgileri:
- Input Size: 180x180x3
- Classes: Benign (0), Malignant (1)
- Accuracy: 88.03%
- AUC: 93.88%

⚠️ Önemli: Bu sadece yardımcı araçtır, tıbbi tanı değildir!
"""
