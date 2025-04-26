
# Agrotopya Yapay Zeka Modeli
Bu proje, **tarımsal sulama ihtiyacını tahmin etmek** amacıyla geliştirilen bir yapay zeka modelini içermektedir. Model, toprak nemi, sıcaklık ve hava nemi gibi sensör verilerine dayanarak sulama pompasının açılıp açılmaması gerektiğini tahmin etmektedir.

---

## Proje Yapısı

```bash
project/
├── api/
│   ├── app.py               # Flask API uygulaması
│   ├── test_api.py           # API test scripti
│   ├── templates/            # Web arayüzü şablon dosyaları (HTML)
│   └── static/               # CSS, JS ve Swagger dökümanları
├── data/
│   ├── sensor_data2.xlsx     # Ana veri seti (model eğitimi için)
│   ├── processed/            # İşlenmiş veriler (npy dosyaları ve scaler)
│   ├── graphs/               # Veri analiz grafikleri
│   └── data_processing_report.txt  # Veri analiz ve işleme raporu
├── model/
│   ├── agrotopya_model.pkl   # Eğitilmiş en iyi yapay zeka modeli
│   ├── scaler.pkl            # Eğitimde kullanılan ölçekleyici (StandardScaler)
│   ├── model_report.txt      # Model karşılaştırmaları ve detaylı rapor
│   ├── confusion_matrix.png  # Modelin karmaşıklık matrisi
│   └── feature_importance.png # Özellik önemi grafiği (varsa)
├── process_data.py           # Veriyi okuma, işleme ve kaydetme scripti
├── analyze_excel.py          # Veri seti üzerinde analiz ve görselleştirme scripti
├── train_model.py            # Model eğitme ve en iyi modeli seçme scripti
└── README.md                 # Proje dökümantasyonu (bu dosya)
```

---

## Kullanılan Teknolojiler
- **Python 3.10+**
- **Flask** - API geliştirme
- **Scikit-learn** - Makine öğrenmesi modelleri
- **Matplotlib** ve **Seaborn** - Grafik ve görselleştirme
- **Pandas**, **NumPy** - Veri işleme
- **GridSearchCV** - Model hiperparametre optimizasyonu

---

## Ana Özellikler
- 📈 Eğitim veri seti ile **en iyi modelin** otomatik seçilmesi ve optimize edilmesi
- 🧪 API üzerinden tahmin yapabilme (`/predict`)
- 📜 Simülasyon ve test verisi üretme (`/simulate`)
- 🧹 Sensör verilerinin temizlenmesi ve ölçeklendirilmesi
- 📊 İstatistiksel analizler (`/stats`) ve geçmiş kaydı (`/history`)
- ⚙️ Raspberry Pi cihazlarından sensör verisi alımı

---

## Kurulum ve Kullanım

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Veriyi işleyin:
   ```bash
   python process_data.py
   ```

3. Modeli eğitin:
   ```bash
   python train_model.py
   ```

4. API'yi çalıştırın:
   ```bash
   cd api
   python app.py
   ```

5. Testleri çalıştırın:
   ```bash
   python test_api.py
   ```

---

## Notlar
- Model ve scaler dosyaları `model/` klasöründe saklanmaktadır.
- API için Swagger dokümantasyonu ve canlı test imkanı sunulmaktadır.
- Test verisi için `test_results.xlsx` gibi sonuç dosyaları da oluşturulmuştur.
