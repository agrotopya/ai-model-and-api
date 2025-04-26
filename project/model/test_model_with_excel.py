import pandas as pd
import pickle
import numpy as np

# Model ve scaler yükle
with open(r'C:\Users\burak\OneDrive\Masaüstü\AGROTOPYA\veriler\forreal\Yapay Zeka Modeli Geliştirme Projesi\project\model\agrotopya_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'C:\Users\burak\OneDrive\Masaüstü\AGROTOPYA\veriler\forreal\Yapay Zeka Modeli Geliştirme Projesi\project\model\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Test verisini oku
test_df = pd.read_excel(r'C:\Users\burak\OneDrive\Masaüstü\AGROTOPYA\veriler\forreal\Yapay Zeka Modeli Geliştirme Projesi\testdata\test_synthetic.xlsx')

# Gerekli kolonları seç
X_test = test_df[['Soil Moisture', 'Temperature', 'Air Humidity']]

# Ölçeklendir
X_test_scaled = scaler.transform(X_test)

# Tahmin yap
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Sonuçları dataframe'e ekle
test_df['Prediction'] = predictions
test_df['Probability'] = probabilities

# Sonuçları yazdır
print(test_df[['Soil Moisture', 'Temperature', 'Air Humidity', 'Prediction', 'Probability']])

# Sonuçları kaydet
test_df.to_excel(r'C:\Users\burak\OneDrive\Masaüstü\AGROTOPYA\veriler\forreal\Yapay Zeka Modeli Geliştirme Projesi\test_results3.xlsx', index=False)
print("Sonuçlar 'test_results3.xlsx' dosyasına kaydedildi.")
