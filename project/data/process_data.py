import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Dinamik dizin ayarlamaları
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data', 'sensor_data2.xlsx')
processed_data_dir = os.path.join(current_dir, 'data', 'processed')
graphs_dir = os.path.join(current_dir, 'data', 'graphs')
report_path = os.path.join(current_dir, 'data', 'data_processing_report.txt')

os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# Veri setini yükle
df = pd.read_excel(data_path)
print("Veri seti boyutu:", df.shape)
print("Veri seti sütunları:", df.columns.tolist())
print(df.head())

# Eksik değer kontrolü
print("\nEksik değer kontrolü:")
print(df.isnull().sum())

# Aykırı değer kontrolü ve görselleştirme
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(2, 2, i+1)
    plt.boxplot(df[column])
    plt.title(f'{column} Boxplot')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'boxplot_analysis.png'))
plt.close()

# Aykırı değer tespiti
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

outlier_report = {}
for column in df.columns:
    if column != 'Pump Data':
        outliers, lower, upper = detect_outliers(df, column)
        outlier_report[column] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower,
            'upper_bound': upper
        }

# Korelasyon analizi
correlation = df.corr()
print("\nKorelasyon matrisi:")
print(correlation)

plt.figure(figsize=(10, 8))
plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation)), correlation.columns)
for i in range(len(correlation)):
    for j in range(len(correlation)):
        plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', ha='center', va='center',
                 color='white' if abs(correlation.iloc[i, j]) > 0.5 else 'black')
plt.title('Değişkenler Arası Korelasyon')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'correlation_matrix.png'))
plt.close()

# Veri dağılımı
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    if column != 'Pump Data':
        plt.subplot(2, 2, i+1)
        plt.hist(df[column], bins=30, alpha=0.7, color='blue')
        plt.title(f'{column} Histogram')
        plt.xlabel(column)
        plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'histograms.png'))
plt.close()

# Pompa durumu dağılımı
plt.figure(figsize=(15, 10))
for i, column in enumerate(['Soil Moisture', 'Temperature', 'Air Humidity']):
    plt.subplot(1, 3, i+1)
    for pump_value in [0, 1]:
        subset = df[df['Pump Data'] == pump_value]
        plt.hist(subset[column], bins=20, alpha=0.5, label=f'Pump {"On" if pump_value else "Off"}')
    plt.title(f'{column} by Pump Status')
    plt.xlabel(column)
    plt.ylabel('Frekans')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'feature_by_pump_status.png'))
plt.close()

# Eğitim ve test veri setleri
X = df[['Soil Moisture', 'Temperature', 'Air Humidity']]
y = df['Pump Data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save(os.path.join(processed_data_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(processed_data_dir, 'y_train.npy'), y_train.values)
np.save(os.path.join(processed_data_dir, 'y_test.npy'), y_test.values)

with open(os.path.join(processed_data_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Veri işleme raporu
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Veri İşleme Raporu\n\n")
    f.write(f"Veri seti boyutu: {df.shape}\n\n")
    f.write("## Veri Seti Sütunları\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    f.write("\n## Veri Seti İstatistikleri\n")
    f.write(df.describe().to_string())
    f.write("\n\n## Eksik Değer Analizi\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\n## Aykırı Değer Analizi\n")
    for column, stats in outlier_report.items():
        f.write(f"### {column}\n")
        f.write(f"- Aykırı değer sayısı: {stats['count']} ({stats['percentage']:.2f}%)\n")
        f.write(f"- Alt sınır: {stats['lower_bound']:.2f}\n")
        f.write(f"- Üst sınır: {stats['upper_bound']:.2f}\n\n")
    f.write("\n## Korelasyon Analizi\n")
    f.write(correlation.to_string())
    f.write("\n\n## Veri Bölme\n")
    f.write(f"- Eğitim seti: {X_train.shape[0]} örnek ({X_train.shape[0]/df.shape[0]*100:.2f}%)\n")
    f.write(f"- Test seti: {X_test.shape[0]} örnek ({X_test.shape[0]/df.shape[0]*100:.2f}%)\n")
    f.write("\n\n## Veri Ölçeklendirme\n")
    f.write("StandardScaler kullanılarak veriler ölçeklendirildi.\n")
    f.write(f"- Ortalama değerler: {scaler.mean_}\n")
    f.write(f"- Standart sapmalar: {scaler.scale_}\n")

print("Veri işleme tamamlandı. İşlenmiş veriler ve raporlar kaydedildi.")
