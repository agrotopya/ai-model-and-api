import pandas as pd
import matplotlib.pyplot as plt
import os


# Dinamik dizin ayarlamaları
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data', 'sensor_data.xlsx')
analysis_output_path = os.path.join(current_dir, 'data', 'data_analysis.txt')
graphs_dir = os.path.join(current_dir, 'data', 'graphs')

os.makedirs(graphs_dir, exist_ok=True)

# Excel dosyasını oku
df = pd.read_excel(excel_path)

# Veri setinin genel bilgilerini yazdır
print("Veri seti boyutu:", df.shape)
print("\nVeri seti sütunları:")
print(df.columns.tolist())
print("\nVeri seti örnek veriler:")
print(df.head())
print("\nVeri seti istatistikleri:")
print(df.describe())
print("\nEksik veriler:")
print(df.isnull().sum())

# Veri setinin yapısını kaydet
with open(analysis_output_path, 'w', encoding='utf-8') as f:
    f.write("Veri Seti Analizi\n")
    f.write("=================\n\n")
    f.write(f"Veri seti boyutu: {df.shape}\n\n")
    f.write("Veri seti sütunları:\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    f.write("\nVeri seti örnek veriler:\n")
    f.write(df.head().to_string())
    f.write("\n\nVeri seti istatistikleri:\n")
    f.write(df.describe().to_string())
    f.write("\n\nEksik veriler:\n")
    f.write(df.isnull().sum().to_string())


# Sayısal sütunlar için histogramlar oluştur
numeric_columns = df.select_dtypes(include=['number']).columns
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[col].dropna(), bins=20, alpha=0.7)
    plt.title(f'{col} Dağılımı')
    plt.xlabel(col)
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(graphs_dir, f'{col}_histogram.png'))
    plt.close()

# Eğer zaman serisi verisi varsa, zaman içindeki değişimi göster
time_columns = [col for col in df.columns if 'tarih' in col.lower() or 'zaman' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
if time_columns:
    time_col = time_columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(by=time_col)
    
    for col in numeric_columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df[time_col], df[col], marker='o', linestyle='-', alpha=0.7)
        plt.title(f'{col} Zaman İçindeki Değişimi')
        plt.xlabel(time_col)
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f'{col}_time_series.png'))
        plt.close()

# Korelasyon analizi
if len(numeric_columns) > 1:
    plt.figure(figsize=(12, 10))
    corr = df[numeric_columns].corr()
    plt.matshow(corr, fignum=1)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(i, j, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                     color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'correlation_matrix.png'))
    plt.close()

print("Analiz tamamlandı. Sonuçlar 'data' klasörüne kaydedildi.")

