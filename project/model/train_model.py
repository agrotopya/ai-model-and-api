import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Dinamik dizin ayarlamaları
current_dir = os.getcwd()
processed_data_dir = os.path.join(current_dir, 'data', 'processed')
model_dir = os.path.join(current_dir, 'model')

# İşlenmiş verileri yükle
X_train = np.load(os.path.join(processed_data_dir, 'X_train.npy'))
X_test = np.load(os.path.join(processed_data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(processed_data_dir, 'y_train.npy'))
y_test = np.load(os.path.join(processed_data_dir, 'y_test.npy'))

# Model sonuçlarını saklamak için sözlük
model_results = {}

# Modelleri değerlendirmek için fonksiyon
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Tahminler yap
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Sonuçları kaydet
    results = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    # Sonuçları yazdır
    print(f"\n{model_name} Sonuçları:")
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Duyarlılık (Recall): {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    print("Karmaşıklık Matrisi:")
    print(cm)
    
    return results

# Modelleri oluştur ve değerlendir
print("Model eğitimi ve değerlendirmesi başlıyor...")

# 1. Lojistik Regresyon
lr_model = LogisticRegression(max_iter=1000, random_state=42)
model_results['Logistic Regression'] = evaluate_model(lr_model, "Lojistik Regresyon", X_train, X_test, y_train, y_test)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
model_results['Random Forest'] = evaluate_model(rf_model, "Random Forest", X_train, X_test, y_train, y_test)

# 3. Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_results['Gradient Boosting'] = evaluate_model(gb_model, "Gradient Boosting", X_train, X_test, y_train, y_test)

# 4. Support Vector Machine
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
model_results['SVM'] = evaluate_model(svm_model, "Support Vector Machine", X_train, X_test, y_train, y_test)

# 5. Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
model_results['Neural Network'] = evaluate_model(nn_model, "Neural Network", X_train, X_test, y_train, y_test)

# En iyi modeli bul
best_model_name = max(model_results, key=lambda k: model_results[k]['f1_score'])
best_model = model_results[best_model_name]['model']
best_accuracy = model_results[best_model_name]['accuracy']
best_f1 = model_results[best_model_name]['f1_score']

print(f"\nEn iyi model: {best_model_name}")
print(f"Doğruluk: {best_accuracy:.4f}")
print(f"F1 Skoru: {best_f1:.4f}")

# En iyi modeli optimize et
print("\nEn iyi modeli optimize ediyorum...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'newton-cg']
    }
    grid_model = LogisticRegression(random_state=42)
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_model = RandomForestClassifier(random_state=42)
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_model = GradientBoostingClassifier(random_state=42)
    
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    grid_model = SVC(probability=True, random_state=42)
    
else:  # Neural Network
    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (10, 5), (20, 10)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    grid_model = MLPClassifier(max_iter=1000, random_state=42)

# Grid Search ile en iyi parametreleri bul
grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi cross-validation skoru: {grid_search.best_score_:.4f}")

# En iyi parametrelerle modeli yeniden eğit
optimized_model = grid_search.best_estimator_
optimized_model.fit(X_train, y_train)

# Test seti üzerinde değerlendir
y_pred = optimized_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
final_precision = precision_score(y_test, y_pred)
final_recall = recall_score(y_test, y_pred)
final_f1 = f1_score(y_test, y_pred)
final_cm = confusion_matrix(y_test, y_pred)

print("\nOptimize Edilmiş Model Test Sonuçları:")
print(f"Doğruluk (Accuracy): {final_accuracy:.4f}")
print(f"Kesinlik (Precision): {final_precision:.4f}")
print(f"Duyarlılık (Recall): {final_recall:.4f}")
print(f"F1 Skoru: {final_f1:.4f}")
print("Karmaşıklık Matrisi:")
print(final_cm)

# Karmaşıklık matrisini görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pompa Kapalı', 'Pompa Açık'],
            yticklabels=['Pompa Kapalı', 'Pompa Açık'])
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Karmaşıklık Matrisi')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))

plt.close()

# Özellik önemini görselleştir (Random Forest veya Gradient Boosting için)
if hasattr(optimized_model, 'feature_importances_'):
    # Orijinal özellik isimlerini al
    feature_names = ['Toprak Nemi', 'Sıcaklık', 'Hava Nemi']
    
    # Özellik önemlerini hesapla
    importances = optimized_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Görselleştir
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Özellikler')
    plt.ylabel('Önem')
    plt.title('Özellik Önemleri')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importance.png'))

    plt.close()
    
    # Özellik önemlerini yazdır
    print("\nÖzellik Önemleri:")
    for i, idx in enumerate(indices):
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# Model klasörünü oluştur
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, 'model')

# Optimize edilmiş modeli kaydet
with open(f'{model_dir}/agrotopya_model.pkl', 'wb') as f:
    pickle.dump(optimized_model, f)

# Ölçeklendirme modelini yükle
with open(f'{processed_data_dir}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Ölçeklendirme modelini model klasörüne de kaydet
with open(f'{model_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Model raporu oluştur
with open(f'{model_dir}/model_report.txt', 'w') as f:
    f.write("# Agrotopya Yapay Zeka Modeli Raporu\n\n")
    
    f.write("## Model Karşılaştırma Sonuçları\n\n")
    f.write("| Model | Doğruluk | Kesinlik | Duyarlılık | F1 Skoru |\n")
    f.write("|-------|----------|----------|------------|----------|\n")
    
    for model_name, results in model_results.items():
        f.write(f"| {model_name} | {results['accuracy']:.4f} | {results['precision']:.4f} | {results['recall']:.4f} | {results['f1_score']:.4f} |\n")
    
    f.write(f"\n## En İyi Model: {best_model_name}\n\n")
    f.write(f"En iyi parametreler: {grid_search.best_params_}\n\n")
    
    f.write("## Optimize Edilmiş Model Test Sonuçları\n\n")
    f.write(f"- Doğruluk (Accuracy): {final_accuracy:.4f}\n")
    f.write(f"- Kesinlik (Precision): {final_precision:.4f}\n")
    f.write(f"- Duyarlılık (Recall): {final_recall:.4f}\n")
    f.write(f"- F1 Skoru: {final_f1:.4f}\n\n")
    
    f.write("### Karmaşıklık Matrisi\n\n")
    f.write("```\n")
    f.write(f"{final_cm}\n")
    f.write("```\n\n")
    
    if hasattr(optimized_model, 'feature_importances_'):
        f.write("### Özellik Önemleri\n\n")
        for i, idx in enumerate(indices):
            f.write(f"- {feature_names[idx]}: {importances[idx]:.4f}\n")
    
    f.write("\n## Model Kullanımı\n\n")
    f.write("Bu model, toprak nemi, sıcaklık ve hava nemi verilerine dayanarak sulama pompasının açılıp açılmaması gerektiğini tahmin eder.\n")
    f.write("Modelin kullanımı için önce sensör verilerinin ölçeklendirilmesi, ardından modelin tahmin fonksiyonunun çağrılması gerekir.\n\n")
    f.write("Örnek kullanım:\n")
    f.write("```python\n")
    f.write("import pickle\n\n")
    f.write("# Modeli ve ölçeklendiriciyi yükle\n")
    f.write("with open('agrotopya_model.pkl', 'rb') as f:\n")
    f.write("    model = pickle.load(f)\n\n")
    f.write("with open('scaler.pkl', 'rb') as f:\n")
    f.write("    scaler = pickle.load(f)\n\n")
    f.write("# Yeni veri\n")
    f.write("new_data = [[soil_moisture, temperature, air_humidity]]\n\n")
    f.write("# Veriyi ölçeklendir\n")
    f.write("scaled_data = scaler.transform(new_data)\n\n")
    f.write("# Tahmin yap\n")
    f.write("prediction = model.predict(scaled_data)\n")
    f.write("# prediction[0] == 1 ise pompa açık, 0 ise pompa kapalı\n")
    f.write("```\n")

print(f"\nModel eğitimi tamamlandı ve '{model_dir}' klasörüne kaydedildi.")
print(f"Model raporu '{model_dir}/model_report.txt' dosyasına kaydedildi.")
