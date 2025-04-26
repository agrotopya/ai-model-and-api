import os
import sys
import json
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pickle
import logging
from datetime import datetime

# windows için uyumlu yol tanımlamaları
template_folder = os.path.join(os.getcwd(), 'api', 'templates')
static_folder = os.path.join(os.getcwd(), 'api', 'static')
model_folder = os.path.join(os.getcwd(), 'model')
log_folder = os.path.join(os.getcwd(), 'api')

# Klasör yapısını oluştur
os.makedirs(template_folder, exist_ok=True)
os.makedirs(os.path.join(static_folder, 'css'), exist_ok=True)
os.makedirs(os.path.join(static_folder, 'js'), exist_ok=True)

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_folder, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('agrotopya_api')

# Flask app başlat
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
# Model ve scaler'ı yükle
MODEL_PATH = os.path.join(model_folder, 'agrotopya_model.pkl')
SCALER_PATH = os.path.join(model_folder, 'scaler.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model başarıyla yüklendi.")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler başarıyla yüklendi.")
except Exception as e:
    logger.error(f"Model veya scaler yüklenirken hata oluştu: {str(e)}")
    model = None
    scaler = None

# Sensör verilerinin geçmişini tutmak için basit bir veritabanı
# Gerçek uygulamada bu bir veritabanı olacaktır
sensor_history = []
MAX_HISTORY_SIZE = 10000  # Maksimum kayıt sayısı

# Web arayüzü için ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# API dokümantasyonu
@app.route('/docs')
def api_docs():
    return render_template('docs.html')

# Swagger UI için JSON şeması
@app.route('/swagger.json')
def swagger_json():
    swagger_path = os.path.join(static_folder, 'swagger.json')
    return send_from_directory(os.path.dirname(swagger_path), os.path.basename(swagger_path))

@app.route('/health', methods=['GET'])
def health_check():
    """API sağlık kontrolü"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Sensör verilerine göre sulama tahmini yapar
    
    Beklenen JSON formatı:
    {
        "soil_moisture": float,
        "temperature": float,
        "air_humidity": float
    }
    
    Dönüş değeri:
    {
        "prediction": 0 veya 1 (0: Sulama yok, 1: Sulama gerekli),
        "probability": float,
        "timestamp": datetime
    }
    """
    try:
        # İstek verilerini al
        data = request.get_json()
        logger.info(f"Gelen istek: {data}")
        
        # Gerekli alanları kontrol et
        required_fields = ['soil_moisture', 'temperature', 'air_humidity']
        for field in required_fields:
            if field not in data:
                logger.error(f"Eksik alan: {field}")
                return jsonify({
                    'error': f"Eksik alan: {field}",
                    'required_fields': required_fields
                }), 400
        
        # Verileri hazırla
        soil_moisture = float(data['soil_moisture'])
        temperature = float(data['temperature'])
        air_humidity = float(data['air_humidity'])
        
        # Değerlerin makul aralıkta olup olmadığını kontrol et
        if not (300 <= soil_moisture <= 1000):
            logger.warning(f"Toprak nemi değeri normal aralık dışında: {soil_moisture}")
        
        if not (15 <= temperature <= 40):
            logger.warning(f"Sıcaklık değeri normal aralık dışında: {temperature}")
        
        if not (30 <= air_humidity <= 90):
            logger.warning(f"Hava nemi değeri normal aralık dışında: {air_humidity}")
        
        # Verileri ölçeklendir
        input_data = np.array([[soil_moisture, temperature, air_humidity]])
        scaled_data = scaler.transform(input_data)
        
        # Tahmin yap
        prediction = int(model.predict(scaled_data)[0])
        
        # Olasılık hesapla (sınıf 1 olasılığı)
        probability = float(model.predict_proba(scaled_data)[0][1])
        
        # Zaman damgası ekle
        timestamp = datetime.now().isoformat()
        
        # Sonucu hazırla
        result = {
            'prediction': prediction,
            'probability': probability,
            'timestamp': timestamp,
            'message': 'Sulama gerekli' if prediction == 1 else 'Sulama gerekli değil'
        }
        
        # Sensör verilerini ve tahmini kaydet
        sensor_record = {
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'air_humidity': air_humidity,
            'prediction': prediction,
            'probability': probability,
            'timestamp': timestamp
        }
        
        # Geçmiş verileri güncelle
        sensor_history.append(sensor_record)
        if len(sensor_history) > MAX_HISTORY_SIZE:
            sensor_history.pop(0)  # En eski kaydı sil
        
        logger.info(f"Tahmin sonucu: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Tahmin sırasında hata oluştu: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Tahmin sırasında bir hata oluştu'
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Sensör verilerinin geçmişini döndürür
    
    Query parametreleri:
    - limit: Döndürülecek maksimum kayıt sayısı (varsayılan: 100)
    - offset: Atlanacak kayıt sayısı (varsayılan: 0)
    """
    try:
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        # Limit ve offset değerlerini doğrula
        if limit <= 0 or limit > 1000:
            limit = 100
        
        if offset < 0:
            offset = 0
        
        # Geçmiş verileri döndür
        result = {
            'history': sensor_history[offset:offset+limit],
            'total_records': len(sensor_history),
            'limit': limit,
            'offset': offset
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Geçmiş verileri alırken hata oluştu: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Geçmiş verileri alırken bir hata oluştu'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Sensör verilerinin istatistiklerini döndürür
    """
    try:
        if not sensor_history:
            return jsonify({
                'message': 'Henüz sensör verisi bulunmuyor'
            })
        
        # Verileri numpy dizilerine dönüştür
        soil_moisture_values = np.array([record['soil_moisture'] for record in sensor_history])
        temperature_values = np.array([record['temperature'] for record in sensor_history])
        air_humidity_values = np.array([record['air_humidity'] for record in sensor_history])
        prediction_values = np.array([record['prediction'] for record in sensor_history])
        
        # İstatistikleri hesapla
        stats = {
            'soil_moisture': {
                'min': float(np.min(soil_moisture_values)),
                'max': float(np.max(soil_moisture_values)),
                'mean': float(np.mean(soil_moisture_values)),
                'median': float(np.median(soil_moisture_values)),
                'std': float(np.std(soil_moisture_values))
            },
            'temperature': {
                'min': float(np.min(temperature_values)),
                'max': float(np.max(temperature_values)),
                'mean': float(np.mean(temperature_values)),
                'median': float(np.median(temperature_values)),
                'std': float(np.std(temperature_values))
            },
            'air_humidity': {
                'min': float(np.min(air_humidity_values)),
                'max': float(np.max(air_humidity_values)),
                'mean': float(np.mean(air_humidity_values)),
                'median': float(np.median(air_humidity_values)),
                'std': float(np.std(air_humidity_values))
            },
            'predictions': {
                'irrigation_count': int(np.sum(prediction_values)),
                'no_irrigation_count': int(len(prediction_values) - np.sum(prediction_values)),
                'irrigation_percentage': float(np.mean(prediction_values) * 100)
            },
            'total_records': len(sensor_history),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"İstatistikleri hesaplarken hata oluştu: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'İstatistikleri hesaplarken bir hata oluştu'
        }), 500

@app.route('/simulate', methods=['POST'])
def simulate_sensor():
    """
    Test amaçlı sensör verisi simülasyonu yapar
    
    Beklenen JSON formatı:
    {
        "count": int,  # Oluşturulacak veri sayısı
        "soil_moisture_range": [min, max],  # Opsiyonel
        "temperature_range": [min, max],  # Opsiyonel
        "air_humidity_range": [min, max]  # Opsiyonel
    }
    """
    try:
        data = request.get_json()
        count = data.get('count', 1)
        
        # Maksimum simülasyon sayısını sınırla
        if count > 100:
            count = 100
        
        # Varsayılan değer aralıkları
        soil_moisture_range = data.get('soil_moisture_range', [400, 900])
        temperature_range = data.get('temperature_range', [20, 35])
        air_humidity_range = data.get('air_humidity_range', [40, 80])
        
        results = []
        
        for _ in range(count):
            # Rastgele sensör değerleri oluştur
            soil_moisture = np.random.uniform(soil_moisture_range[0], soil_moisture_range[1])
            temperature = np.random.uniform(temperature_range[0], temperature_range[1])
            air_humidity = np.random.uniform(air_humidity_range[0], air_humidity_range[1])
            
            # Verileri ölçeklendir
            input_data = np.array([[soil_moisture, temperature, air_humidity]])
            scaled_data = scaler.transform(input_data)
            
            # Tahmin yap
            prediction = int(model.predict(scaled_data)[0])
            probability = float(model.predict_proba(scaled_data)[0][1])
            
            # Zaman damgası ekle
            timestamp = datetime.now().isoformat()
            
            # Sonucu hazırla
            result = {
                'soil_moisture': float(soil_moisture),
                'temperature': float(temperature),
                'air_humidity': float(air_humidity),
                'prediction': prediction,
                'probability': probability,
                'timestamp': timestamp,
                'message': 'Sulama gerekli' if prediction == 1 else 'Sulama gerekli değil'
            }
            
            # Sensör verilerini ve tahmini kaydet
            sensor_history.append(result)
            if len(sensor_history) > MAX_HISTORY_SIZE:
                sensor_history.pop(0)  # En eski kaydı sil
            
            results.append(result)
        
        return jsonify({
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Simülasyon sırasında hata oluştu: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Simülasyon sırasında bir hata oluştu'
        }), 500

# Raspberry Pi sensör verilerini almak için endpoint
@app.route('/sensor/data', methods=['POST'])
def receive_sensor_data():
    """
    Raspberry Pi'dan gelen sensör verilerini alır ve işler
    
    Beklenen JSON formatı:
    {
        "device_id": string,
        "soil_moisture": float,
        "temperature": float,
        "air_humidity": float,
        "timestamp": string (ISO format)
    }
    """
    try:
        data = request.get_json()
        logger.info(f"Raspberry Pi'dan gelen sensör verisi: {data}")
        
        # Gerekli alanları kontrol et
        required_fields = ['device_id', 'soil_moisture', 'temperature', 'air_humidity']
        for field in required_fields:
            if field not in data:
                logger.error(f"Eksik alan: {field}")
                return jsonify({
                    'error': f"Eksik alan: {field}",
                    'required_fields': required_fields
                }), 400
        
        # Verileri hazırla
        device_id = data['device_id']
        soil_moisture = float(data['soil_moisture'])
        temperature = float(data['temperature'])
        air_humidity = float(data['air_humidity'])
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Verileri ölçeklendir
        input_data = np.array([[soil_moisture, temperature, air_humidity]])
        scaled_data = scaler.transform(input_data)
        
        # Tahmin yap
        prediction = int(model.predict(scaled_data)[0])
        probability = float(model.predict_proba(scaled_data)[0][1])
        
        # Sonucu hazırla
        result = {
            'device_id': device_id,
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'air_humidity': air_humidity,
            'prediction': prediction,
            'probability': probability,
            'timestamp': timestamp,
            'message': 'Sulama gerekli' if prediction == 1 else 'Sulama gerekli değil'
        }
        
        # Sensör verilerini ve tahmini kaydet
        sensor_history.append(result)
        if len(sensor_history) > MAX_HISTORY_SIZE:
            sensor_history.pop(0)  # En eski kaydı sil
        
        logger.info(f"Sensör verisi işlendi, tahmin: {prediction}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Sensör verisi işlenirken hata oluştu: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Sensör verisi işlenirken bir hata oluştu'
        }), 500

# Mobil uygulama için komut gönderme endpoint'i
@app.route('/device/command', methods=['POST'])
def send_device_command():
    """
    Mobil uygulamadan cihaza komut gönderir
    
    Beklenen JSON formatı:
    {
        "device_id": string,
        "command": string,
        "parameters": object
    }
    """
    try:
        data = request.get_json()
        logger.info(f"Mobil uygulamadan gelen komut: {data}")
        
        # Gerekli alanları kontrol et
        required_fields = ['device_id', 'command']
        for field in required_fields:
            if field not in data:
                logger.error(f"Eksik alan: {field}")
                return jsonify({
                    'error': f"Eksik alan: {field}",
                    'required_fields': required_fields
                }), 400
        
        # Komut verilerini hazırla
        device_id = data['device_id']
        command = data['command']
        parameters = data.get('parameters', {})
        
        # Gerçek uygulamada burada cihaza komut gönderme işlemi yapılır
        # Şimdilik sadece simüle ediyoruz
        
        # Sonucu hazırla
        result = {
            'device_id': device_id,
            'command': command,
            'parameters': parameters,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'message': f"Komut '{command}' cihaza gönderildi"
        }
        
        logger.info(f"Komut gönderildi: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Komut gönderilirken hata oluştu: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Komut gönderilirken bir hata oluştu'
        }), 500

if __name__ == '__main__':
    # API'yi başlat
    logger.info("Agrotopya API başlatılıyor...")
    app.run(host='0.0.0.0', port=5000, debug=True)
